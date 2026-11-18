"""
dicom_io.py — reading compressed pixel data, and Stage 7: writing anonymized
pixels back into the DICOM dataset with matching pixel-descriptor tags.
"""

import numpy as np

from pydicom.uid import ExplicitVRLittleEndian


def decompress_pixels(ds):
    """Decode compressed Pixel Data in place, before any pixel is touched.

    Returns the original transfer syntax UID if it was compressed (JPEG
    Lossless, JPEG baseline, JPEG-LS, JPEG 2000, RLE), else None. Afterwards
    the dataset is Explicit VR Little Endian and PixelData holds the raw
    pixels, so it can be redacted and saved as-is.

    Without this, the redacted pixels would be written as raw bytes into a
    file whose header still says JPEG: a corrupt file.

    Raises RuntimeError naming the transfer syntax when no installed decoder
    (pylibjpeg, pylibjpeg-libjpeg/-openjpeg/-rle) can decode it.
    """
    meta = getattr(ds, "file_meta", None)
    ts = getattr(meta, "TransferSyntaxUID", None) if meta is not None else None
    if "PixelData" not in ds or ts is None or not ts.is_compressed:
        return None
    try:
        ds.decompress()
        arr = ds.pixel_array
    except Exception as e:
        raise RuntimeError(
            f"cannot decode pixel data in transfer syntax {ts} ({ts.name}): {e}"
        ) from e
    # pydicom 2.x decompress() switches the transfer syntax but leaves the
    # encapsulated JPEG bytes in PixelData. Replace them now, so the dataset
    # is consistent even on paths that never write pixels back.
    _set_pixel_data(ds, arr)
    return ts


def _set_pixel_data(ds, arr):
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    ds.PixelData = arr.tobytes()
    elem = ds["PixelData"]
    elem.is_undefined_length = False  # was encapsulated if the source was compressed
    elem.VR = "OB" if ds.BitsAllocated <= 8 else "OW"


def _fits(arr, bits, signed):
    if arr.size == 0:
        return True
    lo, hi = int(arr.min()), int(arr.max())
    if signed:
        return -(2 ** (bits - 1)) <= lo and hi < 2 ** (bits - 1)
    return 0 <= lo and hi < 2**bits


def render_for_display(arr, ds):
    """8-bit rendering of a frame the way a viewer shows it: Modality LUT
    (rescale), the file's first window (else the full value range), then
    MONOCHROME1 inverted so bone is white."""
    a = arr.astype(np.float64)
    slope = float(ds.get("RescaleSlope", 1) or 1)
    intercept = float(ds.get("RescaleIntercept", 0) or 0)
    a = a * slope + intercept
    center, width = ds.get("WindowCenter"), ds.get("WindowWidth")
    if center is not None and width is not None:
        center = float(
            center[0]
            if hasattr(center, "__len__") and not isinstance(center, str)
            else center
        )
        width = float(
            width[0]
            if hasattr(width, "__len__") and not isinstance(width, str)
            else width
        )
        lo, hi = center - width / 2, center + width / 2
    else:
        lo, hi = float(a.min()), float(a.max())
    if hi <= lo:
        hi = lo + 1
    out = (np.clip((a - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)
    if str(ds.get("PhotometricInterpretation", "")) == "MONOCHROME1":
        out = 255 - out
    return out


def write_pixels_to_dicom(ds, cleaned_array):
    """
    Correctly writes the anonymized pixel array back into the DICOM dataset.
    Sets the pixel descriptor tags from the array's dtype/shape, keeping the
    original BitsStored/HighBit whenever the values still fit in them.
    """
    arr = cleaned_array

    # Handle multi-frame: if shape is (frames, H, W) flatten is not needed;
    # pydicom handles this via NumberOfFrames
    if arr.ndim == 3 and arr.shape[0] > 1:
        # Multi-frame: (F, H, W)
        ds.NumberOfFrames = arr.shape[0]
    elif arr.ndim == 2:
        pass  # single frame

    # Determine bit depth from dtype
    if arr.dtype == np.uint8:
        bits_alloc = 8
        bits_stored = 8
        high_bit = 7
        pix_rep = 0
    elif arr.dtype == np.uint16:
        bits_alloc = 16
        bits_stored = 16
        high_bit = 15
        pix_rep = 0
    elif arr.dtype == np.int16:
        bits_alloc = 16
        bits_stored = 16
        high_bit = 15
        pix_rep = 1
    else:
        # Fallback: cast to uint16
        arr = arr.astype(np.uint16)
        bits_alloc = 16
        bits_stored = 16
        high_bit = 15
        pix_rep = 0

    # Keep the original BitsStored/HighBit when the redacted pixels still fit
    # in them. Viewers take the value range from BitsStored (and invert a
    # MONOCHROME1 image against it), so rewriting a 12-bit image as 16-bit
    # changes how it displays even though no pixel value changed.
    orig_stored = ds.get("BitsStored")
    if (
        orig_stored
        and ds.get("BitsAllocated") == bits_alloc
        and ds.get("PixelRepresentation", pix_rep) == pix_rep
        and 0 < int(orig_stored) < bits_alloc
        and _fits(arr, int(orig_stored), pix_rep)
    ):
        bits_stored = int(orig_stored)
        high_bit = bits_stored - 1

    # Update all pixel descriptor tags
    ds.BitsAllocated = bits_alloc
    ds.BitsStored = bits_stored
    ds.HighBit = high_bit
    ds.PixelRepresentation = pix_rep

    # Uncompressed (Explicit VR Little Endian), never encapsulated: the
    # transfer syntax and the PixelData bytes must always agree.
    _set_pixel_data(ds, arr)

    return ds
