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


def write_pixels_to_dicom(ds, cleaned_array):
    """
    Correctly writes the anonymized pixel array back into the DICOM dataset.
    Updates ALL mandatory pixel descriptor tags to match the array's dtype/shape.
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

    # Update all pixel descriptor tags
    ds.BitsAllocated = bits_alloc
    ds.BitsStored = bits_stored
    ds.HighBit = high_bit
    ds.PixelRepresentation = pix_rep

    # Uncompressed (Explicit VR Little Endian), never encapsulated: the
    # transfer syntax and the PixelData bytes must always agree.
    _set_pixel_data(ds, arr)

    return ds
