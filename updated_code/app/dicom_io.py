"""
dicom_io.py — Stage 7: write anonymized pixels back into the DICOM dataset
with correct pixel-descriptor tag updates.
"""

import numpy as np

from pydicom.uid import ExplicitVRLittleEndian


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
        bits_alloc = 8; bits_stored = 8; high_bit = 7; pix_rep = 0
    elif arr.dtype == np.uint16:
        bits_alloc = 16; bits_stored = 16; high_bit = 15; pix_rep = 0
    elif arr.dtype == np.int16:
        bits_alloc = 16; bits_stored = 16; high_bit = 15; pix_rep = 1
    else:
        # Fallback: cast to uint16
        arr = arr.astype(np.uint16)
        bits_alloc = 16; bits_stored = 16; high_bit = 15; pix_rep = 0

    # Update all pixel descriptor tags
    ds.BitsAllocated      = bits_alloc
    ds.BitsStored         = bits_stored
    ds.HighBit            = high_bit
    ds.PixelRepresentation = pix_rep

    # Ensure uncompressed transfer syntax (Explicit VR Little Endian)
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_implicit_VR  = False
    ds.is_little_endian = True

    # Write pixel data
    ds.PixelData = arr.tobytes()

    return ds
