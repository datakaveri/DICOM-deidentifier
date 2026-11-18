"""Compressed pixel data is decoded before redaction and written back raw.

The PACS X-rays this runs on are JPEG Lossless (Process 14 SV1,
1.2.840.10008.1.2.4.70). The codestreams below are synthetic gradients
encoded with GDCM, so the tests need no .dcm file and no patient data.
"""

import base64

import numpy as np
import pydicom
import pytest
from pydicom.data import get_testdata_file
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.encaps import encapsulate
from pydicom.uid import (
    ExplicitVRLittleEndian,
    JPEGLosslessSV1,
    SecondaryCaptureImageStorage,
    generate_uid,
)

from dicom_io import decompress_pixels, write_pixels_to_dicom

# 32x24, 8-bit: pixel (y, x) = (7x + 3y) % 256
JPEG_LOSSLESS_8BIT = (
    "/9j/wwALCAAYACABAREA/8QAFgABAQEAAAAAAAAAAAAAAAAAAwII/9oACAEBAAEAAM/u7u7u7u7u"
    "7u7u7u7u7u7vbu7u7u7u7u7u7u7u7u7u727u7u7u7u7u7u7u7u7u7u9u7u7u7u7u7u7u7u7u7u7v"
    "bu7u7u7u7u7u7u7u7u7u727u7u7u7u7u7u7u7u7u7u9u7u7u7u7u7u7u7u7u7u7vbu7u7u7u7u7u"
    "7u7u7u7u727u7u7u7u7u7u7u7u7u7u9u7u7u7u7u7u7u7u7u7u7vbu7u7u7u7u7u7u7u7u7u727u"
    "7u7u7u7u7u7u7u7u7u9u7u7u7u7u7u7u7u7u7u7vbu7u7u7u7u7u7u7u7u7u+Brd3d3d3d3d3d3d"
    "3d3d3d3wNbu7u7u7u7u7u7u7u7u7u+Brd3d3d3d3d3d3d3d3d3d8DPbu7u7u7u7u7u7u7u7u7vgZ"
    "7d3d3d3d3d3d3d3d3d3fAzvbu7u7u7u7u7u7u7u7u74Gd7d3d3d3d3d3d3d3d3d3wM7vbu7u7u7u"
    "7u7u7u7u7u+Bnd7d3d3d3d3d3d3d3d3d3wM7vbu7u7u7u7u7u7u7u7vgZ3d//9k="
)
# 32x24, 12 of 16 bits: pixel (y, x) = (131x + 17y) % 4096
JPEG_LOSSLESS_12BIT = (
    "/9j/wwALEAAYACABAREA/8QAFwABAQEBAAAAAAAAAAAAAAAACAUMEP/aAAgBAQABAADkGg0Gg0Gg"
    "0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0HRQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQ"
    "aDQaDQaDQdFBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNB0UGg0Gg0Gg0Gg0Gg0Gg"
    "0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg8EFRQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaD"
    "wQVFBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoPBBUUGg0Gg0Gg0Gg0Gg0Gg0Gg0Gg"
    "0Gg0Gg0Gg0Gg0Gg0Gg0Gg8EFRQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDwQVFBo"
    "NBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoPBBUUGg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg"
    "0Gg0Gg0Gg0Gg0Gg8EFRQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQeCCQdFBoNBoNBo"
    "NBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNB4IJB0UGg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg"
    "0Gg0Gg0HggkHRQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQeCCQdFBoNBoNBoNBoNBo"
    "NBoNBoNBoNBoNBoNBoNBoNBoNBoNB4IJB0UGg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg"
    "0HggkHRQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQeCCQdFBoNBoNBoNBoNBoNBoNBo"
    "NBoNBoNBoNBoNBoNBoNBoNB4IJB0UGg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg8EEg0H"
    "RQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDwQSDQdFBoNBoNBoNBoNBoNBoNBoNBoNBo"
    "NBoNBoNBoNBoNBoPBBINB0UGg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg0Gg8EEg0HRQaDQa"
    "DQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDQaDwQSDQdFBoNBoNBoNBoNBoNBoNBoNBoNBoNBoNBo"
    "NBoNBoNBoPBBINB//9k="
)


def gradient(bits):
    y, x = np.mgrid[0:24, 0:32]
    if bits == 8:
        return ((x * 7 + y * 3) % 256).astype(np.uint8)
    return ((x * 131 + y * 17) % 4096).astype(np.uint16)


def jpeg_lossless_dataset(bits):
    codestream = base64.b64decode(
        JPEG_LOSSLESS_8BIT if bits == 8 else JPEG_LOSSLESS_12BIT
    )
    ds = Dataset()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = JPEGLosslessSV1
    ds.file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID
    ds.Rows, ds.Columns, ds.SamplesPerPixel = 24, 32, 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = 8 if bits == 8 else 16
    ds.BitsStored, ds.HighBit, ds.PixelRepresentation = bits, bits - 1, 0
    ds.PixelData = encapsulate([codestream])
    ds["PixelData"].is_undefined_length = True
    ds["PixelData"].VR = "OB"
    ds.is_implicit_VR, ds.is_little_endian = False, True
    return ds


def reread(ds, tmp_path):
    path = tmp_path / "out.dcm"
    ds.save_as(path, write_like_original=False)
    return pydicom.dcmread(path)


@pytest.mark.parametrize("bits", [8, 12])
def test_jpeg_lossless_decodes_exactly_and_saves_uncompressed(bits, tmp_path):
    ds = jpeg_lossless_dataset(bits)
    assert decompress_pixels(ds) == JPEGLosslessSV1
    assert np.array_equal(ds.pixel_array, gradient(bits))

    out = reread(ds, tmp_path)
    assert out.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert not out["PixelData"].is_undefined_length
    assert np.array_equal(out.pixel_array, gradient(bits))


@pytest.mark.parametrize("bits", [8, 12])
def test_redacted_pixels_written_after_decompress_read_back(bits, tmp_path):
    ds = jpeg_lossless_dataset(bits)
    decompress_pixels(ds)
    redacted = ds.pixel_array.copy()
    redacted[2:10, 4:20] = 0
    out = reread(write_pixels_to_dicom(ds, redacted), tmp_path)
    assert out.file_meta.TransferSyntaxUID == ExplicitVRLittleEndian
    assert np.array_equal(out.pixel_array, redacted)


@pytest.mark.parametrize(
    "name", ["MR_small_jpeg_ls_lossless.dcm", "MR_small_jp2klossless.dcm"]
)
def test_other_lossless_codecs_match_the_uncompressed_reference(name, tmp_path):
    reference = pydicom.dcmread(get_testdata_file("MR_small.dcm")).pixel_array
    ds = pydicom.dcmread(get_testdata_file(name))
    assert decompress_pixels(ds) is not None
    assert np.array_equal(reread(ds, tmp_path).pixel_array, reference)


def test_uncompressed_input_is_left_alone():
    ds = pydicom.dcmread(get_testdata_file("MR_small.dcm"))
    before = ds.PixelData
    assert decompress_pixels(ds) is None
    assert ds.PixelData is before


def test_no_pixel_data_is_left_alone():
    ds = jpeg_lossless_dataset(8)
    del ds.PixelData
    assert decompress_pixels(ds) is None
    assert ds.file_meta.TransferSyntaxUID == JPEGLosslessSV1


def test_missing_decoder_fails_naming_the_transfer_syntax(monkeypatch):
    ds = jpeg_lossless_dataset(8)

    def no_decoder(*args, **kwargs):
        raise NotImplementedError("no handler")

    monkeypatch.setattr(ds, "decompress", no_decoder)
    with pytest.raises(RuntimeError, match=r"1\.2\.840\.10008\.1\.2\.4\.70"):
        decompress_pixels(ds)


def test_write_back_keeps_12_bit_depth(tmp_path):
    # A 12-bit image rewritten as 16-bit displays differently: viewers take
    # the value range (and MONOCHROME1 inversion) from BitsStored.
    ds = jpeg_lossless_dataset(12)
    decompress_pixels(ds)
    redacted = ds.pixel_array.copy()
    redacted[2:10, 4:20] = redacted.min()
    out = reread(write_pixels_to_dicom(ds, redacted), tmp_path)
    assert (out.BitsAllocated, out.BitsStored, out.HighBit) == (16, 12, 11)
    assert np.array_equal(out.pixel_array, redacted)


def test_write_back_widens_only_when_values_no_longer_fit(tmp_path):
    ds = jpeg_lossless_dataset(12)
    decompress_pixels(ds)
    arr = ds.pixel_array.copy()
    arr[0, 0] = 5000  # beyond 12 bits
    out = reread(write_pixels_to_dicom(ds, arr), tmp_path)
    assert out.BitsStored == 16
    assert out.pixel_array[0, 0] == 5000


def test_preview_follows_window_and_inverts_monochrome1():
    from dicom_io import render_for_display

    ds = Dataset()
    ds.PhotometricInterpretation = "MONOCHROME1"
    ds.WindowCenter, ds.WindowWidth = 2047, 4095
    arr = np.array([[0, 4095]], dtype=np.uint16)
    img = render_for_display(arr, ds)
    assert img[0, 0] == 255 and img[0, 1] == 0, "MONOCHROME1: low values are white"
    ds.PhotometricInterpretation = "MONOCHROME2"
    img = render_for_display(arr, ds)
    assert img[0, 0] == 0 and img[0, 1] == 255
