"""
Contract section 4: `verify` produces pixel_verification_status, and a file
whose verification FAILS must still be written, with a clearly failed status.

The user needs to see that redaction was incomplete -- not receive nothing and
no explanation.
"""

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

import pipeline
from de_identification.actions import RunSecrets
from de_identification.job_config import parse_job_config
from de_identification.keystore import KeyStore


@pytest.fixture
def dicom_with_pixels(tmp_path):
    ds = Dataset()
    ds.PatientName = "Doe^Jane"
    ds.PatientID = "UHID0012345"
    ds.StudyDescription = "CHEST PA"
    ds.Modality = "CR"
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = CTImageStorage
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()

    ds.Rows, ds.Columns = 32, 32
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.full((32, 32), 1000, dtype=np.uint16).tobytes()

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta

    path = tmp_path / "in.dcm"
    ds.save_as(str(path), write_like_original=False)
    return str(path)


@pytest.fixture
def stub_detection(monkeypatch):
    """Stubs out the OCR/NLP stages so the test exercises the pipeline's
    control flow, not the models."""
    region = {"text": "DOE^JANE", "bbox": [4, 4, 20, 12], "zone": "border"}
    monkeypatch.setattr(pipeline, "detect_text_regions", lambda *a, **k: [[4, 4, 20, 12]])
    monkeypatch.setattr(pipeline, "detect_text_in_regions", lambda *a, **k: [region])
    monkeypatch.setattr(pipeline, "merge_detections", lambda det: [region])
    monkeypatch.setattr(pipeline, "classify_phi", lambda *a, **k: [region])
    monkeypatch.setattr(pipeline, "expand_phi_blocks", lambda m, p, s: [region])
    monkeypatch.setattr(pipeline, "match_against_stored_tags", lambda *a, **k: [])
    monkeypatch.setattr(pipeline, "load_original_tag_values", lambda p: {})
    return region


def _config():
    return parse_job_config({
        "operations": ["dicom_deidentify", "suppress"],
        "dicom_deidentify": {"default_action": "keep", "tag_actions": {
            "PatientName": {"action": "suppress"},
        }},
    })


def test_a_file_failing_verification_is_still_written(
        tmp_path, dicom_with_pixels, stub_detection, monkeypatch):
    monkeypatch.setattr(
        pipeline, "verify_redaction",
        lambda arr, *a, **k: (arr, "FAILED"),
    )

    out = tmp_path / "after.dcm"
    audit = pipeline.anonymize_dicom_file(
        dicom_with_pixels, str(tmp_path / "before.dcm"), str(out),
        str(tmp_path / "data.json"), None, None, None,
        KeyStore(str(tmp_path / "k.json")),
        job_config=_config(), secrets=RunSecrets(),
    )

    assert audit["verification_status"] == "FAILED"
    assert audit["error"] is None
    assert out.exists(), "a failed verification must still produce a file"

    written = pydicom.dcmread(str(out))
    assert str(written.PatientName) == "", "the tag policy still ran"
    assert str(written.StudyDescription) == "CHEST PA"


def test_a_passing_file_reports_passed(
        tmp_path, dicom_with_pixels, stub_detection, monkeypatch):
    monkeypatch.setattr(
        pipeline, "verify_redaction",
        lambda arr, *a, **k: (arr, "PASSED"),
    )

    out = tmp_path / "after.dcm"
    audit = pipeline.anonymize_dicom_file(
        dicom_with_pixels, str(tmp_path / "before.dcm"), str(out),
        str(tmp_path / "data.json"), None, None, None,
        KeyStore(str(tmp_path / "k.json")),
        job_config=_config(), secrets=RunSecrets(),
    )

    assert audit["verification_status"] == "PASSED"
    assert out.exists()


def test_the_job_config_pins_the_black_pixel_method(
        tmp_path, dicom_with_pixels, stub_detection, monkeypatch):
    """Not a user setting: blur is partially reversible on text."""
    seen = {}

    def spy(image, regions, ds=None, method="inpaint"):
        seen["method"] = method
        return image.copy(), np.zeros(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(pipeline, "redact_pixels", spy)
    monkeypatch.setattr(pipeline, "verify_redaction", lambda arr, *a, **k: (arr, "PASSED"))

    pipeline.anonymize_dicom_file(
        dicom_with_pixels, str(tmp_path / "before.dcm"), str(tmp_path / "after.dcm"),
        str(tmp_path / "data.json"), None, None, None,
        KeyStore(str(tmp_path / "k.json")),
        job_config=_config(), secrets=RunSecrets(),
    )

    assert seen["method"] == "black"


def test_without_a_job_config_the_pixel_method_is_unchanged(
        tmp_path, dicom_with_pixels, stub_detection, monkeypatch):
    seen = {}

    def spy(image, regions, ds=None, method="inpaint"):
        seen["method"] = method
        return image.copy(), np.zeros(image.shape[:2], dtype=np.uint8)

    monkeypatch.setattr(pipeline, "redact_pixels", spy)
    monkeypatch.setattr(pipeline, "verify_redaction", lambda arr, *a, **k: (arr, "PASSED"))

    pipeline.anonymize_dicom_file(
        dicom_with_pixels, str(tmp_path / "before.dcm"), str(tmp_path / "after.dcm"),
        str(tmp_path / "data.json"), None, None, None,
        KeyStore(str(tmp_path / "k.json")),
    )

    assert seen["method"] == "inpaint"
