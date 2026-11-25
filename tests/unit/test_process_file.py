"""process_file: one in-memory file at a time, models held by the caller.
Synthetic data only; a fake OCR and a fake NER stand in for the real models."""

import io
import json
import os

import cv2
import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian

from single_file import Engines, FileResult, Keys, process_file

NAME, MRN = "DOE^JANE", "MRN778899"
JOB = {
    "dicom_deidentify": {
        "default_action": "keep",
        "tag_actions": {
            "PatientName": {"action": "suppress"},
            "PatientID": {"action": "hashing_with_salt"},
            "StudyDescription": {"action": "free_text_anonymization"},
        },
    },
    "operations": ["dicom_deidentify", "suppress", "hashing_with_salt", "free_text_anonymization"],
}


class FakeOCR:
    """Reports the MRN in the top-left corner while bright pixels remain there."""

    def ocr(self, rgb, cls=False):
        if float(rgb[8:34, 8:140].mean()) > 4:
            return [[[[[10, 10], [138, 10], [138, 32], [10, 32]], (MRN, 0.99)]]]
        return [[]]


class FakeNer:
    def spans(self, texts):
        out = []
        for t in texts:
            i = t.find("RAMESH KUMAR")
            out.append([{"start": i, "end": i + 12, "label": "PERSON", "confidence": 0.99}] if i >= 0 else [])
        return out


class DownNer:
    def spans(self, texts):
        return None


def make_dicom(pixels=True, colour=False):
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    meta.MediaStorageSOPInstanceUID = "1.2.826.0.1.3680043.8.498.10"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2.826.0.1.3680043.8.498.1"
    ds.PatientName, ds.PatientID, ds.Modality = NAME, MRN, "CR"
    ds.StudyDescription = "XR CHEST PA - RAMESH KUMAR"
    ds.is_little_endian, ds.is_implicit_VR = True, False
    if pixels:
        img = np.zeros((128, 128), np.uint8)
        cv2.putText(img, MRN, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 255, 2)
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.SamplesPerPixel, ds.Rows, ds.Columns = 1, 128, 128
        ds.BitsAllocated = ds.BitsStored = 8
        ds.HighBit, ds.PixelRepresentation = 7, 0
        ds.PixelData = img.tobytes()
        ds["PixelData"].VR = "OB"
    if colour:
        ds.PhotometricInterpretation, ds.SamplesPerPixel, ds.PlanarConfiguration = "RGB", 3, 0
        ds.Rows = ds.Columns = 16
        ds.BitsAllocated = ds.BitsStored = 8
        ds.HighBit, ds.PixelRepresentation = 7, 0
        ds.PixelData = bytes(16 * 16 * 3)
        ds["PixelData"].VR = "OB"
    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def run(data, config=None, ner=None, engines=None):
    cfg = JOB if config is None else config
    return process_file(data, "a/b.dcm", cfg, engines or Engines(paddle_ocr=FakeOCR(), ner=ner))


def test_job_config_redacts_pixels_and_tags():
    res = run(make_dicom(), ner=FakeNer())
    assert isinstance(res, FileResult) and res.ok and res.error_code is None
    assert res.output_rel_path == "a/b.dcm"
    out = pydicom.dcmread(io.BytesIO(res.output))
    assert out.PatientName == ""  # suppressed (blanked)
    assert out.PatientID != MRN
    assert out.StudyDescription == "XR CHEST PA - [PERSON]"
    assert out.pixel_array[8:34, 8:140].max() == 0  # burned-in MRN blacked out
    assert res.audit["redacted_region_count"] == 1


def test_audit_carries_no_phi():
    res = run(make_dicom(), ner=FakeNer())
    blob = json.dumps(res.audit)
    for secret in (NAME, "DOE", MRN, "RAMESH", "KUMAR"):
        assert secret not in blob


def test_free_text_is_blanked_when_ner_is_unavailable():
    for ner in (DownNer(), None):  # injected-but-down, and no socket at all
        res = run(make_dicom(), ner=ner)
        assert res.ok
        assert pydicom.dcmread(io.BytesIO(res.output)).StudyDescription == ""


def test_writes_no_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    run(make_dicom(), ner=FakeNer())
    assert list(tmp_path.iterdir()) == []


def test_not_dicom_and_colour_fail_with_codes_instead_of_raising():
    bad = run(b"this is not dicom at all" * 20)
    assert (bad.ok, bad.output, bad.error_code) == (False, None, "NOT_DICOM")
    colour = run(make_dicom(pixels=False, colour=True))
    assert (colour.ok, colour.error_code) == (False, "UNSUPPORTED_COLOUR")


def test_invalid_config_is_a_result_not_an_exception():
    res = run(make_dicom(), config={"dicom_deidentify": {"default_action": "suppress"}})
    assert (res.ok, res.error_code) == (False, "INVALID_CONFIG")


def test_pixel_less_object_passes_through_the_tag_pass():
    res = run(make_dicom(pixels=False), ner=FakeNer())
    assert res.ok and res.audit["verification_status"] == "SKIPPED"


def test_keys_in_config_make_two_workers_agree():
    cfg = dict(JOB, keys={"salt": "ab" * 32})
    a = run(make_dicom(), config=cfg, ner=FakeNer())
    b = run(make_dicom(), config=cfg, ner=FakeNer())  # a second Engines, as in a second actor
    pid = lambda r: pydicom.dcmread(io.BytesIO(r.output)).PatientID  # noqa: E731
    assert pid(a) == pid(b)
    other = run(make_dicom(), config=dict(JOB, keys={"salt": "cd" * 32}), ner=FakeNer())
    assert pid(other) != pid(a)


def test_fixed_policy_uses_passed_hash_keys():
    keys = {"hash_keys": {"PatientID": "11" * 16}}
    cfg = {"keys": keys}
    a = run(make_dicom(), config=cfg)
    b = run(make_dicom(), config=cfg)
    assert a.ok and a.output == b.output
    assert pydicom.dcmread(io.BytesIO(a.output)).PatientID != MRN


def test_keys_object_can_be_passed_explicitly():
    eng = Engines(paddle_ocr=FakeOCR(), ner=FakeNer())
    res = process_file(make_dicom(), "x.dcm", JOB, eng, keys=Keys.from_config({"keys": {"salt": "ee" * 32}}))
    assert res.ok


def test_models_are_reused_not_reloaded():
    eng = Engines(paddle_ocr=FakeOCR(), ner=FakeNer())
    ocr = eng.paddle_ocr
    for _ in range(3):
        assert process_file(make_dicom(), "x.dcm", JOB, eng).ok
    assert eng.paddle_ocr is ocr and len(eng._job_configs) == 1
