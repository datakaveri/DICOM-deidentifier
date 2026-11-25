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


# ── job-wide UID remap (keys.uid_key) ────────────────────────────────────────
from pydicom.dataset import Dataset  # noqa: E402
from pydicom.sequence import Sequence  # noqa: E402

SERIES_A = "1.2.826.0.1.3680043.8.498.20"
INSTANCE_B = "1.2.826.0.1.3680043.8.498.30"


def with_references():
    ds = pydicom.dcmread(io.BytesIO(make_dicom()))
    ref = Dataset()
    ref.ReferencedSOPClassUID = ds.SOPClassUID
    ref.ReferencedSOPInstanceUID = INSTANCE_B
    ser = Dataset()
    ser.SeriesInstanceUID = SERIES_A
    ser.ReferencedSOPSequence = Sequence([ref])
    ds.ReferencedSeriesSequence = Sequence([ser])
    ds.FrameOfReferenceUID = "1.2.826.0.1.3680043.8.498.40"
    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def uids_of(res):
    ds = pydicom.dcmread(io.BytesIO(res.output))
    ref = ds.ReferencedSeriesSequence[0]
    return ds, ref


def test_uid_key_remaps_every_uid_consistently_across_workers():
    for base in ({}, JOB):  # fixed policy and job-config policy
        cfg = dict(base, keys={"uid_key": "job-secret", "salt": "ab" * 32})
        a = run(with_references(), config=cfg, ner=FakeNer())
        b = run(with_references(), config=cfg, ner=FakeNer())  # a different Engines
        # (other fixed-policy tags use random per-tag keys unless passed, so
        # compare the UIDs, which are what uid_key governs)
        ds, ref = uids_of(a)
        ds_b, ref_b = uids_of(b)
        assert ds.SOPInstanceUID == ds_b.SOPInstanceUID and ds.StudyInstanceUID == ds_b.StudyInstanceUID
        assert ref.SeriesInstanceUID == ref_b.SeriesInstanceUID
        seen = {str(e.value) for e in ds.iterall() if e.VR == "UI"}
        for original in (SERIES_A, INSTANCE_B, "1.2.826.0.1.3680043.8.498.1", "1.2.826.0.1.3680043.8.498.10",
                         "1.2.826.0.1.3680043.8.498.40"):
            assert original not in seen
        assert ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.1"  # a class, kept
        assert ref.ReferencedSOPSequence[0].ReferencedSOPClassUID == ds.SOPClassUID
        assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
        # same input UID, same output UID, whichever tag it sits in
        from de_identification.uids import hmac_uid

        assert ref.ReferencedSOPSequence[0].ReferencedSOPInstanceUID == hmac_uid("job-secret", INSTANCE_B)
        assert ds.SOPInstanceUID == hmac_uid("job-secret", "1.2.826.0.1.3680043.8.498.10")
        other = run(with_references(), config=dict(cfg, keys={"uid_key": "different"}), ner=FakeNer())
        assert uids_of(other)[0].SOPInstanceUID != ds.SOPInstanceUID


def test_without_uid_key_the_old_uid_policy_is_untouched():
    res = run(with_references(), ner=FakeNer())
    ds, ref = uids_of(res)
    # no uid_key: the unnamed referenced UIDs pass through, as before
    assert ref.ReferencedSOPSequence[0].ReferencedSOPInstanceUID == INSTANCE_B


# ── overlay planes ───────────────────────────────────────────────────────────
def with_overlays():
    ds = pydicom.dcmread(io.BytesIO(make_dicom(pixels=False)))
    img = np.full((128, 128), 100, np.uint16)
    img[40:50, 40:90] |= 1 << 12  # an overlay bitmap stored in bit 12 of the pixels
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.SamplesPerPixel, ds.Rows, ds.Columns = 1, 128, 128
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = img.tobytes()
    ds["PixelData"].VR = "OW"
    for g in (0x6000, 0x6002, 0x6020, 0x60FE):  # 6020+ were missed before
        ds.add_new((g, 0x0010), "US", 128)
        ds.add_new((g, 0x0011), "US", 128)
        ds.add_new((g, 0x0100), "US", 1)
        ds.add_new((g, 0x1500), "LO", "PATIENT NAME BURNED INTO OVERLAY")
    ds.add_new((0x6000, 0x3000), "OW", b"\xff" * 2048)
    ds.add_new((0x6002, 0x0102), "US", 12)  # embedded plane: no Overlay Data
    buf = io.BytesIO()
    ds.save_as(buf, write_like_original=False)
    return buf.getvalue()


def test_every_overlay_plane_and_embedded_overlay_bits_are_removed():
    for cfg in ({}, JOB):
        res = run(with_overlays(), config=cfg, ner=FakeNer())
        assert res.ok
        out = pydicom.dcmread(io.BytesIO(res.output))
        assert not [e for e in out if 0x6000 <= e.tag.group <= 0x60FF]
        assert b"BURNED INTO OVERLAY" not in res.output
        assert out.pixel_array.max() < (1 << 12)  # bit 12 bitmap cleared
        assert res.audit["verification_status"] is not None
