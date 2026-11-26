"""Tag-first matching, preserved tags, mirror layout and resume."""

import json

import numpy as np
import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

import main
import pipeline
from de_identification.keystore import KeyStore


def entry(tag, keyword, vr="LO"):
    return {"tag": tag, "keyword": keyword, "vr": vr, "value": "x"}


def test_fixed_policy_filter_keeps_only_tags_the_tag_pass_changes():
    keep = pipeline.deidentified_tag_filter(None)
    assert keep(entry("(0010, 0010)", "PatientName", "PN"))
    assert keep(entry("(0008, 0080)", "InstitutionName"))
    assert keep(entry("(0008, 1234)", "SomethingDate", "DA"))  # date catch-all
    assert keep(entry("(0009, 0010)", "", "LO"))  # private
    assert not keep(entry("(0008, 0070)", "Manufacturer"))  # not de-identified
    assert not keep(entry("(0008, 0060)", "Modality", "CS"))


def test_job_config_filter_follows_tag_actions():
    from de_identification.job_config import parse_job_config

    cfg = parse_job_config(
        {
            "operations": ["dicom_deidentify", "suppress"],
            "dicom_deidentify": {
                "tag_actions": {"PatientName": {"action": "suppress"}}
            },
        }
    )
    keep = pipeline.deidentified_tag_filter(cfg)
    assert keep(entry("(0010, 0010)", "PatientName", "PN"))
    assert not keep(entry("(0008, 0080)", "InstitutionName"))  # user kept it


def test_preserved_tag_is_never_matched(monkeypatch):
    monkeypatch.setattr(pipeline, "PRESERVE_TAGS", ("PatientID",))
    keep = pipeline.deidentified_tag_filter(None)
    assert not keep(entry("(0010, 0020)", "PatientID"))


@pytest.fixture
def image_file(tmp_path):
    ds = Dataset()
    ds.PatientName = "Doe^Jane"
    ds.PatientID = "UHID0012345"
    ds.SOPClassUID = CTImageStorage
    ds.SOPInstanceUID = generate_uid()
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.Rows, ds.Columns, ds.SamplesPerPixel = 32, 32, 1
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = np.full((32, 32), 1000, dtype=np.uint16).tobytes()
    ds.file_meta = FileMetaDataset()
    ds.file_meta.MediaStorageSOPClassUID = CTImageStorage
    ds.file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    path = tmp_path / "in.dcm"
    ds.save_as(str(path), write_like_original=False)
    return str(path), ds


def test_tag_matched_text_is_redacted_without_ner(tmp_path, image_file, monkeypatch):
    path, _ = image_file
    name = {"text": "DOE JANE", "bbox": [2, 2, 20, 8]}
    other = {"text": "SOME WORDS", "bbox": [2, 20, 20, 26]}
    seen = {}
    monkeypatch.setattr(
        pipeline, "detect_text_in_regions", lambda *a, **k: [name, other]
    )
    monkeypatch.setattr(pipeline, "merge_detections", lambda det: det)
    monkeypatch.setattr(
        pipeline, "load_original_tag_values", lambda p, **k: ["DOE JANE"]
    )

    def classify(dets, *a, **k):
        seen["texts"] = [d["text"] for d in dets]
        return []

    monkeypatch.setattr(pipeline, "classify_phi", classify)
    monkeypatch.setattr(
        pipeline, "verify_redaction", lambda arr, *a, **k: (arr, "PASSED")
    )
    audit = pipeline.redact_file_pixels(
        path,
        str(tmp_path / "b.dcm"),
        str(tmp_path / "a.dcm"),
        str(tmp_path / "d.json"),
        None,
        None,
    )
    assert audit["error"] is None
    assert seen["texts"] == ["SOME WORDS"], "the matched name must not reach NER"
    assert [r["text"] for r in audit["redacted_regions"]] == ["DOE JANE"]
    assert audit["tag_matched_regions"] == 1
    assert {"read", "decode", "ocr", "tag_match", "classify", "redact"} <= set(
        audit["timings"]
    )


def test_preserved_uids_survive_the_fixed_policy(tmp_path, image_file, monkeypatch):
    path, src = image_file
    monkeypatch.setattr(
        pipeline,
        "PRESERVE_TAGS",
        ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"),
    )
    audit = {
        "file": "in.dcm",
        "before_output_path": path,
        "output_path": str(tmp_path / "o" / "x.dcm"),
        "pixel_stage_complete": True,
        "error": None,
    }
    pipeline.deidentify_file_tags(audit, KeyStore(str(tmp_path / "k.json")))
    out = pydicom.dcmread(str(tmp_path / "o" / "x.dcm"))
    for kw in ("StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID"):
        assert out[kw].value == src[kw].value, kw
    assert out.file_meta.MediaStorageSOPInstanceUID == src.SOPInstanceUID
    assert "PatientName" not in out or str(out.PatientName) != "Doe^Jane"
    assert any(e["technique"] == "preserved" for e in audit["deidentified_tags"])


def test_mirror_layout_keeps_relative_path_and_name(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "OUTPUT_LAYOUT", "mirror")
    monkeypatch.setattr(main, "INPUT_DIR", "/in")
    monkeypatch.setattr(main, "OUTPUT_DIR", "/out")
    monkeypatch.setattr(main, "REPORT_DIR", "/reports")
    monkeypatch.setattr(main, "_SCRATCH_DIR", None)
    p = main._paths("/in/a b/c/img.1.dcm")
    assert p["final"] == "/out/a b/c/img.1.dcm"
    assert p["audit"].startswith("/reports/a b/c/img.1/")
    assert p["preview"].startswith("/reports/") and p["before"].startswith("/reports/")


def test_resume_only_skips_a_finished_successful_file(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "OUTPUT_LAYOUT", "per-file")
    monkeypatch.setattr(main, "OUTPUT_DIR", str(tmp_path))
    monkeypatch.setattr(main, "_SCRATCH_DIR", None)
    monkeypatch.setattr(main, "_OUT_NAMES", {})
    src = "/in/img.dcm"
    p = main._paths(src)
    assert main._already_done(src) is None
    (tmp_path / "img").mkdir()
    open(p["final"], "w").close()
    json.dump({"input_path": src, "error": None}, open(p["audit"], "w"))
    json.dump([], open(p["tag_audit"], "w"))
    assert main._already_done(src)["resumed"] is True
    json.dump({"input_path": src, "error": "boom"}, open(p["audit"], "w"))
    assert main._already_done(src) is None


def test_stats():
    st = main._stats([1, 2, 3, 4, None])
    assert st["n"] == 4 and st["mean"] == 2.5 and st["max"] == 4
