"""
Tests for the run-level policy the job config pins: the pixel method, what may
be written to the output volume, and how the manifest is keyed
(contract sections 4, 5, 6).
"""

import numpy as np
import pydicom
import pytest
from pydicom.uid import generate_uid

from de_identification.actions import RunSecrets
from de_identification.keystore import KeyStore
from masking import _redact_black, redact_pixels


@pytest.fixture
def keystore(tmp_path):
    return KeyStore(str(tmp_path / "secured.json"))


# ── Pixel method "black" (section 4) ─────────────────────────────────────────

def test_black_fill_blanks_the_whole_bbox():
    image = np.full((40, 40), 200, dtype=np.uint16)
    image[10:20, 10:20] = 4000  # "burned-in text"

    cleaned = _redact_black(image.copy(), 10, 10, 20, 20)

    assert cleaned[10:20, 10:20].max() == 200, "the bbox must be filled flat"
    assert cleaned[0:5, 0:5].max() == 200, "outside the bbox is untouched"


def test_black_fill_leaves_no_residue_of_the_original_values():
    """Unlike blur, which is partially reversible on text, a black fill keeps
    nothing of what was there."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 4096, size=(40, 40), dtype=np.uint16)
    image[10:20, 10:20] = rng.integers(0, 4096, size=(10, 10), dtype=np.uint16)

    cleaned = _redact_black(image.copy(), 10, 10, 20, 20)

    region = cleaned[10:20, 10:20]
    assert region.min() == region.max(), "a black fill must be a single value"


def test_redact_pixels_black_method_fills_every_region():
    image = np.full((60, 60), 500, dtype=np.uint16)
    image[5:15, 5:15] = 3000
    image[40:50, 40:50] = 3000
    regions = [
        {"bbox": [5, 5, 15, 15], "zone": "border", "text": "JANE DOE"},
        {"bbox": [40, 40, 50, 50], "zone": "anatomy", "text": "UHID 12345"},
    ]

    cleaned, mask = redact_pixels(image, regions, method="black")

    assert cleaned[5:15, 5:15].max() == 500
    assert cleaned[40:50, 40:50].max() == 500
    assert mask[5:15, 5:15].all() and mask[40:50, 40:50].all()


def test_black_method_ignores_the_zone_split():
    """The zone-aware inpaint/border split is the pre-contract behaviour; the
    pinned "black" method treats every region the same way."""
    image = np.full((60, 60), 500, dtype=np.uint16)
    image[5:15, 5:15] = 3000
    border = redact_pixels(image, [{"bbox": [5, 5, 15, 15], "zone": "border"}],
                           method="black")[0]
    anatomy = redact_pixels(image, [{"bbox": [5, 5, 15, 15], "zone": "anatomy"}],
                            method="black")[0]
    assert np.array_equal(border, anatomy)


def test_inpaint_remains_the_default():
    """The pre-contract path must behave exactly as it does today."""
    image = np.full((60, 60), 500, dtype=np.uint16)
    image[20:30, 20:30] = 3000
    regions = [{"bbox": [20, 20, 30, 30], "zone": "anatomy"}]

    default_result, _ = redact_pixels(image, regions)
    explicit, _ = redact_pixels(image, regions, method="inpaint")

    assert np.array_equal(default_result, explicit)


# ── emit_pixel_text_report pinned false (section 5) ──────────────────────────

def test_recognised_burned_in_text_is_stripped_from_the_published_audit():
    """The recognised text is patient names and MRNs -- exactly the PHI the
    pass just blacked out. Writing it beside the output is the plaintext
    sidecar emit_pixel_text_report exists to forbid."""
    from main import _publishable_audit

    audit = {
        "file": "chest_xray.dcm",
        "verification_status": "PASSED",
        "redacted_regions": [
            {"text": "DOE^JANE", "bbox": [10, 10, 90, 30]},
            {"text": "UHID 0012345", "bbox": [10, 40, 120, 60]},
        ],
        "deidentified_tags": [],
    }

    published = _publishable_audit(audit)
    serialized = str(published)

    assert "DOE^JANE" not in serialized
    assert "UHID 0012345" not in serialized
    assert len(published["redacted_regions"]) == 2, "the count must survive"
    assert published["redacted_regions"][0]["bbox"] == [10, 10, 90, 30]
    assert published["verification_status"] == "PASSED"


def test_stripping_the_audit_does_not_mutate_the_original():
    from main import _publishable_audit

    audit = {"redacted_regions": [{"text": "DOE^JANE", "bbox": [1, 2, 3, 4]}]}
    _publishable_audit(audit)
    assert audit["redacted_regions"][0]["text"] == "DOE^JANE"


# ── Manifest shape (section 6) ───────────────────────────────────────────────

def test_manifest_entry_carries_the_fields_the_ui_reads():
    from main import _summarize

    audit = {
        "verification_status": "PASSED",
        "redacted_regions": [{"bbox": [1, 2, 3, 4]}] * 3,
        "deidentified_tags": [
            {"technique": "suppress", "changed": True},
            {"technique": "suppress", "changed": True},
            {"technique": "hashing_with_salt", "changed": True},
            {"technique": "masking", "changed": True},
        ],
        "execution_time_seconds": 12.5,
        "error": None,
    }

    entry = _summarize(audit, "/in/chest_xray.dcm", "chest_xray.dcm", "/out/chest_xray")

    assert entry["error"] is None
    assert entry["pixel_verification_status"] == "PASSED"
    assert entry["redacted_regions"] == 3
    assert entry["tags_touched"] == 4
    assert entry["tags_by_technique"] == {
        "suppress": 2, "hashing_with_salt": 1, "masking": 1,
    }


def test_tags_by_technique_is_keyed_off_the_six_action_identifiers():
    """They line up exactly with what the config sent."""
    from main import _summarize

    audit = {
        "verification_status": "PASSED",
        "redacted_regions": [],
        "deidentified_tags": [
            {"technique": t, "changed": True} for t in
            ["suppress", "hashing_with_salt", "encrypt", "masking",
             "tokenization", "charcloak"]
        ],
    }

    entry = _summarize(audit, "/in/a.dcm", "a.dcm", "/out/a")

    assert set(entry["tags_by_technique"]) == {
        "suppress", "hashing_with_salt", "encrypt", "masking",
        "tokenization", "charcloak",
    }


def test_untouched_tags_do_not_inflate_tags_touched():
    from main import _summarize

    audit = {
        "verification_status": "PASSED",
        "redacted_regions": [],
        "deidentified_tags": [
            {"technique": "suppress", "changed": True},
            {"technique": "hashing_with_salt", "changed": False},
        ],
    }

    entry = _summarize(audit, "/in/a.dcm", "a.dcm", "/out/a")

    assert entry["tags_touched"] == 1
    assert "hashing_with_salt" not in entry["tags_by_technique"]


# ── A masked date must still survive a write/read cycle ──────────────────────

def test_a_masked_date_still_writes_and_reads_back(tmp_path, keystore):
    """'1978****' is not a valid DA value and pydicom says so, but the
    Research-safe profile emits exactly that and the file must still load."""
    import sys
    sys.path.insert(0, str(tmp_path))
    from de_identification.deidentify import deidentify_dataset_from_config
    from de_identification.job_config import parse_job_config
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

    ds = Dataset()
    ds.PatientBirthDate = "19780412"
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = CTImageStorage
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta

    cfg = parse_job_config({
        "operations": ["dicom_deidentify", "masking"],
        "dicom_deidentify": {"default_action": "keep", "tag_actions": {
            "PatientBirthDate": {"action": "masking", "characters_to_mask": [5, 6, 7, 8]},
        }},
    })
    deidentify_dataset_from_config(ds, keystore, RunSecrets(), cfg)

    path = tmp_path / "masked.dcm"
    ds.save_as(str(path), write_like_original=False)
    assert str(pydicom.dcmread(str(path)).PatientBirthDate) == "1978****"
