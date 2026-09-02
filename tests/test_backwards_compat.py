"""
Contract section 7: a config with no `dicom_deidentify` block must behave
exactly as it does today.

The two paths have deliberately opposite semantics in places -- the old one
deletes a suppressed tag, the new one blanks it; the old one hashes with a
persisted per-tag key so values correlate across runs, the new one salts per
run so they cannot. These tests pin that the old path did not drift.
"""

import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from de_identification.deidentify import deidentify_dataset
from de_identification.keystore import KeyStore
from pipeline import _deidentify_tags


@pytest.fixture
def keystore(tmp_path):
    return KeyStore(str(tmp_path / "secured.json"))


def make_dataset():
    ds = Dataset()
    ds.PatientName = "Doe^Jane"
    ds.PatientID = "UHID0012345"
    ds.PatientBirthDate = "19780412"
    ds.InstitutionName = "General Hospital"
    ds.Modality = "CR"
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = CTImageStorage
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta
    return ds


def test_no_job_config_runs_the_fixed_policy(keystore):
    """_deidentify_tags with job_config None must produce exactly what
    deidentify_dataset produces."""
    through_pipeline = _deidentify_tags(make_dataset(), keystore, None, None)
    direct = deidentify_dataset(make_dataset(), keystore)

    assert [(e["tag"], e["technique"], e["action"]) for e in through_pipeline] == \
           [(e["tag"], e["technique"], e["action"]) for e in direct]


def test_the_fixed_policy_still_deletes_a_suppressed_tag(keystore):
    """The old path deletes; only the new one blanks. Changing this would
    break every consumer of the pre-contract output."""
    ds = make_dataset()
    _deidentify_tags(ds, keystore, None, None)

    assert "PatientName" not in ds, "fixed policy deletes PatientName"
    assert "InstitutionName" not in ds


def test_the_fixed_policy_still_hashes_patient_id_with_a_persisted_key(keystore):
    """Its key lives in the keystore, so values correlate across runs -- the
    opposite of the salted per-run hashing the config path uses."""
    first = make_dataset()
    second = make_dataset()
    _deidentify_tags(first, keystore, None, None)
    _deidentify_tags(second, keystore, None, None)

    assert first.PatientID == second.PatientID
    assert first.PatientID != "UHID0012345"
    assert keystore.hash_keys, "the fixed policy persists its hash key"


def test_the_fixed_policy_still_masks_dates_to_the_year(keystore):
    ds = make_dataset()
    _deidentify_tags(ds, keystore, None, None)
    assert str(ds.PatientBirthDate) == "1978****"


def test_the_fixed_policy_still_retains_modality(keystore):
    ds = make_dataset()
    _deidentify_tags(ds, keystore, None, None)
    assert str(ds.Modality) == "CR"


def test_the_new_path_is_not_weaker_than_the_old_default(keystore):
    """The UI's default profile suppresses every identifying tag, so a user who
    clicks through without reading gets no less de-identification than before.
    Whatever the two paths do differently, neither may leave a direct
    identifier in the clear."""
    from de_identification.actions import RunSecrets
    from de_identification.deidentify import deidentify_dataset_from_config
    from de_identification.job_config import parse_job_config

    default_profile = parse_job_config({
        "operations": ["dicom_deidentify", "suppress", "hashing_with_salt", "masking"],
        "dicom_deidentify": {"default_action": "keep", "tag_actions": {
            "PatientName":      {"action": "suppress"},
            "PatientID":        {"action": "hashing_with_salt"},
            "PatientBirthDate": {"action": "masking",
                                 "characters_to_mask": [5, 6, 7, 8]},
            "InstitutionName":  {"action": "suppress"},
        }},
    })

    old = make_dataset()
    new = make_dataset()
    _deidentify_tags(old, keystore, None, None)
    deidentify_dataset_from_config(new, keystore, RunSecrets(), default_profile)

    for identifier in ("Doe^Jane", "UHID0012345", "General Hospital", "19780412"):
        assert identifier not in str(old.values()) if hasattr(old, "values") else True
        assert identifier not in "".join(
            str(elem.value) for elem in new if elem.value is not None
        ), f"{identifier} survived the config-driven path"
