"""tests/test_deidentifier.py — unit tests for the de-identification core."""

import io
import os
import sys

import pydicom
import pytest
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from deidentifier import deidentify, inspect, ALL_PHI_FIELDS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_ds(**kwargs) -> Dataset:
    """Build a minimal in-memory DICOM dataset populated with kwargs."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_implicit_VR  = False
    ds.is_little_endian = True
    ds.SOPClassUID    = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID

    for k, v in kwargs.items():
        setattr(ds, k, v)
    return ds


@pytest.fixture
def phi_ds():
    return _make_ds(
        PatientName             = "SMITH^JANE",
        PatientID               = "MRN789012",
        PatientBirthDate        = "19850322",
        PatientSex              = "F",
        PatientAddress          = "47 ELM STREET",
        PatientTelephoneNumbers = "+91-9876543210",
        ReferringPhysicianName  = "PATEL^ANISHA^DR",
        InstitutionName         = "CITY EYE CLINIC",
        StudyDate               = "20241110",
        AccessionNumber         = "ACC20241110001",
        Modality                = "OP",
    )


# ── deidentify() ─────────────────────────────────────────────────────────────

class TestDeidentify:

    def test_returns_tuple(self, phi_ds):
        result = deidentify(phi_ds)
        assert isinstance(result, tuple) and len(result) == 2

    def test_original_not_mutated(self, phi_ds):
        original_name = str(phi_ds.PatientName)
        deidentify(phi_ds)
        assert str(phi_ds.PatientName) == original_name

    def test_patient_name_masked(self, phi_ds):
        cleaned, _ = deidentify(phi_ds)
        assert str(cleaned.PatientName) == "*"

    def test_patient_id_hashed(self, phi_ds):
        cleaned, _ = deidentify(phi_ds)
        pid = str(cleaned.PatientID)
        assert pid != "MRN789012"
        assert len(pid) == 12        # SHA-256 truncated to 12 hex chars

    def test_patient_id_hash_deterministic(self, phi_ds):
        c1, _ = deidentify(phi_ds)
        c2, _ = deidentify(phi_ds)
        assert str(c1.PatientID) == str(c2.PatientID)

    def test_birth_date_year_only(self, phi_ds):
        cleaned, _ = deidentify(phi_ds)
        assert str(cleaned.PatientBirthDate) == "19850101"

    def test_study_date_year_only(self, phi_ds):
        cleaned, _ = deidentify(phi_ds)
        assert str(cleaned.StudyDate) == "20240101"

    def test_institution_masked(self, phi_ds):
        cleaned, _ = deidentify(phi_ds)
        assert str(cleaned.InstitutionName) == "*"

    def test_patient_identity_removed_flag(self, phi_ds):
        cleaned, _ = deidentify(phi_ds)
        assert str(cleaned.PatientIdentityRemoved) == "YES"

    def test_uids_regenerated(self, phi_ds):
        cleaned, _ = deidentify(phi_ds)
        assert cleaned.StudyInstanceUID  != phi_ds.get("StudyInstanceUID")
        assert cleaned.SeriesInstanceUID != phi_ds.get("SeriesInstanceUID")
        assert cleaned.SOPInstanceUID    != phi_ds.SOPInstanceUID

    def test_non_phi_field_untouched(self, phi_ds):
        cleaned, _ = deidentify(phi_ds)
        assert str(cleaned.Modality) == "OP"

    def test_audit_log_not_empty(self, phi_ds):
        _, audit = deidentify(phi_ds)
        assert len(audit) > 0

    def test_audit_log_has_required_keys(self, phi_ds):
        _, audit = deidentify(phi_ds)
        for entry in audit:
            assert "field" in entry
            assert "old_value" in entry
            assert "action" in entry

    def test_audit_actions_are_valid(self, phi_ds):
        _, audit = deidentify(phi_ds)
        valid = {"masked", "hashed", "date_generalised"}
        for entry in audit:
            assert entry["action"] in valid

    def test_missing_fields_skipped_gracefully(self):
        """Dataset with no PHI fields should produce empty audit."""
        ds = _make_ds(Modality="CT")
        cleaned, audit = deidentify(ds)
        assert audit == []

    def test_address_masked(self, phi_ds):
        cleaned, _ = deidentify(phi_ds)
        assert str(cleaned.PatientAddress) == "*"


# ── inspect() ────────────────────────────────────────────────────────────────

class TestInspect:

    def test_returns_list(self, phi_ds):
        result = inspect(phi_ds)
        assert isinstance(result, list)

    def test_finds_patient_name(self, phi_ds):
        tags = inspect(phi_ds)
        fields = [t["field"] for t in tags]
        assert "PatientName" in fields

    def test_each_entry_has_field_and_value(self, phi_ds):
        for entry in inspect(phi_ds):
            assert "field" in entry
            assert "value" in entry

    def test_empty_dataset_returns_empty_list(self):
        ds = _make_ds(Modality="CT")
        assert inspect(ds) == []


# ── Round-trip via file serialisation ────────────────────────────────────────

class TestSerialisation:

    def test_cleaned_dataset_saves_and_reloads(self, phi_ds):
        cleaned, _ = deidentify(phi_ds)
        buf = io.BytesIO()
        cleaned.save_as(buf)
        buf.seek(0)
        reloaded = pydicom.dcmread(buf)
        assert str(reloaded.PatientName) == "*"
        assert str(reloaded.PatientIdentityRemoved) == "YES"
