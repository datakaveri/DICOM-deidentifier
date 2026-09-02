"""
Tests for the six actions and the dataset walk they drive (contract sections
2, 3, 8).

These cover the semantics that fail SILENTLY -- where a wrong result still
looks like a successfully de-identified file: a mask off by one position, a
suppress that deleted instead of blanked, a hash that is stable across runs
when it must not be, a description that did not come out byte-identical.
"""

import pydicom
import pytest
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from de_identification.actions import RunSecrets
from de_identification.deidentify import deidentify_dataset_from_config
from de_identification.job_config import parse_job_config
from de_identification.keystore import KeyStore


@pytest.fixture
def keystore(tmp_path):
    return KeyStore(str(tmp_path / "secured.json"))


@pytest.fixture
def secrets():
    return RunSecrets()


def make_config(tag_actions, operations=None):
    return parse_job_config({
        "data_type": "chest_xray",
        "dataset_name": "chest_xray.dcm",
        "format": "dicom",
        "operations": operations or ["dicom_deidentify"],
        "dicom_deidentify": {"default_action": "keep", "tag_actions": tag_actions},
    })


def make_dataset(**overrides):
    """A small but realistic dataset: the tags the UI can name, with values
    that exercise the VRs each action has to respect."""
    ds = Dataset()
    ds.PatientName = "Doe^Jane"
    ds.PatientID = "UHID0012345"
    ds.PatientBirthDate = "19780412"
    ds.PatientSex = "F"
    ds.AccessionNumber = "ACC7781234"
    ds.StudyDate = "20240115"
    ds.StudyDescription = "CHEST PA"
    ds.SeriesDescription = "CHEST PA ERECT"
    ds.InstitutionName = "General Hospital"
    ds.Modality = "CR"
    ds.Manufacturer = "Acme Imaging"
    ds.BodyPartExamined = "CHEST"
    ds.Rows = 512
    ds.Columns = 512
    ds.BitsAllocated = 16
    ds.StudyInstanceUID = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.SOPInstanceUID = generate_uid()
    ds.SOPClassUID = CTImageStorage

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.file_meta = meta

    for key, value in overrides.items():
        setattr(ds, key, value)
    return ds


def run(ds, tag_actions, keystore, secrets, operations=None):
    cfg = make_config(tag_actions, operations)
    audit = deidentify_dataset_from_config(ds, keystore, secrets, cfg)
    return ds, audit


# ── suppress: blank, tag RETAINED (never deleted) ────────────────────────────

def test_suppress_leaves_the_tag_present_and_empty(keystore, secrets):
    """Deleting the tag breaks readers that expect a Type 2 attribute to be
    present. This is the difference from the fixed-policy path, which deletes."""
    ds, _ = run(make_dataset(), {"PatientName": {"action": "suppress"}}, keystore, secrets)

    assert "PatientName" in ds, "suppress must retain the tag, not delete it"
    assert ds[(0x0010, 0x0010)].value in ("", None)
    assert str(ds.PatientName) == ""


def test_suppress_on_several_tags_retains_all_of_them(keystore, secrets):
    ds, _ = run(make_dataset(), {
        "PatientName": {"action": "suppress"},
        "InstitutionName": {"action": "suppress"},
        "ReferringPhysicianName": {"action": "suppress"},
    }, keystore, secrets)
    assert "PatientName" in ds
    assert "InstitutionName" in ds
    assert str(ds.InstitutionName) == ""


def test_suppressed_dataset_still_round_trips_through_pydicom(tmp_path, keystore, secrets):
    """A blanked Type 2 attribute must still write and read back."""
    ds, _ = run(make_dataset(), {
        "PatientName": {"action": "suppress"},
        "InstitutionName": {"action": "suppress"},
    }, keystore, secrets)

    path = tmp_path / "out.dcm"
    ds.save_as(str(path), write_like_original=False)
    reread = pydicom.dcmread(str(path))
    assert "PatientName" in reread
    assert str(reread.PatientName) == ""


# ── hashing_with_salt: stable within a run, different across runs ────────────

def test_hashing_is_stable_within_a_run(keystore, secrets):
    """Same input, same output within a run, so records still join."""
    actions = {"PatientID": {"action": "hashing_with_salt"}}
    first, _ = run(make_dataset(), actions, keystore, secrets)
    second, _ = run(make_dataset(), actions, keystore, secrets)
    assert first.PatientID == second.PatientID
    assert first.PatientID != "UHID0012345"


def test_hashing_differs_across_runs(keystore, tmp_path):
    """Different salt per run, so there is no cross-run linkage -- this holds
    even against a keystore that persists, because the salt never enters it."""
    actions = {"PatientID": {"action": "hashing_with_salt"}}
    run_one, _ = run(make_dataset(), actions, keystore, RunSecrets())
    run_two, _ = run(make_dataset(), actions, KeyStore(str(tmp_path / "k2.json")), RunSecrets())
    assert run_one.PatientID != run_two.PatientID


def test_the_salt_never_reaches_the_keystore(keystore, secrets, tmp_path):
    """The salt must never leave the enclave -- not to disk, not to an audit."""
    run(make_dataset(), {"PatientID": {"action": "hashing_with_salt"}}, keystore, secrets)
    keystore.save()

    written = (tmp_path / "secured.json").read_text()
    assert secrets.salt not in written


def test_hash_is_not_in_the_audit(keystore, secrets):
    """The audit records which technique hit which tag, never any value."""
    ds, audit = run(make_dataset(), {"PatientID": {"action": "hashing_with_salt"}},
                    keystore, secrets)
    serialized = str(audit)
    assert "UHID0012345" not in serialized
    assert str(ds.PatientID) not in serialized


def test_hash_fits_the_vr_length_limit(keystore, secrets):
    """A hash written into an LO must fit its 64-character limit."""
    ds, _ = run(make_dataset(), {"PatientID": {"action": "hashing_with_salt"}},
                keystore, secrets)
    assert ds[(0x0010, 0x0020)].VR == "LO"
    assert 0 < len(str(ds.PatientID)) <= 64


def test_hash_of_a_uid_stays_a_valid_uid(keystore, secrets):
    """A raw hex digest is not a valid UID. The user can assign a hash to one
    anyway; the result is still syntactically valid."""
    ds, _ = run(make_dataset(), {"SOPInstanceUID": {"action": "hashing_with_salt"}},
                keystore, secrets)
    uid = str(ds.SOPInstanceUID)
    assert uid.startswith("2.25.")
    assert len(uid) <= 64
    assert all(part.isdigit() for part in uid.split("."))


def test_equal_values_in_different_tags_still_join(keystore, secrets):
    """Salted hashing is keyed on the run's salt alone, so the same value
    hashes identically wherever it appears -- that is what makes records join."""
    ds = make_dataset(PatientID="SHARED123", AccessionNumber="SHARED123")
    ds, _ = run(ds, {
        "PatientID": {"action": "hashing_with_salt"},
        "AccessionNumber": {"action": "hashing_with_salt"},
    }, keystore, secrets)
    # AccessionNumber is SH (16 chars) and PatientID LO (64), so compare the
    # common prefix rather than the truncated whole.
    assert str(ds.PatientID)[:16] == str(ds.AccessionNumber)[:16]


# ── masking: 1-based positions ───────────────────────────────────────────────

def test_masking_positions_are_1_based(keystore, secrets):
    """The single most silent failure available: 0-based positions turn
    19780412 into 197*****-shaped nonsense that still looks de-identified."""
    ds, _ = run(make_dataset(), {
        "PatientBirthDate": {"action": "masking", "masking_char": "*",
                             "characters_to_mask": [5, 6, 7, 8]},
    }, keystore, secrets)
    assert str(ds.PatientBirthDate) == "1978****"


def test_masking_a_date_coarsens_it_without_date_awareness(keystore, secrets):
    """There is no generalisation action; coarsening a date is expressed as an
    ordinary positional mask over YYYYMMDD's month and day."""
    ds = make_dataset(PatientBirthDate="20011231", StudyDate="19991005")
    ds, _ = run(ds, {
        "PatientBirthDate": {"action": "masking", "characters_to_mask": [5, 6, 7, 8]},
        "StudyDate": {"action": "masking", "characters_to_mask": [5, 6, 7, 8]},
    }, keystore, secrets)
    assert str(ds.PatientBirthDate) == "2001****"
    assert str(ds.StudyDate) == "1999****"


def test_masking_leaves_unlisted_positions_alone(keystore, secrets):
    ds = make_dataset(PatientID="ABCDEFGH")
    ds, _ = run(ds, {
        "PatientID": {"action": "masking", "masking_char": "#",
                      "characters_to_mask": [1, 3]},
    }, keystore, secrets)
    assert str(ds.PatientID) == "#B#DEFGH"


def test_masking_honours_a_custom_masking_char(keystore, secrets):
    ds, _ = run(make_dataset(), {
        "PatientBirthDate": {"action": "masking", "masking_char": "X",
                             "characters_to_mask": [5, 6, 7, 8]},
    }, keystore, secrets)
    assert str(ds.PatientBirthDate) == "1978XXXX"


def test_masking_positions_past_the_end_are_ignored(keystore, secrets):
    ds = make_dataset(PatientID="AB")
    ds, _ = run(ds, {
        "PatientID": {"action": "masking", "characters_to_mask": [1, 2, 50]},
    }, keystore, secrets)
    assert str(ds.PatientID) == "**"


# ── encrypt ──────────────────────────────────────────────────────────────────

def test_format_preserving_encrypt_stays_valid_for_the_vr(keystore, secrets):
    """format_preserving is the only variant safe to write into a typed
    element: same length, same character classes, so it still fits the VR."""
    original = "ACC7781234"
    ds = make_dataset(AccessionNumber=original)
    ds, _ = run(ds, {
        "AccessionNumber": {"action": "encrypt", "format_preserving": True},
    }, keystore, secrets, operations=["dicom_deidentify", "encrypt"])

    encrypted = str(ds.AccessionNumber)
    assert encrypted != original
    assert len(encrypted) == len(original)
    assert len(encrypted) <= 16, "AccessionNumber is SH: 16 characters"
    for got, want in zip(encrypted, original):
        assert got.isdigit() == want.isdigit()
        assert got.isalpha() == want.isalpha()
        assert got.isupper() == want.isupper()


def test_format_preserving_encrypt_round_trips_through_pydicom(tmp_path, keystore, secrets):
    ds = make_dataset()
    ds, _ = run(ds, {
        "PatientID": {"action": "encrypt", "format_preserving": True},
    }, keystore, secrets, operations=["dicom_deidentify", "encrypt"])

    path = tmp_path / "out.dcm"
    ds.save_as(str(path), write_like_original=False)
    reread = pydicom.dcmread(str(path))
    assert str(reread.PatientID) == str(ds.PatientID)


def test_format_preserving_encrypt_is_deterministic_within_a_run(keystore, secrets):
    actions = {"PatientID": {"action": "encrypt", "format_preserving": True}}
    ops = ["dicom_deidentify", "encrypt"]
    first, _ = run(make_dataset(), actions, keystore, secrets, ops)
    second, _ = run(make_dataset(), actions, keystore, secrets, ops)
    assert first.PatientID == second.PatientID


def test_non_format_preserving_encrypt_widens_the_value(keystore, secrets):
    """With format_preserving false the ciphertext does not keep the input's
    shape -- it is written whole rather than truncated, since truncating
    ciphertext would destroy it."""
    ds = make_dataset()
    ds, _ = run(ds, {
        "PatientID": {"action": "encrypt", "format_preserving": False},
    }, keystore, secrets, operations=["dicom_deidentify", "encrypt"])

    encrypted = str(ds.PatientID)
    assert encrypted.startswith("ENC$")
    assert "UHID0012345" not in encrypted


# ── tokenization and charcloak ───────────────────────────────────────────────

def test_tokenization_is_opaque_and_unrelated_to_the_input(keystore, secrets):
    ds = make_dataset()
    ds, _ = run(ds, {"PatientID": {"action": "tokenization"}}, keystore, secrets,
                operations=["dicom_deidentify", "tokenization"])
    token = str(ds.PatientID)
    assert token != "UHID0012345"
    assert "UHID" not in token
    assert len(token) <= 64


def test_tokenization_is_not_stable_across_records(keystore, secrets):
    """An opaque generated value with no stability across records -- unlike
    hashing_with_salt, two records with the same input get different tokens."""
    actions = {"PatientID": {"action": "tokenization"}}
    ops = ["dicom_deidentify", "tokenization"]
    first, _ = run(make_dataset(), actions, keystore, secrets, ops)
    second, _ = run(make_dataset(), actions, keystore, secrets, ops)
    assert first.PatientID != second.PatientID


def test_charcloak_preserves_character_classes(keystore, secrets):
    original = "UHID0012345"
    ds = make_dataset()
    ds, _ = run(ds, {"PatientID": {"action": "charcloak"}}, keystore, secrets,
                operations=["dicom_deidentify", "charcloak"])
    cloaked = str(ds.PatientID)
    assert cloaked != original
    assert len(cloaked) == len(original)
    for got, want in zip(cloaked, original):
        assert got.isdigit() == want.isdigit()


# ── default_action "keep": unlisted tags come out byte-identical ─────────────

def test_descriptions_pass_through_byte_identical(keystore, secrets):
    """With no free-text redaction the only outcomes are keeping the field
    whole or losing it whole, and deleting takes "CHEST PA" with "Jane Doe".
    No profile makes that trade for the user."""
    ds = make_dataset()
    before_study = str(ds.StudyDescription)
    before_series = str(ds.SeriesDescription)

    ds, _ = run(ds, {"PatientName": {"action": "suppress"}}, keystore, secrets)

    assert str(ds.StudyDescription) == before_study == "CHEST PA"
    assert str(ds.SeriesDescription) == before_series == "CHEST PA ERECT"


def test_a_description_named_explicitly_is_suppressed(keystore, secrets):
    """The UI offers Remove, so a suppress on a description is deliberate --
    honoured literally."""
    ds, _ = run(make_dataset(), {"StudyDescription": {"action": "suppress"}},
                keystore, secrets)
    assert "StudyDescription" in ds
    assert str(ds.StudyDescription) == ""


def test_unlisted_non_phi_tags_are_untouched(keystore, secrets):
    """Modality, BodyPartExamined and Manufacturer are what a reader needs to
    know what the image is. Quietly removing them destroys the file's
    usefulness with no way to tell why."""
    ds, _ = run(make_dataset(), {"PatientName": {"action": "suppress"}}, keystore, secrets)
    assert str(ds.Modality) == "CR"
    assert str(ds.Manufacturer) == "Acme Imaging"
    assert str(ds.BodyPartExamined) == "CHEST"
    assert ds.Rows == 512
    assert ds.Columns == 512
    assert ds.BitsAllocated == 16
    assert str(ds.PatientSex) == "F"


def test_an_empty_tag_actions_changes_no_named_tag(keystore, secrets):
    """default_action is keep, so an empty tag_actions leaves every standard
    tag alone. Only the standing UID and private-tag policies apply."""
    ds = make_dataset()
    snapshot = {elem.tag: str(elem.value) for elem in ds
                if elem.keyword and "UID" not in elem.keyword}

    ds, _ = run(ds, {}, keystore, secrets)

    for tag, value in snapshot.items():
        assert str(ds[tag].value) == value, f"{tag} changed under default_action keep"


# ── UID policy and private tags ──────────────────────────────────────────────

def test_unnamed_uids_get_the_existing_remap(keystore, secrets):
    """None of the six actions can act on a UID correctly, which is why the UI
    does not send one. The existing deterministic remap applies instead."""
    ds = make_dataset()
    original_study = str(ds.StudyInstanceUID)
    original_sop = str(ds.SOPInstanceUID)

    ds, _ = run(ds, {"PatientName": {"action": "suppress"}}, keystore, secrets)

    assert str(ds.StudyInstanceUID) != original_study
    assert str(ds.SOPInstanceUID) != original_sop
    assert str(ds.StudyInstanceUID).startswith("2.25.")


def test_uid_remap_keeps_a_study_linked_across_files(keystore, secrets):
    """Deterministic: every object of a study stays linked to the others."""
    shared = generate_uid()
    first, _ = run(make_dataset(StudyInstanceUID=shared), {}, keystore, secrets)
    second, _ = run(make_dataset(StudyInstanceUID=shared), {}, keystore, secrets)
    assert first.StudyInstanceUID == second.StudyInstanceUID


def test_sop_class_uid_is_not_remapped(keystore, secrets):
    """Remapping SOPClassUID would make the file unreadable -- it identifies
    the object type, not the patient."""
    ds = make_dataset()
    ds, _ = run(ds, {}, keystore, secrets)
    assert ds.SOPClassUID == CTImageStorage


def test_file_meta_sop_instance_uid_is_kept_in_sync(keystore, secrets):
    """The two are required to match; a stale meta header both invalidates the
    file and leaks the original UID that was just remapped."""
    ds = make_dataset()
    original = str(ds.SOPInstanceUID)
    ds, _ = run(ds, {}, keystore, secrets)

    assert ds.file_meta.MediaStorageSOPInstanceUID == ds.SOPInstanceUID
    assert ds.file_meta.MediaStorageSOPInstanceUID != original


def test_private_tags_are_stripped(keystore, secrets):
    """Invisible to the UI, so a user cannot name them, and vendor private
    blocks routinely carry a copy of the patient demographics."""
    ds = make_dataset()
    block = ds.private_block(0x000B, "ACME PRIVATE", create=True)
    block.add_new(0x01, "LO", "Doe^Jane")

    ds, _ = run(ds, {"PatientName": {"action": "suppress"}}, keystore, secrets)

    assert not [elem for elem in ds if elem.tag.group % 2 == 1]


# ── Multi-valued elements ────────────────────────────────────────────────────

def test_multi_valued_elements_are_transformed_per_item(keystore, secrets):
    """Hashing the repr() of a MultiValue would produce one nonsense value
    where there were three real ones."""
    ds = make_dataset()
    ds.OtherPatientNames = ["Doe^Jane", "Doe^J", "Smith^Jane"]

    ds, _ = run(ds, {"OtherPatientNames": {"action": "hashing_with_salt"}},
                keystore, secrets)

    values = list(ds.OtherPatientNames)
    assert len(values) == 3
    assert all("Doe" not in str(v) for v in values)
    assert len(set(str(v) for v in values)) == 3


# ── Audit / manifest keying ──────────────────────────────────────────────────

def test_audit_techniques_are_the_wire_action_identifiers(keystore, secrets):
    """tags_by_technique is keyed off these, and they line up exactly with the
    six action identifiers the config sent."""
    _, audit = run(make_dataset(), {
        "PatientName": {"action": "suppress"},
        "PatientID": {"action": "hashing_with_salt"},
        "PatientBirthDate": {"action": "masking", "characters_to_mask": [5, 6, 7, 8]},
    }, keystore, secrets)

    techniques = {entry["technique"] for entry in audit}
    assert {"suppress", "hashing_with_salt", "masking"} <= techniques


def test_an_empty_value_is_reported_as_untouched(keystore, secrets):
    """A tag that was already empty was not touched, and must not inflate
    tags_touched."""
    ds = make_dataset(PatientID="")
    _, audit = run(ds, {"PatientID": {"action": "hashing_with_salt"}}, keystore, secrets)

    entries = [e for e in audit if e["field"] == "PatientID"]
    assert entries and entries[0]["changed"] is False


def test_private_tag_stripping_is_not_counted_as_the_users_suppress(keystore, secrets):
    """tags_by_technique is what the UI shows back. Folding a few dozen
    stripped vendor tags into the two suppressions the user chose makes that
    number meaningless."""
    ds = make_dataset()
    block = ds.private_block(0x000B, "ACME PRIVATE", create=True)
    block.add_new(0x01, "LO", "Doe^Jane")
    block.add_new(0x02, "LO", "UHID0012345")

    _, audit = run(ds, {"PatientName": {"action": "suppress"}}, keystore, secrets)

    by_technique = {}
    for entry in audit:
        by_technique[entry["technique"]] = by_technique.get(entry["technique"], 0) + 1

    assert by_technique["suppress"] == 1, "only the user's own choice counts"
    assert by_technique["private_tag_policy"] >= 2
