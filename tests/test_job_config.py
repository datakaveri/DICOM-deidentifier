"""
Tests for parsing the SPIDEr job config document and enforcing the fields
that are fixed server-side (contract sections 1, 2, 3, 5, 7).
"""

import json

import pytest

from de_identification import job_config
from de_identification.job_config import JobConfigError, parse_job_config, load_job_config


def _document(**block_overrides):
    """The contract's canonical document, with the dicom_deidentify block
    optionally overridden."""
    block = {
        "input_dir": "/app/data",
        "output_dir": "/app/output",
        "temp_dir": "/tmp/skald-dicom",
        "emit_pixel_text_report": False,
        "emit_uid_crosswalk": False,
        "fail_on_unparsable": True,
        "pixel": {"redact_burned_in_text": True, "method": "black", "verify": True},
        "default_action": "keep",
        "tag_actions": {
            "PatientName":      {"action": "suppress"},
            "PatientID":        {"action": "hashing_with_salt"},
            "AccessionNumber":  {"action": "hashing_with_salt"},
            "PatientBirthDate": {"action": "masking", "masking_char": "*",
                                 "characters_to_mask": [5, 6, 7, 8]},
            "InstitutionName":  {"action": "suppress"},
        },
    }
    block.update(block_overrides)
    return {
        "data_type": "chest_xray",
        "dataset_name": "chest_xray.dcm",
        "format": "dicom",
        "operations": ["dicom_deidentify", "suppress", "hashing_with_salt", "masking"],
        "dicom_deidentify": block,
    }


# ── Backwards compatibility (section 7) ──────────────────────────────────────

def test_pre_contract_document_takes_the_old_path():
    """The four original top-level fields, no dicom_deidentify block: None,
    meaning main.py runs the original fixed policy unchanged."""
    document = {
        "data_type": "chest_xray",
        "dataset_name": "chest_xray.dcm",
        "operations": ["dicom_deidentify"],
        "format": "dicom",
    }
    assert parse_job_config(document) is None


def test_absent_config_file_takes_the_old_path(tmp_path):
    assert load_job_config(str(tmp_path / "nothing-here.json")) is None


def test_present_but_invalid_config_file_raises(tmp_path):
    """A config that exists and cannot be understood must not quietly fall
    back to a different policy than the user chose."""
    path = tmp_path / "config.json"
    path.write_text("{not json")
    with pytest.raises(JobConfigError):
        load_job_config(str(path))


def test_valid_config_file_round_trips(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(_document()))
    cfg = load_job_config(str(path))
    assert cfg is not None
    assert len(cfg.tag_actions) == 5


# ── Keyword -> tag (section 1) ───────────────────────────────────────────────

def test_keywords_map_to_the_contracts_tags():
    cfg = parse_job_config(_document())
    assert cfg.action_for((0x0010, 0x0010)).action == "suppress"
    assert cfg.action_for((0x0010, 0x0020)).action == "hashing_with_salt"
    assert cfg.action_for((0x0008, 0x0050)).action == "hashing_with_salt"
    assert cfg.action_for((0x0010, 0x0030)).action == "masking"
    assert cfg.action_for((0x0008, 0x0080)).action == "suppress"


@pytest.mark.parametrize("keyword,tag", [
    ("PatientName",             (0x0010, 0x0010)),
    ("PatientID",               (0x0010, 0x0020)),
    ("PatientBirthDate",        (0x0010, 0x0030)),
    ("PatientSex",              (0x0010, 0x0040)),
    ("PatientAge",              (0x0010, 0x1010)),
    ("PatientAddress",          (0x0010, 0x1040)),
    ("PatientTelephoneNumbers", (0x0010, 0x2154)),
    ("PatientWeight",           (0x0010, 0x1030)),
    ("OtherPatientNames",       (0x0010, 0x1001)),
    ("EthnicGroup",             (0x0010, 0x2160)),
    ("InstitutionName",         (0x0008, 0x0080)),
    ("InstitutionAddress",      (0x0008, 0x0081)),
    ("ReferringPhysicianName",  (0x0008, 0x0090)),
    ("PerformingPhysicianName", (0x0008, 0x1050)),
    ("OperatorsName",           (0x0008, 0x1070)),
    ("AccessionNumber",         (0x0008, 0x0050)),
    ("StudyDate",               (0x0008, 0x0020)),
    ("SeriesDate",              (0x0008, 0x0021)),
    ("AcquisitionDate",         (0x0008, 0x0022)),
    ("StudyDescription",        (0x0008, 0x1030)),
    ("SeriesDescription",       (0x0008, 0x103E)),
    ("Modality",                (0x0008, 0x0060)),
    ("Manufacturer",            (0x0008, 0x0070)),
    ("BodyPartExamined",        (0x0018, 0x0015)),
    ("StudyInstanceUID",        (0x0020, 0x000D)),
    ("SeriesInstanceUID",       (0x0020, 0x000E)),
    ("SOPInstanceUID",          (0x0008, 0x0018)),
    ("Rows",                    (0x0028, 0x0010)),
    ("Columns",                 (0x0028, 0x0011)),
    ("BitsAllocated",           (0x0028, 0x0100)),
])
def test_every_keyword_the_ui_can_send_maps_to_its_contract_tag(keyword, tag):
    assert job_config.resolve_keyword(keyword) == tag


def test_pixel_preview_keywords_are_filtered_out():
    """The UI filters these before sending; if one arrives it is ignored, not
    turned into a tag action -- the pixels are the redaction pass's job."""
    cfg = parse_job_config(_document(tag_actions={
        "PatientName": {"action": "suppress"},
        "PixelDataSample": {"action": "suppress"},
        "PixelDataBytes": {"action": "suppress"},
    }))
    assert len(cfg.tag_actions) == 1


def test_keyword_naming_no_dicom_tag_is_recorded_not_applied():
    """The contract's own example carries InsuranceID, which is in neither the
    UI keyword table nor pydicom's dictionary. It names no element, so it is
    surfaced as unresolved rather than failing the batch or being applied to
    something else."""
    cfg = parse_job_config(_document(tag_actions={
        "PatientName": {"action": "suppress"},
        "InsuranceID": {"action": "encrypt", "format_preserving": True},
    }))
    assert cfg.unresolved_keywords == ["InsuranceID"]
    assert len(cfg.tag_actions) == 1


# ── Rejecting unimplemented work at parse time (section 2) ───────────────────

@pytest.mark.parametrize("operation", [
    "generalize", "date_shift", "uid_remap", "free_text_anonymization", "nonsense",
])
def test_unimplemented_operations_are_rejected_at_parse_time(operation):
    """Rejected here, before the first file is opened -- not discovered
    mid-file with half a batch already written."""
    document = _document()
    document["operations"].append(operation)
    with pytest.raises(JobConfigError, match="Unsupported operation"):
        parse_job_config(document)


def test_all_six_actions_are_accepted():
    document = _document(tag_actions={
        "PatientName":       {"action": "suppress"},
        "PatientID":         {"action": "hashing_with_salt"},
        "AccessionNumber":   {"action": "encrypt", "format_preserving": True},
        "PatientBirthDate":  {"action": "masking", "characters_to_mask": [5, 6, 7, 8]},
        "OperatorsName":     {"action": "tokenization"},
        "InstitutionName":   {"action": "charcloak"},
    })
    document["operations"] = [
        "dicom_deidentify", "suppress", "hashing_with_salt", "encrypt",
        "masking", "tokenization", "charcloak",
    ]
    cfg = parse_job_config(document)
    assert len(cfg.tag_actions) == 6


def test_unsupported_tag_action_is_rejected():
    with pytest.raises(JobConfigError, match="unsupported action"):
        parse_job_config(_document(tag_actions={
            "PatientBirthDate": {"action": "generalize"},
        }))


def test_masking_without_positions_is_rejected():
    """Masking nothing would silently keep the value in the clear."""
    with pytest.raises(JobConfigError, match="characters_to_mask"):
        parse_job_config(_document(tag_actions={
            "PatientBirthDate": {"action": "masking", "masking_char": "*"},
        }))


@pytest.mark.parametrize("positions", [[0], [-1], ["5"], [True], [1.5]])
def test_masking_positions_must_be_1_based_positive_integers(positions):
    with pytest.raises(JobConfigError, match="1-based positive integers"):
        parse_job_config(_document(tag_actions={
            "PatientBirthDate": {"action": "masking", "characters_to_mask": positions},
        }))


# ── default_action (section 3) ───────────────────────────────────────────────

def test_default_action_keep_is_the_only_accepted_value():
    """Defaulting unlisted tags to suppress quietly destroys Modality,
    BodyPartExamined and Manufacturer with no way to tell why."""
    assert parse_job_config(_document()).default_action == "keep"
    with pytest.raises(JobConfigError, match="default_action"):
        parse_job_config(_document(default_action="suppress"))


def test_a_tag_absent_from_tag_actions_has_no_action():
    cfg = parse_job_config(_document())
    for tag in [(0x0008, 0x1030),   # StudyDescription
                (0x0008, 0x103E),   # SeriesDescription
                (0x0008, 0x0060),   # Modality
                (0x0018, 0x0015),   # BodyPartExamined
                (0x0008, 0x0070)]:  # Manufacturer
        assert cfg.action_for(tag) is None


def test_a_description_named_explicitly_is_honoured_literally():
    """The UI states descriptions pass through and offers Remove, so a
    suppress on one is a deliberate choice."""
    cfg = parse_job_config(_document(tag_actions={
        "StudyDescription": {"action": "suppress"},
    }))
    assert cfg.action_for((0x0008, 0x1030)).action == "suppress"


# ── Server-side enforcement of the fixed fields (section 5) ──────────────────

@pytest.mark.parametrize("field,tampered", [
    ("temp_dir", "/app/output/tmp"),
    ("output_dir", "/somewhere/else"),
    ("input_dir", "/somewhere/else"),
    ("emit_pixel_text_report", True),
    ("emit_uid_crosswalk", True),
    ("fail_on_unparsable", False),
])
def test_tampered_fixed_fields_are_ignored_in_favour_of_server_constants(field, tampered):
    """A config is just a document and this one arrives from a browser."""
    from config import (
        INPUT_DIR, OUTPUT_DIR, TEMP_DIR, EMIT_PIXEL_TEXT_REPORT,
        EMIT_UID_CROSSWALK, FAIL_ON_UNPARSABLE,
    )
    enforced = {
        "input_dir": INPUT_DIR,
        "output_dir": OUTPUT_DIR,
        "temp_dir": TEMP_DIR,
        "emit_pixel_text_report": EMIT_PIXEL_TEXT_REPORT,
        "emit_uid_crosswalk": EMIT_UID_CROSSWALK,
        "fail_on_unparsable": FAIL_ON_UNPARSABLE,
    }
    cfg = parse_job_config(_document(**{field: tampered}))
    assert getattr(cfg, field) == enforced[field]
    assert getattr(cfg, field) != tampered


def test_temp_dir_stays_under_tmp():
    """Anything beneath the output volume would write the original,
    still-identified DICOM onto the volume that leaves the enclave."""
    cfg = parse_job_config(_document())
    assert cfg.temp_dir.startswith("/tmp")


@pytest.mark.parametrize("tampered", [
    {"redact_burned_in_text": False, "method": "black", "verify": True},
    {"redact_burned_in_text": True, "method": "blur", "verify": True},
    {"redact_burned_in_text": True, "method": "black", "verify": False},
])
def test_pixel_policy_is_fixed_not_a_setting(tampered):
    """Turning redaction off leaves PHI printed into the pixels that no tag
    action can reach, and blur is partially reversible on text."""
    cfg = parse_job_config(_document(pixel=tampered))
    assert cfg.pixel == {"redact_burned_in_text": True, "method": "black", "verify": True}


# ── Malformed documents ──────────────────────────────────────────────────────

@pytest.mark.parametrize("document", [
    "not an object",
    {"dicom_deidentify": "not an object"},
])
def test_malformed_documents_are_rejected(document):
    with pytest.raises(JobConfigError):
        parse_job_config(document)


def test_tag_actions_entry_must_carry_an_action():
    with pytest.raises(JobConfigError, match="missing 'action'"):
        parse_job_config(_document(tag_actions={"PatientName": {}}))
