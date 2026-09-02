"""
job_config.py -- parses the SPIDEr job config document for
`application: "skald_dicom"`.

The document reaches this container already decrypted: the middleware
dispatches the job, decrypts `config_ciphertext` (Fernet) and drops the
resulting JSON on the CONFIG_DIR mount (config.JOB_CONFIG_FILE, i.e.
/app/config/config.json in the image). Nothing here decrypts anything.

Two shapes are supported, and which one arrives decides which policy runs:

  - No `dicom_deidentify` block (the pre-contract document: data_type,
    dataset_name, operations, format) -- `load_job_config` returns None and
    main.py runs the original fixed-policy path, byte-for-byte unchanged.
  - With a `dicom_deidentify` block -- the user picked per-tag actions in the
    UI and `tag_actions` drives the run.

Only `tag_actions` (and `default_action`, always "keep") is read from the
wire. Every other field in the block -- input_dir, output_dir, temp_dir,
emit_pixel_text_report, emit_uid_crosswalk, fail_on_unparsable and the whole
`pixel` object -- is fixed policy, taken from config.py and enforced here: a
wire value that disagrees is logged and discarded rather than honoured. A
config is just a document and this one came from a browser.
"""

import json
import os

from pydicom.datadict import tag_for_keyword

# Imported as a module, not as names: the enforced constants are read at call
# time so a config.py reloaded with different env vars (SKALD_OUTPUT_DIR and
# friends) is honoured, rather than a snapshot taken when this module first
# loaded.
import config
from config import log


class JobConfigError(ValueError):
    """Raised for a malformed or unsupported job config. Fatal at parse time:
    a config describing work this application cannot do must fail before the
    first file is opened, not halfway through a batch."""


# ── Wire action identifiers ──────────────────────────────────────────────────
# The six k-anonymisation techniques the UI can assign to a tag. These are the
# same names, with the same meanings, as in the tabular (CSV) pipeline -- each
# one below dispatches to the SKALD primitive that already implements it (see
# actions.py). `tags_by_technique` in the manifest is keyed off these too.
SUPPRESS = "suppress"
HASHING_WITH_SALT = "hashing_with_salt"
ENCRYPT = "encrypt"
MASKING = "masking"
TOKENIZATION = "tokenization"
CHARCLOAK = "charcloak"

SUPPORTED_ACTIONS = frozenset({
    SUPPRESS, HASHING_WITH_SALT, ENCRYPT, MASKING, TOKENIZATION, CHARCLOAK,
})

# The operation that selects this application at all; never a tag action.
DICOM_DEIDENTIFY = "dicom_deidentify"

# `default_action` is always "keep": a tag absent from tag_actions is written
# through unchanged. Defaulting unlisted tags to suppress would quietly destroy
# Modality/BodyPartExamined/Manufacturer with no way for the user to tell why.
KEEP = "keep"


# ── Keyword -> tag ───────────────────────────────────────────────────────────
# tag_actions is keyed by tag KEYWORD, not (gggg,eeee): that is what the
# browser parses out of the preview and what the user actually selected.
# This is the complete set the UI can send.
KEYWORD_TO_TAG = {
    # Patient identity & demographics
    "PatientName":             (0x0010, 0x0010),
    "PatientID":               (0x0010, 0x0020),
    "PatientBirthDate":        (0x0010, 0x0030),
    "PatientSex":              (0x0010, 0x0040),
    "PatientAge":              (0x0010, 0x1010),
    "PatientAddress":          (0x0010, 0x1040),
    "PatientTelephoneNumbers": (0x0010, 0x2154),
    "PatientWeight":           (0x0010, 0x1030),
    "OtherPatientNames":       (0x0010, 0x1001),
    "EthnicGroup":             (0x0010, 0x2160),
    # Institutions & staff
    "InstitutionName":         (0x0008, 0x0080),
    "InstitutionAddress":      (0x0008, 0x0081),
    "ReferringPhysicianName":  (0x0008, 0x0090),
    "PerformingPhysicianName": (0x0008, 0x1050),
    "OperatorsName":           (0x0008, 0x1070),
    # Order / study identifiers & dates
    "AccessionNumber":         (0x0008, 0x0050),
    "StudyDate":               (0x0008, 0x0020),
    "SeriesDate":              (0x0008, 0x0021),
    "AcquisitionDate":         (0x0008, 0x0022),
    # Descriptions (free text)
    "StudyDescription":        (0x0008, 0x1030),
    "SeriesDescription":       (0x0008, 0x103E),
    # Acquisition context -- not PHI, but assignable
    "Modality":                (0x0008, 0x0060),
    "Manufacturer":            (0x0008, 0x0070),
    "BodyPartExamined":        (0x0018, 0x0015),
    # UIDs
    "StudyInstanceUID":        (0x0020, 0x000D),
    "SeriesInstanceUID":       (0x0020, 0x000E),
    "SOPInstanceUID":          (0x0008, 0x0018),
    # Image geometry
    "Rows":                    (0x0028, 0x0010),
    "Columns":                 (0x0028, 0x0011),
    "BitsAllocated":           (0x0028, 0x0100),
}

# The UI shows these in its preview and filters them out before sending, so
# they should never appear in tag_actions. Ignored with a warning rather than
# rejected -- the pixels are the redaction pass's job either way, and failing
# a whole batch over a cosmetic UI regression helps nobody.
PIXEL_PREVIEW_KEYWORDS = frozenset({"PixelDataSample", "PixelDataBytes"})

# Type 1 attributes: an empty value makes the file invalid, so `suppress` on
# one produces a broken DICOM. A user can assign an action anyway and the UI
# warns the result may not load; that choice is honoured literally, with a
# warning logged here so the reason is discoverable afterwards.
TYPE_1_KEYWORDS = frozenset({
    "StudyInstanceUID", "SeriesInstanceUID", "SOPInstanceUID",
    "Rows", "Columns", "BitsAllocated",
})


class TagAction:
    """One entry of `tag_actions`, resolved to a tag and validated."""

    def __init__(self, keyword, tag, action, masking_char="*",
                 characters_to_mask=None, format_preserving=False):
        self.keyword = keyword
        self.tag = tag
        self.action = action
        self.masking_char = masking_char
        self.characters_to_mask = characters_to_mask or []
        self.format_preserving = format_preserving

    def __repr__(self):
        return f"<TagAction {self.keyword} {self.action}>"


class JobConfig:
    """A parsed job config carrying a `dicom_deidentify` block.

    `tag_actions` maps (group, element) -> TagAction. The fixed fields are
    exposed as attributes so callers read the enforced constants through the
    same object, never the wire values.
    """

    def __init__(self, data_type=None, dataset_name=None, fmt=None,
                 operations=None, default_action=KEEP, tag_actions=None,
                 ingest_mode=None, unresolved_keywords=None):
        self.data_type = data_type
        self.dataset_name = dataset_name
        self.format = fmt
        self.operations = operations or []
        self.default_action = default_action
        self.tag_actions = tag_actions or {}
        self.ingest_mode = ingest_mode
        # Keywords from tag_actions that name no DICOM tag. Surfaced in the
        # manifest so a UI sending one can see it was not applied.
        self.unresolved_keywords = unresolved_keywords or []

        # Fixed policy, from config.py -- never from the document.
        self.input_dir = config.INPUT_DIR
        self.output_dir = config.OUTPUT_DIR
        self.temp_dir = config.TEMP_DIR
        self.emit_pixel_text_report = config.EMIT_PIXEL_TEXT_REPORT
        self.emit_uid_crosswalk = config.EMIT_UID_CROSSWALK
        self.fail_on_unparsable = config.FAIL_ON_UNPARSABLE
        self.pixel = dict(config.PIXEL_POLICY)

    def action_for(self, tag_id):
        """The TagAction for `tag_id`, or None -- meaning "not listed", which
        under default_action "keep" means write the tag through unchanged."""
        return self.tag_actions.get(tag_id)


# ── Fixed-field enforcement ──────────────────────────────────────────────────

def _enforce_fixed_fields(block):
    """Logs every fixed field whose wire value disagrees with the enforced
    constant. The wire values are discarded either way -- JobConfig always
    reads from config.py -- so this exists to make tampering (or a UI bug)
    visible in the run log rather than silent."""
    fixed = {
        "input_dir": config.INPUT_DIR,
        "output_dir": config.OUTPUT_DIR,
        "temp_dir": config.TEMP_DIR,
        "emit_pixel_text_report": config.EMIT_PIXEL_TEXT_REPORT,
        "emit_uid_crosswalk": config.EMIT_UID_CROSSWALK,
        "fail_on_unparsable": config.FAIL_ON_UNPARSABLE,
    }
    for field, enforced in fixed.items():
        if field in block and block[field] != enforced:
            log.warning(
                f"  [job config] Ignoring wire value for fixed field '{field}': "
                f"{block[field]!r} -- enforcing {enforced!r} from server constants."
            )

    wire_pixel = block.get("pixel")
    if isinstance(wire_pixel, dict):
        for field, enforced in config.PIXEL_POLICY.items():
            if field in wire_pixel and wire_pixel[field] != enforced:
                log.warning(
                    f"  [job config] Ignoring wire value for fixed field "
                    f"'pixel.{field}': {wire_pixel[field]!r} -- enforcing "
                    f"{enforced!r} from server constants."
                )

    # A temp_dir outside /tmp is the one enforcement failure that cannot be
    # logged-and-continued: the still-identified checkpoint would land on a
    # volume that leaves the enclave.
    if not os.path.abspath(config.TEMP_DIR).startswith("/tmp" + os.sep):
        raise JobConfigError(
            f"TEMP_DIR must live under /tmp (got {config.TEMP_DIR!r}): the "
            f"pre-de-identification checkpoint is written there and must never "
            f"reach the output volume."
        )


# ── Parsing ──────────────────────────────────────────────────────────────────

def _parse_operations(raw):
    """Validates `operations`. Every entry must be one this application
    actually implements -- an unimplemented technique is rejected here, at
    parse time, rather than discovered mid-file with half a batch written."""
    if raw is None:
        return []
    if not isinstance(raw, list) or not all(isinstance(op, str) for op in raw):
        raise JobConfigError("'operations' must be a list of strings")

    unknown = [op for op in raw if op != DICOM_DEIDENTIFY and op not in SUPPORTED_ACTIONS]
    if unknown:
        raise JobConfigError(
            f"Unsupported operation(s) {unknown}: this application implements "
            f"{sorted(SUPPORTED_ACTIONS)} plus '{DICOM_DEIDENTIFY}'. "
            f"There is deliberately no generalize, date_shift, uid_remap or "
            f"free_text_anonymization -- coarsening a date is expressed as "
            f"'masking' over its month/day positions."
        )
    return raw


def resolve_keyword(keyword):
    """(group, element) for `keyword`, or None if it names no DICOM tag.

    KEYWORD_TO_TAG above is the set the UI sends today and is authoritative for
    it. Anything else falls back to pydicom's own data dictionary, so a keyword
    the UI adds later resolves without a change here.

    Returns None for a keyword neither knows -- the contract's own example
    config carries one ("InsuranceID", which is in no DICOM dictionary and not
    in the contract's keyword table either). A keyword that names no tag can
    match no element in any dataset, so there is nothing for an action to act
    on and nothing it could leave behind in the clear.
    """
    tag = KEYWORD_TO_TAG.get(keyword)
    if tag is not None:
        return tag

    resolved = tag_for_keyword(keyword)
    if resolved is None:
        return None
    return (resolved >> 16, resolved & 0xFFFF)


def _parse_tag_action(keyword, entry):
    """Parses one tag_actions entry into a TagAction, or None if the keyword
    names no DICOM tag."""
    if not isinstance(entry, dict):
        raise JobConfigError(f"tag_actions['{keyword}'] must be an object")

    action = entry.get("action")
    if not isinstance(action, str) or not action:
        raise JobConfigError(f"tag_actions['{keyword}'] is missing 'action'")
    if action not in SUPPORTED_ACTIONS:
        raise JobConfigError(
            f"tag_actions['{keyword}'] requests unsupported action {action!r}; "
            f"supported: {sorted(SUPPORTED_ACTIONS)}"
        )

    tag = resolve_keyword(keyword)
    if tag is None:
        log.warning(
            f"  [job config] tag_actions['{keyword}'] names no DICOM tag in the "
            f"UI's keyword set or pydicom's dictionary, so no element can match "
            f"it. Recorded as unresolved and skipped; the '{action}' action is "
            f"not silently applied to something else."
        )
        return None

    masking_char = "*"
    characters_to_mask = []
    if action == MASKING:
        raw_char = entry.get("masking_char", "*")
        if not isinstance(raw_char, str) or not raw_char:
            raise JobConfigError(
                f"tag_actions['{keyword}'].masking_char must be a non-empty string"
            )
        masking_char = raw_char[0]

        raw_positions = entry.get("characters_to_mask")
        if not isinstance(raw_positions, list) or not raw_positions:
            raise JobConfigError(
                f"tag_actions['{keyword}'] uses 'masking' but gives no "
                f"'characters_to_mask'; masking nothing would silently keep the "
                f"value in the clear"
            )
        for pos in raw_positions:
            # bool is an int subclass; a JSON `true` here is a config error.
            if isinstance(pos, bool) or not isinstance(pos, int) or pos < 1:
                raise JobConfigError(
                    f"tag_actions['{keyword}'].characters_to_mask must be "
                    f"1-based positive integers (got {pos!r})"
                )
            characters_to_mask.append(pos)

    format_preserving = False
    if action == ENCRYPT:
        format_preserving = bool(entry.get("format_preserving", False))

    if keyword in TYPE_1_KEYWORDS:
        log.warning(
            f"  [job config] {keyword} is a Type 1 attribute and was assigned "
            f"'{action}'. Honouring it literally as configured -- the resulting "
            f"file may not load in strict readers."
        )

    return TagAction(
        keyword=keyword, tag=tag, action=action, masking_char=masking_char,
        characters_to_mask=characters_to_mask, format_preserving=format_preserving,
    )


def _parse_tag_actions(raw):
    """Returns (tag_actions, unresolved_keywords)."""
    if raw is None:
        return {}, []
    if not isinstance(raw, dict):
        raise JobConfigError("'tag_actions' must be an object keyed by tag keyword")

    parsed = {}
    unresolved = []
    for keyword, entry in raw.items():
        if keyword in PIXEL_PREVIEW_KEYWORDS:
            log.warning(
                f"  [job config] Ignoring tag_actions['{keyword}']: pixel data is "
                f"the burned-in-text redaction pass's job, not a tag action."
            )
            continue
        tag_action = _parse_tag_action(keyword, entry)
        if tag_action is None:
            unresolved.append(keyword)
            continue
        parsed[tag_action.tag] = tag_action
    return parsed, unresolved


def parse_job_config(document):
    """Parses a decrypted job config `document` (a dict).

    Returns a JobConfig when the document carries a `dicom_deidentify` block,
    or None when it does not -- the pre-contract shape, which means "run the
    original fixed policy" (contract section 7).

    Raises JobConfigError for a document that is malformed, or that asks for
    work this application does not implement.
    """
    if not isinstance(document, dict):
        raise JobConfigError("Job config must be a JSON object")

    block = document.get(DICOM_DEIDENTIFY)
    if block is None:
        log.info(
            "  [job config] No 'dicom_deidentify' block -- running the original "
            "fixed-policy de-identification path."
        )
        return None
    if not isinstance(block, dict):
        raise JobConfigError("'dicom_deidentify' must be an object")

    _enforce_fixed_fields(block)

    default_action = block.get("default_action", KEEP)
    if default_action != KEEP:
        raise JobConfigError(
            f"default_action must be {KEEP!r} (got {default_action!r}): a tag "
            f"absent from tag_actions is written through unchanged. Defaulting "
            f"unlisted tags to anything else quietly destroys Modality, "
            f"BodyPartExamined and Manufacturer with no way to tell why."
        )

    operations = _parse_operations(document.get("operations"))
    tag_actions, unresolved = _parse_tag_actions(block.get("tag_actions"))

    # operations is documented as "dicom_deidentify plus every action used
    # below". A mismatch is a UI inconsistency, not a safety problem --
    # tag_actions is what actually runs -- so warn rather than fail the batch.
    declared = set(operations)
    used = {ta.action for ta in tag_actions.values()}
    undeclared = sorted(used - declared)
    if undeclared:
        log.warning(
            f"  [job config] Action(s) {undeclared} appear in tag_actions but not "
            f"in 'operations'. Applying them as configured."
        )

    cfg = JobConfig(
        data_type=document.get("data_type"),
        dataset_name=document.get("dataset_name"),
        fmt=document.get("format"),
        operations=operations,
        default_action=default_action,
        tag_actions=tag_actions,
        ingest_mode=document.get("ingest_mode"),
        unresolved_keywords=unresolved,
    )

    log.info(
        f"  [job config] Loaded: {len(cfg.tag_actions)} tag action(s), "
        f"default_action={cfg.default_action}. Fixed policy enforced from "
        f"server constants (temp_dir={cfg.temp_dir}, "
        f"emit_pixel_text_report={cfg.emit_pixel_text_report}, "
        f"emit_uid_crosswalk={cfg.emit_uid_crosswalk}, "
        f"fail_on_unparsable={cfg.fail_on_unparsable}, pixel={cfg.pixel})."
    )
    return cfg


def load_job_config(path=None):
    """Loads and parses the decrypted job config from `path`
    (config.JOB_CONFIG_FILE by default).

    Returns None when the file is absent or carries no `dicom_deidentify`
    block -- both mean "run the original fixed policy". A file that exists but
    is unreadable or invalid raises: a config that cannot be understood must
    not silently fall back to a different policy than the user chose.
    """
    path = path or config.JOB_CONFIG_FILE
    if not os.path.exists(path):
        log.info(
            f"  [job config] No job config at {path} -- running the original "
            f"fixed-policy de-identification path."
        )
        return None

    try:
        with open(path, "r", encoding="utf-8") as f:
            document = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise JobConfigError(f"Could not read job config at {path}: {e}") from e

    return parse_job_config(document)
