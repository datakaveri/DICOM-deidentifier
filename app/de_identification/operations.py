"""
operations.py — executes one anonymization technique against one DICOM
element's value. This is the Python counterpart of SKALD's per-column
transform steps in preprocess/mod.rs, adapted from "one technique per CSV
column" to "one technique per DICOM tag".

Design notes:
  - hash is plain, unsalted SHA-256 (crypto.hash_hex). This keeps it
    deterministic across separate script runs (no salt state to lose),
    which is required so Study/Series/SOP/Frame-of-Reference UID hashes
    stay consistent whenever the *same* original UID is re-hashed while
    de-identifying multiple files of the same study.
  - UID-VR tags (and any tag in tag_mapping.UID_VALUED_TAGS) are hashed
    into a syntactically valid DICOM UID ("2.25.<int>", the standard
    UUID-derived-UID form — see PS3.5 Annex B) instead of a raw hex
    digest, since a UI-VR element must stay a dotted numeric string.
  - Non-UID hash results are truncated to the target VR's max length
    (VR_MAX_LENGTH) so the mutated value still fits its declared VR.
"""

import uuid

from .tag_mapping import (
    SUPPRESS, HASH, TOKENISE, ENCRYPT, ENCRYPT_FPE, CHARCLOAK, MASK, SCRUB, RETAIN,
    UID_VALUED_TAGS,
)
from .crypto import hash_hex, pseudo_encrypt, format_preserving_encrypt_general, randomize_preserving_class
from .masking import MaskingConfigLite, RegexPatternConfig, RegexPatternKind, apply_masking_value

# Max character length per VR (DICOM PS3.5 Table 6.2-1), used to keep
# hashed values valid for their declared VR.
VR_MAX_LENGTH = {
    "AE": 16, "AS": 4, "CS": 16, "DA": 8, "DS": 16, "DT": 26, "FL": 4, "FD": 8,
    "IS": 12, "LO": 64, "LT": 10240, "PN": 64, "SH": 16, "ST": 1024, "TM": 16,
    "UI": 64, "UL": 4, "US": 2,
}

# ── "mask" technique config: DA-VR dates (YYYYMMDD) ─────────────────────────
# Retains year+month, zeroes the day (chars 7-8) via the "characters" step —
# same masking.py engine SKALD uses for CSV columns, just configured for a
# DICOM date instead of a delimiter-bound column.
DATE_MASK_CONFIG = MaskingConfigLite(
    column="dicom_date",
    masking_char="0",
    characters_to_mask=[7, 8],
    apply_order=["characters"],
)

# ── "scrub" technique config: free text (Allergies, Image Comments, ...) ───
# Same Indian-health-context + universal PII patterns used by the burned-in
# pixel pipeline's classify.py, run through the ported regex-masking engine
# (full-match masking, since none of these patterns declare mask_groups).
_FREE_TEXT_PII_PATTERNS = {
    "aadhaar":     r'\b\d{4}\s?\d{4}\s?\d{4}\b',
    "abha":        r'\b\d{2}-\d{4}-\d{4}-\d{4}\b',
    "phone":       r'\b(?:\+91|0)?[6-9]\d{9}\b',
    "date":        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b',
    "uhid_mrn":    r'\b(?:UHID|MRN|REG|IPD|OPD|CR|PID|HID)[\s:/-]?\d+\b',
    "age_sex":     r'\b\d{1,3}\s*[/]\s*[MFO]\b',
    "name_prefix": r'\b(?:DR\.?|MR\.?|MRS\.?|MS\.?|SHRI|SMT|S/O|D/O|W/O)\b',
    "accession":   r'\b(?:ACC|ACCNO|ACCESSION)[\s:.-]?\w+\b',
}

FREE_TEXT_SCRUB_CONFIG = MaskingConfigLite(
    column="dicom_free_text",
    masking_char="*",
    apply_order=["regex"],
    regex_patterns=[
        RegexPatternConfig(kind=RegexPatternKind.literal(pattern), masking_char="*")
        for pattern in _FREE_TEXT_PII_PATTERNS.values()
    ],
)


def hash_to_uid(value: str) -> str:
    """
    Deterministically derives a valid DICOM UID ('2.25.<int>') from `value`,
    using the same SKALD-ported crypto.hash_hex() as every other hash
    technique — just reshaped into a dotted-numeric UID afterwards, since a
    UI-VR element can't hold a raw hex digest.
    """
    digest_hex = hash_hex(value)
    digest_bytes = bytes.fromhex(digest_hex)[:16]
    derived = uuid.UUID(bytes=digest_bytes)
    return ("2.25." + str(derived.int))[:64]


def hash_value(value: str, vr: str) -> str:
    """Hex SHA-256 digest of `value`, truncated to fit VR's max length."""
    digest = hash_hex(value)
    max_len = VR_MAX_LENGTH.get(vr, 64)
    return digest[:max_len]


def tag_column_key(elem) -> str:
    """A stable per-tag key for vault/key-store lookups (keyword, else GGGGEEEE)."""
    return elem.keyword or f"{elem.tag.group:04X}{elem.tag.element:04X}"


def apply_technique(technique: str, value: str, elem, keystore, tag_id) -> str:
    """Applies `technique` to `value` for element `elem`, returning the new value."""
    column = tag_column_key(elem)

    if technique == RETAIN:
        return value

    if technique == HASH:
        if tag_id in UID_VALUED_TAGS or elem.VR == "UI":
            return hash_to_uid(value)
        return hash_value(value, elem.VR)

    if technique == TOKENISE:
        return keystore.tokenise(column, value)

    if technique == ENCRYPT:
        key = keystore.get_or_create_key(keystore.symmetric_keys, column)
        return pseudo_encrypt(value, key, column)

    if technique == ENCRYPT_FPE:
        key = keystore.get_or_create_key(keystore.fpe_encrypt_keys, column)
        return format_preserving_encrypt_general(value, key, column)

    if technique == CHARCLOAK:
        return randomize_preserving_class(value)

    if technique == MASK:
        return apply_masking_value(value, DATE_MASK_CONFIG, randomize_preserving_class)

    if technique == SCRUB:
        return apply_masking_value(value, FREE_TEXT_SCRUB_CONFIG, randomize_preserving_class)

    # Unknown technique: defensive no-op, leave value untouched.
    return value
