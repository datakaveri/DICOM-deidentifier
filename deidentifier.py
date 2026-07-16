"""
deidentifier.py — DICOM PHI tag de-identification core module.

Standalone: no external PII report required. Uses a hardcoded, audited
policy table aligned with DICOM Attribute Confidentiality Profiles (PS3.15).

Intended as a library imported by server.py and tests.
"""

import copy
import hashlib

import pydicom
from pydicom.uid import generate_uid

# ── PHI field policy ──────────────────────────────────────────────────────────
# Fields masked with a VR-safe placeholder  (* / 19000101 / 0)
MASK_FIELDS = [
    "PatientName",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientComments",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "RequestingPhysician",
    "InstitutionName",
    "InstitutionAddress",
    "InstitutionalDepartmentName",
    "AdditionalPatientHistory",
    "OtherPatientNames",
    "PatientMotherBirthName",
    "EthnicGroup",
]

# Fields replaced with a truncated SHA-256 hash (pseudonymisation)
HASH_FIELDS = [
    "PatientID",
    "AccessionNumber",
]

# Date fields: retain year only  (YYYYMMDD → YYYY0101)
DATE_FIELDS = [
    "PatientBirthDate",
    "StudyDate",
    "SeriesDate",
    "AcquisitionDate",
]

ALL_PHI_FIELDS = MASK_FIELDS + HASH_FIELDS + DATE_FIELDS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _vr_safe_mask(elem) -> str:
    """Return a placeholder that is legal for the element's VR."""
    vr = elem.VR
    if vr == "DA":
        return "19000101"
    if vr in ("DS", "FL", "FD", "IS", "US", "SS", "UL", "SL"):
        return "0"
    return "*"


def _hash(value: str) -> str:
    return hashlib.sha256(str(value).encode()).hexdigest()[:12]


# ── Public API ────────────────────────────────────────────────────────────────

def deidentify(ds: pydicom.Dataset) -> tuple:
    """
    De-identify PHI tags on a deep copy of ds.

    Returns:
        cleaned_ds  — pydicom.Dataset with PHI removed / pseudonymised
        audit       — list[dict]  one entry per processed field:
                        {"field", "old_value", "action"}
    """
    dc = copy.deepcopy(ds)
    audit = []

    for kw in MASK_FIELDS:
        if not hasattr(dc, kw):
            continue
        old = str(getattr(dc, kw)).strip()
        if not old or old == "None":
            continue
        elem = dc.data_element(kw)
        new = _vr_safe_mask(elem)
        elem.value = new
        audit.append({"field": kw, "old_value": old[:200], "action": "masked"})

    for kw in HASH_FIELDS:
        if not hasattr(dc, kw):
            continue
        old = str(getattr(dc, kw)).strip()
        if not old or old == "None":
            continue
        new = _hash(old)
        setattr(dc, kw, new)
        audit.append({"field": kw, "old_value": old[:200], "action": "hashed"})

    for kw in DATE_FIELDS:
        if not hasattr(dc, kw):
            continue
        old = str(getattr(dc, kw)).strip()
        if not old or old == "None":
            continue
        new = (old[:4] + "0101") if len(old) >= 4 else "19000101"
        setattr(dc, kw, new)
        audit.append({"field": kw, "old_value": old[:200], "action": "date_generalised"})

    # Global de-identification markers
    dc.PatientIdentityRemoved = "YES"
    dc.StudyInstanceUID  = generate_uid()
    dc.SeriesInstanceUID = generate_uid()
    dc.SOPInstanceUID    = generate_uid()

    return dc, audit


def inspect(ds: pydicom.Dataset) -> list:
    """Return a list of dicts for every PHI field present in ds."""
    tags = []
    for kw in ALL_PHI_FIELDS:
        if hasattr(ds, kw):
            val = str(getattr(ds, kw)).strip()
            if val and val != "None":
                tags.append({"field": kw, "value": val[:200]})
    return tags
