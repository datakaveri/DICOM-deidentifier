"""
phi_tags.py — Step 1: PHI tag identification (read-only — no tag value is
ever modified) plus the Step 3 cross-check against burned-in OCR text.
"""

import os
import json
import re

import pydicom

from config import log, EXTRA_ID_FIELDS
from classify import _is_clinical


def _is_id_field(keyword: str) -> bool:
    """True for identifier-style keywords, excluding UIDs (SOPInstanceUID, ...)."""
    if not keyword:
        return False
    if keyword.endswith("UID"):
        return False
    if keyword.endswith("ID") or keyword.endswith("IDs"):
        return True
    return keyword in EXTRA_ID_FIELDS


def identify_phi_tags(ds: pydicom.Dataset) -> list:
    """
    Scans every tag to find which ones carry PHI, by the same VR/keyword
    rules as before (not a hardcoded per-file tag list) — but records the
    match instead of changing anything. No tag is hashed, removed, or
    generalized here or anywhere else in this pipeline; ds is never mutated.

    Categories: time, date, age, accession, weight, description, address,
    id, name.

    Returns a list of {"tag", "field", "category", "value"} dicts — the
    original values Step 3 later searches for in the image's burned-in text.
    """
    found = []

    for elem in ds.iterall():
        if elem.VR == "SQ":
            continue

        keyword = elem.keyword
        value = str(elem.value).strip()
        if value == "" or value == "None":
            continue

        if elem.VR == "TM":
            category = "time"
        elif elem.VR == "DA":
            category = "date"
        elif elem.VR == "AS" or keyword == "PatientAge":
            category = "age"
        elif keyword == "AccessionNumber":
            category = "accession"
        elif "weight" in keyword.lower():
            category = "weight"
        elif "description" in keyword.lower():
            category = "description"
        elif "address" in keyword.lower():
            category = "address"
        elif keyword == "PatientID" or _is_id_field(keyword):
            category = "id"
        elif elem.VR == "PN" or "name" in keyword.lower():
            category = "name"
        else:
            continue

        found.append({
            "tag": str(elem.tag),
            "field": keyword,
            "category": category,
            "value": value,
        })

    return found


def dump_original_tags(ds: pydicom.Dataset, path: str) -> None:
    """Snapshot every tag's original value to a JSON file, before any
    de-identification runs. Bulk binary data (pixel data, VR OB/OW/UN) is
    skipped since it isn't PHI in itself and isn't useful in a text snapshot.
    """
    snapshot = []
    for elem in ds.iterall():
        if elem.VR in ("SQ", "OB", "OW", "UN") or elem.keyword == "PixelData":
            continue
        try:
            value = str(elem.value)
        except Exception:
            value = "<unreadable>"
        snapshot.append({
            "tag": str(elem.tag),
            "keyword": elem.keyword,
            "vr": elem.VR,
            "value": value,
        })

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(snapshot, f, indent=2)


def load_original_tag_values(path: str) -> list:
    """
    Loads the full original-tag backup (data.json, written by
    dump_original_tags() before anything runs) and returns the values to
    match burned-in OCR text against.
    
    Includes comprehensive DICOM normalization:
      - Person Names (PN): 'MEYER^STEPHANIE' -> 'MEYER STEPHANIE', 'STEPHANIE MEYER',
        and individual parts 'MEYER', 'STEPHANIE'.
      - Dates (DA): '19530716' -> '07.16.1953', '16.07.1953', '07/16/1953', '16/07/1953', etc.
      - Identifiers: PatientID, AccessionNumber, etc.
    """
    if not os.path.exists(path):
        return []

    with open(path, "r") as f:
        snapshot = json.load(f)

    values = set()
    for entry in snapshot:
        val = (entry.get("value") or "").strip()
        keyword = entry.get("keyword", "")
        vr = entry.get("vr", "")
        if not val or val == "None" or len(val) < 3:
            continue
        if vr == "UI" or keyword.endswith("UID"):
            continue
        if _is_clinical(val):
            continue

        values.add(val)

        # Handle DICOM Person Names (PN VR or keyword ending with Name)
        if vr == "PN" or keyword.endswith("Name") or "^" in val:
            # Replace carets with spaces
            clean_name = val.replace("^", " ").strip()
            if clean_name and len(clean_name) >= 3 and not _is_clinical(clean_name):
                values.add(clean_name)
                parts = clean_name.split()
                if len(parts) >= 2:
                    # Also add reversed name ("STEPHANIE MEYER")
                    values.add(" ".join(reversed(parts)))
                for part in parts:
                    if len(part) >= 3 and not _is_clinical(part):
                        values.add(part)

        # Handle DICOM Dates (YYYYMMDD)
        if (vr == "DA" or keyword.endswith("Date")) and len(val) == 8 and val.isdigit():
            yyyy = val[:4]
            mm = val[4:6]
            dd = val[6:8]
            # Add various common burned-in date formats
            values.add(f"{mm}.{dd}.{yyyy}")
            values.add(f"{dd}.{mm}.{yyyy}")
            values.add(f"{mm}/{dd}/{yyyy}")
            values.add(f"{dd}/{mm}/{yyyy}")
            values.add(f"{yyyy}-{mm}-{dd}")
            values.add(f"{yyyy}.{mm}.{dd}")

    return list(values)


def match_against_stored_tags(merged, stored_values, image_shape):
    """
    Step 3: cross-checks OCR-detected burned-in text against the PHI tag
    values identify_phi_tags() found in Step 1 -- e.g. this file's actual
    PatientName, PatientID, InstitutionName, dates, etc.
    """
    if not stored_values:
        return []

    h, w = image_shape[:2]
    normalized_stored = {v.strip().upper() for v in stored_values if len(v.strip()) >= 3}
    matches = []

    for det in merged:
        text = det["text"].strip()
        if not text or len(text) < 3:
            continue
        if _is_clinical(text):
            continue

        norm_text = text.upper()
        # Clean alphanumeric tokens from the OCR text
        tokens = [t for t in re.findall(r'[A-Z0-9]+', norm_text) if len(t) >= 3 and not _is_clinical(t)]

        # Check full substring matches
        hit = any(
            norm_text == sv or (len(sv) >= 4 and sv in norm_text) or (len(norm_text) >= 4 and norm_text in sv)
            for sv in normalized_stored
        )

        # Check token-level matches (e.g. surname or given name matching a stored token)
        if not hit:
            for tok in tokens:
                if tok in normalized_stored:
                    hit = True
                    break

        if not hit:
            continue

        bbox = det["bbox"]
        log.info(f"    REDACT-TAG-MATCH: '{text}' matches stored original tag value")
        matches.append({"text": text, "bbox": bbox})

    return matches
