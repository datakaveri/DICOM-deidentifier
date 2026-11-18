#!/usr/bin/env python3
"""Inventory DICOM folders before a batch run.

For every *.dcm under each folder (headers only, pixels never decoded):
  - one row in an Excel sheet per folder, one column per tag (keyword), plus
    the chest decision and its reason;
  - the relative paths of the chest DICOMs to process, and of the JPG that
    sits next to each one (same name, .jpg), for upload and pass-through.

Chest rule (conservative: when the tags disagree, the file is not processed):
  1. an image object: has Rows/Columns and is not a presentation state (PR);
  2. BodyPartExamined is CHEST;
  3. StudyDescription mentions CHEST, or is empty;
  4. SeriesDescription names no other anatomy (HAND, KNEE, SPINE, ...).

The workbook holds every original tag value, i.e. patient data: keep it where
the DICOMs themselves are allowed to be.

Usage:
  inventory.py --out DIR FOLDER [FOLDER ...]
Writes DIR/DICOM_tag_inventory.xlsx, DIR/selection/<folder>.dcm.txt,
DIR/selection/<folder>.jpg.txt and DIR/selection/summary.json.
"""

import argparse
import json
import os
import re
import sys
import time

import pydicom
from openpyxl import Workbook

OTHER_ANATOMY = re.compile(
    r"\b(HAND|FINGER|THUMB|WRIST|FOREARM|ELBOW|HUMERUS|SHOULDER|CLAVICLE|SCAPULA|"
    r"FOOT|TOE|ANKLE|HEEL|CALCANEUM|TIBIA|FIBULA|KNEE|PATELLA|FEMUR|HIP|PELVIS|"
    r"SPINE|CERVICAL|THORACIC SPINE|LUMBAR|L S|LS|SACRUM|COCCYX|SKULL|NASAL|SINUS|"
    r"PNS|MANDIBLE|FACE|ORBIT|ABDOMEN|KUB)\b"
)
MAX_CELL = 32000  # Excel's limit is 32767 characters
FIXED_COLUMNS = [
    "RelativePath",
    "IsChest",
    "Decision",
    "Reason",
    "SOPClass",
    "FileSizeBytes",
]


def chest_decision(ds):
    """(is_chest, reason)"""
    modality = str(ds.get("Modality", "")).upper().strip()
    if "Rows" not in ds or modality == "PR":
        return False, f"not an image object (Modality={modality or 'none'})"
    body = str(ds.get("BodyPartExamined", "")).upper().strip()
    if body != "CHEST":
        return False, f"BodyPartExamined={body or 'empty'}"
    study = str(ds.get("StudyDescription", "")).upper().strip()
    if study and "CHEST" not in study:
        return False, f"StudyDescription does not mention CHEST ({study[:60]})"
    series = str(ds.get("SeriesDescription", "")).upper().strip()
    m = OTHER_ANATOMY.search(series)
    if m:
        return False, f"SeriesDescription names other anatomy ({m.group(0)})"
    return True, "image; BodyPartExamined=CHEST; descriptions consistent"


def cell_value(elem):
    if elem.VR == "SQ":
        return f"<sequence: {len(elem.value)} item(s)>"
    if elem.VR in ("OB", "OW", "OF", "OD", "OL", "UN") or elem.tag == 0x7FE00010:
        n = len(elem.value) if elem.value is not None else 0
        return f"<binary: {n} bytes>"
    v = elem.value
    if isinstance(v, (list, pydicom.multival.MultiValue)):
        v = "\\".join(str(x) for x in v)
    s = str(v) if v is not None else ""
    # Strip characters Excel rejects (control characters).
    s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
    return s[:MAX_CELL]


def column_name(elem):
    if elem.keyword:
        return elem.keyword
    return f"({elem.tag.group:04X},{elem.tag.element:04X})"


def scan(folder):
    rows, dcm_sel, jpg_sel = [], [], []
    columns = {}
    for dirpath, dirnames, files in os.walk(folder):
        dirnames.sort()
        for name in sorted(files):
            if not name.lower().endswith(".dcm"):
                continue
            path = os.path.join(dirpath, name)
            rel = os.path.relpath(path, folder)
            row = {"RelativePath": rel, "FileSizeBytes": os.path.getsize(path)}
            try:
                ds = pydicom.dcmread(path, stop_before_pixels=True)
            except Exception as e:
                row.update(
                    IsChest=False, Decision="not anonymised", Reason=f"unreadable: {e}"
                )
                rows.append(row)
                continue
            is_chest, reason = chest_decision(ds)
            row.update(
                IsChest=is_chest,
                Decision="anonymised" if is_chest else "not anonymised (not chest)",
                Reason=reason,
                SOPClass=getattr(ds.get("SOPClassUID"), "name", ""),
            )
            for meta in ds.file_meta:
                col = "meta." + column_name(meta)
                columns.setdefault(col, None)
                row[col] = cell_value(meta)
            for elem in ds:
                col = column_name(elem)
                columns.setdefault(col, None)
                row[col] = cell_value(elem)
            rows.append(row)
            if is_chest:
                dcm_sel.append(rel)
                jpg = os.path.splitext(path)[0] + ".jpg"
                if os.path.exists(jpg):
                    jpg_sel.append(os.path.relpath(jpg, folder))
    return rows, list(columns), dcm_sel, jpg_sel


def sheet_title(folder, used):
    base = re.sub(r"[\[\]:*?/\\]", "_", os.path.basename(os.path.normpath(folder)))[:31]
    title, n = base, 2
    while title in used:
        title = f"{base[:28]}_{n}"
        n += 1
    used.add(title)
    return title


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", required=True)
    ap.add_argument("folders", nargs="+")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "selection"), exist_ok=True)
    wb = Workbook(write_only=True)
    used, summary = set(), {}
    for folder in args.folders:
        t0 = time.time()
        rows, tag_cols, dcm_sel, jpg_sel = scan(folder)
        cols = FIXED_COLUMNS + tag_cols
        ws = wb.create_sheet(sheet_title(folder, used))
        ws.freeze_panes = "B2"
        ws.append(cols)
        for row in rows:
            ws.append([row.get(c, "") for c in cols])
        label = os.path.basename(os.path.normpath(folder))
        with open(os.path.join(args.out, "selection", f"{label}.dcm.txt"), "w") as f:
            f.write("".join(p + "\n" for p in dcm_sel))
        with open(os.path.join(args.out, "selection", f"{label}.jpg.txt"), "w") as f:
            f.write("".join(p + "\n" for p in jpg_sel))
        reasons = {}
        for r in rows:
            if not r.get("IsChest"):
                key = r["Reason"].split(" (")[0].split("=")[0]
                reasons[key] = reasons.get(key, 0) + 1
        summary[label] = {
            "dicoms": len(rows),
            "chest_to_anonymise": len(dcm_sel),
            "chest_jpgs": len(jpg_sel),
            "not_anonymised_by_reason": reasons,
            "tag_columns": len(tag_cols),
            "scan_seconds": round(time.time() - t0, 1),
        }
        print(label, json.dumps(summary[label]), flush=True)
    xlsx = os.path.join(args.out, "DICOM_tag_inventory.xlsx")
    wb.save(xlsx)
    with open(os.path.join(args.out, "selection", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("wrote", xlsx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
