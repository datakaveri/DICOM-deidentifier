import pydicom
import json
import pandas as pd
import re

from presidio_analyzer import AnalyzerEngine

INPUT_DCM = "input/dicom/synthetic_phi.dcm"

# -----------------------------
# Load DICOM
# -----------------------------
ds = pydicom.dcmread(INPUT_DCM)

# -----------------------------
# Presidio
# -----------------------------
analyzer = AnalyzerEngine()

records = []

# -----------------------------
# Helper patterns
# -----------------------------
date_pat = r"\b\d{8}\b|\b\d{4}-\d{2}-\d{2}\b"
mrn_pat = r"\b(MRN|PAT|ID|ACC)[A-Z0-9_-]*\b"

# -----------------------------
# Scan all tags
# -----------------------------
for elem in ds.iterall():

    if elem.VR == "SQ":
        continue

    tag = str(elem.tag)
    field = elem.keyword if elem.keyword else "Unknown"
    value = str(elem.value).strip()

    if value == "":
        continue

    score = 0
    flags = []

    # Private tag
    if elem.tag.is_private:
        score += 15
        flags.append("private_tag")

    # Sensitive keyword
    low = field.lower()
    for k in ["patient","name","birth","phone","address","physician","institution"]:
        if k in low:
            score += 35
            flags.append("sensitive_keyword")

    # Regex checks
    if re.search(date_pat, value):
        score += 25
        flags.append("date_pattern")

    if re.search(mrn_pat, value.upper()):
        score += 35
        flags.append("mrn_pattern")

    # Presidio NLP
    try:
        findings = analyzer.analyze(
            text=value,
            language="en"
        )

        for f in findings:
            score += int(f.score * 40)
            flags.append(f.entity_type)

    except:
        pass

    if score > 0:
        confidence = min(score, 100)

        records.append({
            "tag": tag,
            "field": field,
            "value": value[:150],
            "confidence": confidence,
            "flags": list(set(flags))
        })

# -----------------------------
# Save JSON
# -----------------------------
import os; os.makedirs("output/reports", exist_ok=True)

with open("output/reports/pii_report.json", "w") as f:
    json.dump({"pii_candidates": records}, f, indent=2)

# -----------------------------
# Save CSV
# -----------------------------
pd.DataFrame(records).to_csv("output/reports/pii_report.csv", index=False)

print("Created output/reports/pii_report.json")
print("Created output/reports/pii_report.csv")
print("Candidates found:", len(records))
