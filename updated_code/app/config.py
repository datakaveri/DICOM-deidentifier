"""
config.py — shared configuration, constants, and logging setup for the
DICOM anonymization pipeline.

Policy: DICOM tag VALUES are left exactly as-is everywhere in this pipeline
(only UIDs are regenerated and private/vendor tags are stripped, so the file
can't be linked back to the original study). No tag is hashed, removed, or
generalized. PHI is removed from the pixel data instead (see masking.py).
"""

import os
import logging

# ─── Suppress verbose sub-library logs ───────────────────────────────────────
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["DISABLE_AUTO_LOGGING_CONFIG"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("dicom_anonymizer")

# =============================================================================
# CONFIGURATION
# =============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("SKALD_DATA_DIR", os.path.join(BASE_DIR, "data"))
CONFIG_DIR = os.getenv("SKALD_CONFIG_DIR", os.path.join(BASE_DIR, "config"))
OUTPUT_DIR = os.getenv("SKALD_OUTPUT_DIR", os.path.join(BASE_DIR, "output"))
KEYSTORE_DIR = os.path.join(OUTPUT_DIR, "keystore")

INPUT_DIR = DATA_DIR

BEFORE_OUTPUT_NAME = "before_deidentification.dcm"
FINAL_OUTPUT_NAME = "after_deidentification.dcm"
DATA_SNAPSHOT_NAME = "data.json"
PHI_TAGS_SNAPSHOT_NAME = "phi_tags.json"
PIPELINE_AUDIT_SNAPSHOT_NAME = "pipeline_audit.json"

# Set to True to enable DICOM header tag de-identification (hashing/masking/FPE).
# Set to False to keep all DICOM header tags 100% original and untouched (focusing solely on burned-in pixel text redaction).
ENABLE_TAG_DEIDENTIFICATION = False

# Single consolidated file holding every hash/tokenise/encrypt key used by
# de_identification/deidentify.py. One KeyStore is loaded from this file,
# shared across every DICOM file in the batch, and saved back once at the
# end -- so e.g. the same PatientID hashes/tokenises to the same value no
# matter which file in the batch it appears in.
SECURED_KEYSTORE_FILE = "secured.json"

# Identifier fields that function as an ID but whose keyword doesn't end in
# "ID"/"IDs" (so the generic suffix check below wouldn't catch them).
# AccessionNumber is deliberately NOT here — it gets its own "accession"
# category in identify_phi_tags (see phi_tags.py).
EXTRA_ID_FIELDS = set()

# Clinical terms that must NEVER be redacted (anatomy / positioning markers)
CLINICAL_ALLOWLIST = {
    "L", "R", "LT", "RT", "LEFT", "RIGHT",
    "PA", "AP", "LAT", "LL", "RL", "LATERAL",
    "ERECT", "SUPINE", "PRONE", "DECUBITUS", "UPRIGHT",
    "PORTABLE", "MOBILE", "STAT", "ROUTINE",
    "CHEST", "ABDOMEN", "PELVIS", "SKULL", "SPINE",
    "KVP", "MAS", "MA", "SEC", "CM", "MM", "FOV",
    "CXR", "CX", "PA VIEW", "AP VIEW",
}

# PII patterns (Indian health context + universal)
PII_PATTERNS = {
    "aadhaar":     r'\b\d{4}\s?\d{4}\s?\d{4}\b',
    "abha":        r'\b\d{2}-\d{4}-\d{4}-\d{4}\b',
    "phone":       r'\b(?:\+91|0)?[6-9]\d{9}\b',
    "date":        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b',
    "uhid_mrn":    r'\b(?:UHID|MRN|REG|IPD|OPD|CR|PID|HID)[\s:/-]?\d+\b',
    "age_sex":     r'\b\d{1,3}\s*[/]\s*[MFO]\b',
    "name_prefix": r'\b(?:DR\.?|MR\.?|MRS\.?|MS\.?|SHRI|SMT|S/O|D/O|W/O)\b',
    "accession":   r'\b(?:ACC|ACCNO|ACCESSION)[\s:.-]?\w+\b',
}
