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
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # app/
PROJECT_ROOT = os.path.dirname(BASE_DIR)                       # repo root

# Local-dev defaults mirror the container's mounted volumes: input and output
# live at the project root (see .gitignore), config stays beside the code.
# In the image all three are set explicitly as env vars to /app/{data,config,output}.
DATA_DIR = os.getenv("SKALD_DATA_DIR", os.path.join(PROJECT_ROOT, "data"))
CONFIG_DIR = os.getenv("SKALD_CONFIG_DIR", os.path.join(BASE_DIR, "config"))
OUTPUT_DIR = os.getenv("SKALD_OUTPUT_DIR", os.path.join(PROJECT_ROOT, "output"))
KEYSTORE_DIR = os.path.join(OUTPUT_DIR, "keystore")

INPUT_DIR = DATA_DIR

# =============================================================================
# ENFORCED JOB-CONFIG FIELDS (SPIDEr "skald_dicom" contract, section 5)
# =============================================================================
# The job config arrives from a browser (Fernet-encrypted as config_ciphertext,
# decrypted to JSON by the middleware). Only its `tag_actions` block varies
# between runs; every field below is fixed policy and is read from HERE, never
# from the wire, so a tampered config cannot move it even if a future UI bug
# lets one through. job_config.py logs -- and discards -- any wire value that
# disagrees with these.

# Where the still-identified intermediates go. MUST stay under /tmp: anything
# beneath OUTPUT_DIR would write the original, un-de-identified DICOM onto the
# volume that leaves the enclave.
TEMP_DIR = os.getenv("SKALD_TEMP_DIR", "/tmp/skald-dicom")

# When true the OCR-recognised burned-in text (patient names, MRNs) would be
# written to a plaintext sidecar on the output volume -- the PHI would survive
# in readable form even though the pixels were blacked out. Pinned false: the
# pipeline audit copied to OUTPUT_DIR has its region text stripped.
EMIT_PIXEL_TEXT_REPORT = False

# Nothing here remaps UIDs through a crosswalk, so there is none to emit today.
# Pinned false anyway: an enclave that grows the capability later must not
# start publishing a re-identification key beside the output.
EMIT_UID_CROSSWALK = False

# A DICOM that cannot be parsed must fail loudly, never pass through
# un-de-identified on the grounds that nothing could be found to remove.
FAIL_ON_UNPARSABLE = True

# Burned-in text is always detected, always blacked out, and the result is
# always verified. These were briefly UI switches and were removed: turning
# redaction off leaves PHI printed into pixels no tag action can reach, and
# `blur` is partially reversible on text -- the one thing this pass exists to
# prevent.
PIXEL_POLICY = {
    "redact_burned_in_text": True,
    "method": "black",
    "verify": True,
}

# Filename of the decrypted job config on the CONFIG_DIR mount. Absent ->
# the pre-contract fixed-policy path runs unchanged (contract section 7).
JOB_CONFIG_NAME = os.getenv("SKALD_JOB_CONFIG_NAME", "config.json")
JOB_CONFIG_FILE = os.getenv("SKALD_JOB_CONFIG", os.path.join(CONFIG_DIR, JOB_CONFIG_NAME))

BEFORE_OUTPUT_NAME = "before_deidentification.dcm"
FINAL_OUTPUT_NAME = "after_deidentification.dcm"
DATA_SNAPSHOT_NAME = "data.json"
PHI_TAGS_SNAPSHOT_NAME = "phi_tags.json"
PIPELINE_AUDIT_SNAPSHOT_NAME = "pipeline_audit.json"
TAG_AUDIT_SNAPSHOT_NAME = "tag_audit.json"
MANIFEST_NAME = "manifest.json"
BBOX_IMAGE_NAME = "bbox_regions.png"

# The bbox preview is a PNG of the frame BEFORE redaction, with the detected
# PHI regions drawn on top — i.e. it still shows the burned-in PHI in the
# clear. Useful when tuning detection locally; never written by default, so
# a container run can't leak PHI into the mounted output volume.
SAVE_BBOX_PREVIEW = os.getenv("SKALD_SAVE_BBOX_PREVIEW", "0").lower() in ("1", "true", "yes")

# Set to True to enable DICOM header tag de-identification (hashing/masking/FPE).
# Set to False to keep all DICOM header tags 100% original and untouched (focusing solely on burned-in pixel text redaction).
ENABLE_TAG_DEIDENTIFICATION = True

# Single consolidated file holding every hash/tokenise/encrypt key used by
# de_identification/deidentify.py. One KeyStore is loaded from this file,
# shared across every DICOM file in the batch, and saved back once at the
# end -- so e.g. the same PatientID hashes/tokenises to the same value no
# matter which file in the batch it appears in.
#
# It lives under OUTPUT_DIR (the mounted output volume in the container), not
# beside the code: key material written into the container's own filesystem
# would be discarded with the container, re-minting every key on the next run
# and breaking correlation between batches.
SECURED_KEYSTORE_FILE = os.path.join(KEYSTORE_DIR, "secured.json")

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
    "date":        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b',
    "uhid_mrn":    r'\b(?:UHID|MRN|REG|IPD|OPD|CR|PID|HID)[\s:/-]?\d+\b',
    "age_sex":     r'\b\d{1,3}\s*[/]\s*[MFO]\b',
    "name_prefix": r'\b(?:DR\.?|MR\.?|MRS\.?|MS\.?|SHRI|SMT|S/O|D/O|W/O)\b',
    "accession":   r'\b(?:ACC|ACCNO|ACCESSION)[\s:.-]?\w+\b',
}
