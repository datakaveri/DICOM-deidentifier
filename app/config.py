"""
config.py — shared configuration, constants, and logging setup for the
DICOM anonymization pipeline.

Architecture:
  - Pixel Redaction: Character stroke segmentation + Navier-Stokes neighbor reconstruction
    for burned-in PHI text (preserves underlying tissue and clinical markers).
  - Tag De-Identification: Header tag tokenization, hashing, encryption, and suppression
    per tag_mapping.py with keystore persistence (secured.json).
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
    datefmt="%H:%M:%S",
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
TAG_AUDIT_SNAPSHOT_NAME = "tag_audit.json"
MANIFEST_NAME = "manifest.json"
BBOX_IMAGE_NAME = "bbox_regions.png"

# The bbox preview is the frame BEFORE redaction with the detected PHI regions
# drawn on top, so it shows the burned-in PHI in the clear. Useful when tuning
# detection locally; never written unless asked for.
SAVE_BBOX_PREVIEW = os.getenv("SKALD_SAVE_BBOX_PREVIEW", "0").lower() in (
    "1",
    "true",
    "yes",
)


# ── Batch-run options (all opt-in; the defaults match the enclave contract) ──
def _flag(name):
    return os.getenv(name, "0").strip().lower() in ("1", "true", "yes")


# "per-file" (default): OUTPUT_DIR/<name>/after_deidentification.dcm, with the
# audits beside it -- the layout the SPIDEr enclave reads.
# "mirror": OUTPUT_DIR/<relative path of the input>, same file name, and
# nothing else in OUTPUT_DIR; audits, previews and benchmark logs go to
# REPORT_DIR/<relative path without extension>/.
OUTPUT_LAYOUT = os.getenv("SKALD_OUTPUT_LAYOUT", "per-file").strip().lower()
if OUTPUT_LAYOUT not in ("per-file", "mirror"):
    raise SystemExit(
        f"SKALD_OUTPUT_LAYOUT must be per-file or mirror, not {OUTPUT_LAYOUT!r}"
    )
REPORT_DIR = os.getenv("SKALD_REPORT_DIR", OUTPUT_DIR)

# Tags (by keyword, comma-separated) the tag pass must leave exactly as they
# are, e.g. "StudyInstanceUID,SeriesInstanceUID,SOPInstanceUID" when the
# de-identified files must still line up with the source study/series/image.
PRESERVE_TAGS = tuple(
    k.strip() for k in os.getenv("SKALD_PRESERVE_TAGS", "").split(",") if k.strip()
)

# Write the still-identified intermediates (before_deidentification.dcm,
# data.json, phi_tags.json) to scratch under TEMP_DIR and delete each file's
# set as soon as its tag stage is done. Always on under a job config.
DISCARD_INTERMEDIATES = _flag("SKALD_DISCARD_INTERMEDIATES")

# Skip inputs whose de-identified output and audit already exist from an
# earlier run that succeeded, so an interrupted batch can be re-run. Pseudonyms
# stay consistent only if the same keys directory is reused (fixed policy).
RESUME = _flag("SKALD_RESUME")

# Save the keystore every N finished files, not only at the end, so a crash
# mid-batch does not lose the keys behind files already written.
KEYSTORE_SAVE_EVERY = int(os.getenv("SKALD_KEYSTORE_SAVE_EVERY", "25"))

BENCHMARK_FILES_NAME = "benchmark_files.jsonl"
BENCHMARK_SUMMARY_NAME = "benchmark_summary.json"
PREVIEW_NAME = "after_preview.png"

# =============================================================================
# SPIDEr JOB CONFIG (application "skald_dicom")
# =============================================================================
# The middleware decrypts the job config and drops it on the config mount.
# With a `dicom_deidentify` block, the user's per-tag choices drive tag
# de-identification (de_identification/job_config.py); without one, or with no
# file, the fixed policy in tag_mapping.py runs.
JOB_CONFIG_NAME = os.getenv("SKALD_JOB_CONFIG_NAME", "config.json")
JOB_CONFIG_FILE = os.getenv(
    "SKALD_JOB_CONFIG", os.path.join(CONFIG_DIR, JOB_CONFIG_NAME)
)

# Fixed policy. The config comes from a browser, so only its tag_actions vary
# between runs; every field below is read from here, never from the document,
# and job_config.py logs and discards any wire value that disagrees.

# Where the still-identified intermediates (data.json, phi_tags.json,
# before_deidentification.dcm) and the run's keystore go under a job config.
# Must stay under /tmp: OUTPUT_DIR is the volume that leaves the enclave.
TEMP_DIR = os.getenv("SKALD_TEMP_DIR", "/tmp/skald-dicom")

# True would publish the OCR-recognised burned-in text (names, MRNs) beside the
# output. Pinned false: published audits carry region geometry only.
EMIT_PIXEL_TEXT_REPORT = False

# No UID crosswalk exists. Pinned false so one is never published later.
EMIT_UID_CROSSWALK = False

# Under a job config, a file pydicom cannot parse fails instead of being read
# with force=True into a near-empty dataset that passes every stage.
FAIL_ON_UNPARSABLE = True

# Under a job config, burned-in text is always redacted with a solid black
# fill (not reversible, leaves no stroke residue) and verification re-checks
# after escalation, so a file can report FAILED.
PIXEL_POLICY = {
    "redact_burned_in_text": True,
    "method": "black",
    "verify": True,
}

# Set to True to enable DICOM header tag de-identification (hashing/masking/FPE).
# Set to False to keep all DICOM header tags 100% original and untouched (focusing solely on burned-in pixel text redaction).
ENABLE_TAG_DEIDENTIFICATION = True

# Single consolidated file holding every hash/tokenise/encrypt key used by
# de_identification/deidentify.py. One KeyStore is loaded from this file,
# shared across every DICOM file in the batch, and saved back once at the
# end -- so e.g. the same PatientID hashes/tokenises to the same value no
# matter which file in the batch it appears in.
#
# The file holds key material and the reverse token vault, so it must never
# land in a directory that is shipped as the release. SKALD_KEYS_DIR places it
# (the container sets /app/keys, a separate volume from /app/output); it
# defaults to the working directory, where earlier versions always wrote it.
KEYS_DIR = os.getenv("SKALD_KEYS_DIR", os.getcwd())
SECURED_KEYSTORE_FILE = os.path.join(KEYS_DIR, "secured.json")

# Identifier fields that function as an ID but whose keyword doesn't end in
# "ID"/"IDs" (so the generic suffix check below wouldn't catch them).
# AccessionNumber is deliberately NOT here — it gets its own "accession"
# category in identify_phi_tags (see phi_tags.py).
EXTRA_ID_FIELDS = set()

# Clinical terms that must NEVER be redacted (anatomy / positioning markers)
CLINICAL_ALLOWLIST = {
    "L",
    "R",
    "LT",
    "RT",
    "LEFT",
    "RIGHT",
    "PA",
    "AP",
    "LAT",
    "LL",
    "RL",
    "LATERAL",
    "ERECT",
    "SUPINE",
    "PRONE",
    "DECUBITUS",
    "UPRIGHT",
    "PORTABLE",
    "MOBILE",
    "STAT",
    "ROUTINE",
    "CHEST",
    "ABDOMEN",
    "PELVIS",
    "SKULL",
    "SPINE",
    "KVP",
    "MAS",
    "MA",
    "SEC",
    "CM",
    "MM",
    "FOV",
    "CXR",
    "CX",
    "PA VIEW",
    "AP VIEW",
}

# PII patterns (Indian health context + universal)
PII_PATTERNS = {
    "aadhaar": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "abha": r"\b\d{2}-\d{4}-\d{4}-\d{4}\b",
    "phone": r"\b(?:\+91|0)?[6-9]\d{9}\b",
    "date": r"\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b|\b\d{4}[-/.]\d{1,2}[-/.]\d{1,2}\b",
    "uhid_mrn": r"\b(?:UHID|MRN|REG|IPD|OPD|CR|PID|HID)[\s:/-]?\d+\b",
    "age_sex": r"\b\d{1,3}\s*[/]\s*[MFO]\b",
    "name_prefix": r"\b(?:DR\.?|MR\.?|MRS\.?|MS\.?|SHRI|SMT|S/O|D/O|W/O)\b",
    "accession": r"\b(?:ACC|ACCNO|ACCESSION)[\s:.-]?\w+\b",
}
