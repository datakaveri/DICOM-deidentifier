"""
main.py — entry point. Reads the raw DICOM once, snapshots/identifies tags
(read-only, Step 1), then runs the burned-in pixel text anonymization
pipeline (Step 2 + Step 3) and reports the result.

Pipeline order (see pipeline.py): pixels are redacted first and checkpointed
to before_deidentification.dcm (tags still original at that point), then tag
de-identification (de_identification/deidentify.py — hash PatientID, mask
dates, suppress direct identifiers, etc. per tag_mapping.py) runs last, and
that final result is saved to after_deidentification.dcm.
"""

import os
import json

import pydicom

from config import (
    INPUT_DCM, FINAL_OUTPUT_DCM, DATA_SNAPSHOT,
    PHI_TAGS_SNAPSHOT, PIPELINE_AUDIT_SNAPSHOT,
)
from phi_tags import dump_original_tags, identify_phi_tags
from engines import check_gpu_available, initialize_engines
from pipeline import anonymize_dicom_file
from de_identification.keystore import KeyStore

KEYSTORE_DIR = "de_identification/keystore"


def main():
    ds = pydicom.dcmread(INPUT_DCM)

    # Snapshot every original tag value, for reference (no tag is ever modified)
    dump_original_tags(ds, DATA_SNAPSHOT)
    print(f"Original tags saved -> {DATA_SNAPSHOT}\n")

    # ── Step 1: identify which tags carry PHI (read-only, ds untouched) ────
    phi_tags = identify_phi_tags(ds)
    with open(PHI_TAGS_SNAPSHOT, "w") as f:
        json.dump(phi_tags, f, indent=2)

    print(f"Identified {len(phi_tags)} PHI-bearing tag(s) (left unmodified). "
          f"Saved -> {PHI_TAGS_SNAPSHOT}\n")
    for t in phi_tags:
        print(f"{t['tag']:>14}  {t['field']:<28} {t['category']:<14} {t['value'][:40]!r}")

    # ── Step 2 (+ Step 3): burned-in pixel redaction, then tag de-identification ──
    # Runs directly on the raw input: PHI baked into the pixels is redacted
    # first (masking.py), using the original tag values above (Step 3) to
    # help find matching burned-in text on the image; the pixel-redacted
    # result is checkpointed to before_deidentification.dcm, then tag
    # de-identification (metadata.py) runs last, producing after_deidentification.dcm.
    print("\nRunning de-identification pipeline (pixels, then tags)...")
    use_gpu = check_gpu_available()
    paddle_ocr, easy_ocr, analyzer = initialize_engines(use_gpu=use_gpu)
    keystore = KeyStore(KEYSTORE_DIR)

    os.makedirs(os.path.dirname(FINAL_OUTPUT_DCM), exist_ok=True)
    pipeline_audit = anonymize_dicom_file(
        INPUT_DCM, FINAL_OUTPUT_DCM, paddle_ocr, easy_ocr, analyzer, keystore
    )
    keystore.save()

    # Full result of the complete flow (overwrites on every run, never appended)
    with open(PIPELINE_AUDIT_SNAPSHOT, "w") as f:
        json.dump(pipeline_audit, f, indent=2)

    print(f"\nPixel anonymization status: {pipeline_audit['verification_status']}")
    print(f"Redacted regions: {len(pipeline_audit['redacted_regions'])}")
    print(f"Tags de-identified: {len(pipeline_audit['deidentified_tags'])}")
    print(f"Final anonymized DICOM saved -> {FINAL_OUTPUT_DCM}")
    print(f"Full pipeline audit saved -> {PIPELINE_AUDIT_SNAPSHOT}")


if __name__ == "__main__":
    main()
