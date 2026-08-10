"""
main.py — batch entry point. Processes every .dcm file found directly under
INPUT_DIR: for each, snapshots/identifies tags (read-only, Step 1), then
runs the burned-in pixel text anonymization pipeline (Step 2 + Step 3) and
reports the result.

Pipeline order (see pipeline.py): pixels are redacted first and checkpointed
to before_deidentification.dcm (tags still original at that point), then tag
de-identification (de_identification/deidentify.py — hash PatientID, mask
dates, suppress direct identifiers, etc. per tag_mapping.py) runs last, and
that final result is saved to after_deidentification.dcm.

One KeyStore is loaded from SECURED_KEYSTORE_FILE ("secured.json") and
shared across every file in the batch, then saved once at the end — so the
same PatientID/AccessionNumber/etc. hashes or tokenises to the same value no
matter which file in the batch it appears in.
"""

import os
import glob
import json

import pydicom

from config import (
    INPUT_DIR, OUTPUT_DIR, BEFORE_OUTPUT_NAME, FINAL_OUTPUT_NAME,
    DATA_SNAPSHOT_NAME, PHI_TAGS_SNAPSHOT_NAME, PIPELINE_AUDIT_SNAPSHOT_NAME,
    SECURED_KEYSTORE_FILE,
)
from phi_tags import dump_original_tags, identify_phi_tags
from engines import check_gpu_available, initialize_engines
from pipeline import anonymize_dicom_file
from de_identification.keystore import KeyStore


import time

def process_file(input_path, paddle_ocr, easy_ocr, analyzer, keystore,
                 deid_model=None, medical_ner=None, gliner_model=None):
    """Runs the full pipeline for one DICOM file; returns its pipeline audit dict."""
    start_time = time.time()
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)

    before_output = os.path.join(out_dir, BEFORE_OUTPUT_NAME)
    final_output = os.path.join(out_dir, FINAL_OUTPUT_NAME)
    data_snapshot = os.path.join(out_dir, DATA_SNAPSHOT_NAME)
    phi_tags_snapshot = os.path.join(out_dir, PHI_TAGS_SNAPSHOT_NAME)
    pipeline_audit_snapshot = os.path.join(out_dir, PIPELINE_AUDIT_SNAPSHOT_NAME)

    print(f"\n{'#'*60}\n# {os.path.basename(input_path)}\n{'#'*60}")

    ds = pydicom.dcmread(input_path)

    # Snapshot every original tag value, for reference (no tag is ever modified)
    dump_original_tags(ds, data_snapshot)
    print(f"Original tags saved -> {data_snapshot}\n")

    # ── Step 1: identify which tags carry PHI (read-only, ds untouched) ────
    phi_tags = identify_phi_tags(ds)
    with open(phi_tags_snapshot, "w") as f:
        json.dump(phi_tags, f, indent=2)

    print(f"Identified {len(phi_tags)} PHI-bearing tag(s) (left unmodified). "
          f"Saved -> {phi_tags_snapshot}\n")
    for t in phi_tags:
        print(f"{t['tag']:>14}  {t['field']:<28} {t['category']:<14} {t['value'][:40]!r}")

    # ── Step 2 (+ Step 3): burned-in pixel redaction, then tag de-identification ──
    print(f"\nRunning de-identification pipeline (pixels, then tags) on {os.path.basename(input_path)}...")
    pipeline_audit = anonymize_dicom_file(
        input_path, before_output, final_output, data_snapshot,
        paddle_ocr, easy_ocr, analyzer, keystore,
        deid_model=deid_model, medical_ner=medical_ner, gliner_model=gliner_model
    )

    elapsed_sec = round(time.time() - start_time, 2)
    pipeline_audit["execution_time_seconds"] = elapsed_sec

    with open(pipeline_audit_snapshot, "w") as f:
        json.dump(pipeline_audit, f, indent=2)

    print(f"\nPixel anonymization status: {pipeline_audit['verification_status']}")
    print(f"Redacted regions: {len(pipeline_audit['redacted_regions'])}")
    print(f"Tags de-identified: {len(pipeline_audit['deidentified_tags'])}")
    print(f"Pipeline execution time: {elapsed_sec:.2f} seconds")
    print(f"Final anonymized DICOM saved -> {final_output}")
    if pipeline_audit.get("bbox_image_path"):
        print(f"Bounding-box visualization saved -> {pipeline_audit['bbox_image_path']}")
    print(f"Full pipeline audit saved -> {pipeline_audit_snapshot}")

    return pipeline_audit


def main():
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.dcm")))
    if not input_files:
        print(f"No .dcm files found in {INPUT_DIR}/")
        return

    print(f"Found {len(input_files)} file(s) in {INPUT_DIR}/ to process.")

    use_gpu = check_gpu_available()
    paddle_ocr, easy_ocr, analyzer, deid_model, medical_ner, gliner_model = initialize_engines(use_gpu=use_gpu)

    # One KeyStore shared across the whole batch, saved once at the end, so
    # every file's hash/tokenise/encrypt values stay consistent with each other.
    keystore = KeyStore(SECURED_KEYSTORE_FILE)

    results = []
    for input_path in input_files:
        audit = process_file(input_path, paddle_ocr, easy_ocr, analyzer, keystore,
                             deid_model=deid_model, medical_ner=medical_ner, gliner_model=gliner_model)
        results.append(audit)

    keystore.save()
    print(f"\nShared key/token material for this batch saved -> {SECURED_KEYSTORE_FILE}")

    print(f"\n{'='*60}\nBatch complete: {len(results)} file(s) processed.")
    for audit in results:
        print(f"  {audit['file']:<30} {audit['verification_status']}")


if __name__ == "__main__":
    main()
