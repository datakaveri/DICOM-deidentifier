"""
main.py — batch entry point. Processes every .dcm file found directly under
INPUT_DIR: for each, snapshots/identifies tags (read-only, Step 1), then
runs the burned-in pixel text anonymization pipeline (Step 2 + Step 3) and
reports the result.

Pipeline order (see pipeline.py): pixels are redacted first and checkpointed
to a temp before_deidentification.dcm (tags still original at that point),
then tag de-identification (de_identification/deidentify.py — hash
PatientID, mask dates, suppress direct identifiers, etc. per
tag_mapping.py) runs last.

All intermediate artifacts (the pixel-only checkpoint, the bbox
visualization, phi_tags.json, the raw pipeline audit) live in a per-file
temp directory and are discarded once processing finishes. OUTPUT_DIR ends
up with exactly two files per input: FINAL_OUTPUT_NAME (the de-identified
DICOM) and STATUS_SNAPSHOT_NAME (input/output locations, the tag-level
operations applied, and the original-tag data.json snapshot).

One KeyStore is loaded from SECURED_KEYSTORE_FILE ("secured.json") and
shared across every file in the batch, then saved once at the end — so the
same PatientID/AccessionNumber/etc. hashes or tokenises to the same value no
matter which file in the batch it appears in.
"""

import os
import glob
import json
import tempfile

import pydicom

from config import (
    INPUT_DIR, OUTPUT_DIR, BEFORE_OUTPUT_NAME, FINAL_OUTPUT_NAME,
    DATA_SNAPSHOT_NAME, PHI_TAGS_SNAPSHOT_NAME, STATUS_SNAPSHOT_NAME,
    SECURED_KEYSTORE_FILE,
)
from phi_tags import dump_original_tags, identify_phi_tags
from engines import check_gpu_available, initialize_engines
from pipeline import anonymize_dicom_file
from de_identification.keystore import KeyStore


def process_file(input_path, paddle_ocr, easy_ocr, analyzer, keystore):
    """Runs the full pipeline for one DICOM file; returns its pipeline audit dict."""
    stem = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, stem)
    os.makedirs(out_dir, exist_ok=True)
    final_output = os.path.join(out_dir, FINAL_OUTPUT_NAME)
    status_path = os.path.join(out_dir, STATUS_SNAPSHOT_NAME)

    print(f"\n{'#'*60}\n# {os.path.basename(input_path)}\n{'#'*60}")

    with tempfile.TemporaryDirectory(prefix=f"{stem}_") as work_dir:
        before_output = os.path.join(work_dir, BEFORE_OUTPUT_NAME)
        data_snapshot = os.path.join(work_dir, DATA_SNAPSHOT_NAME)
        phi_tags_snapshot = os.path.join(work_dir, PHI_TAGS_SNAPSHOT_NAME)

        ds = pydicom.dcmread(input_path)

        # Snapshot every original tag value (working copy; ends up embedded
        # in status.json below, not kept as its own file in OUTPUT_DIR)
        dump_original_tags(ds, data_snapshot)

        # ── Step 1: identify which tags carry PHI (read-only, ds untouched) ────
        phi_tags = identify_phi_tags(ds)
        with open(phi_tags_snapshot, "w") as f:
            json.dump(phi_tags, f, indent=2)

        print(f"Identified {len(phi_tags)} PHI-bearing tag(s) (left unmodified).\n")
        for t in phi_tags:
            print(f"{t['tag']:>14}  {t['field']:<28} {t['category']:<14} {t['value'][:40]!r}")

        # ── Step 2 (+ Step 3): burned-in pixel redaction, then tag de-identification ──
        # PHI baked into the pixels is redacted first (masking.py), using the
        # original tag values above (Step 3) to help find matching burned-in
        # text on the image; the pixel-redacted result is checkpointed to a
        # temp file, then tag de-identification (same keystore for every file
        # in the batch) runs last, producing the final output.dcm.
        print(f"\nRunning de-identification pipeline (pixels, then tags) on {os.path.basename(input_path)}...")
        pipeline_audit = anonymize_dicom_file(
            input_path, before_output, final_output, data_snapshot,
            paddle_ocr, easy_ocr, analyzer, keystore,
        )

        with open(data_snapshot) as f:
            original_tags = json.load(f)

        status = {
            "input_file": os.path.basename(input_path),
            "input_path": input_path,
            "output_file": FINAL_OUTPUT_NAME,
            "output_path": final_output,
            "verification_status": pipeline_audit["verification_status"],
            "tag_operations": pipeline_audit["deidentified_tags"],
            "data": original_tags,
        }
        with open(status_path, "w") as f:
            json.dump(status, f, indent=2)

    print(f"\nPixel anonymization status: {pipeline_audit['verification_status']}")
    print(f"Tags de-identified: {len(pipeline_audit['deidentified_tags'])}")
    print(f"Final anonymized DICOM saved -> {final_output}")
    print(f"Status saved -> {status_path}")

    return pipeline_audit


def main():
    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.dcm")))
    if not input_files:
        print(f"No .dcm files found in {INPUT_DIR}/")
        return

    print(f"Found {len(input_files)} file(s) in {INPUT_DIR}/ to process.")

    use_gpu = check_gpu_available()
    paddle_ocr, easy_ocr, analyzer = initialize_engines(use_gpu=use_gpu)

    # One KeyStore shared across the whole batch, saved once at the end, so
    # every file's hash/tokenise/encrypt values stay consistent with each other.
    keystore = KeyStore(SECURED_KEYSTORE_FILE)

    results = []
    for input_path in input_files:
        audit = process_file(input_path, paddle_ocr, easy_ocr, analyzer, keystore)
        results.append(audit)

    keystore.save()
    print(f"\nShared key/token material for this batch saved -> {SECURED_KEYSTORE_FILE}")

    print(f"\n{'='*60}\nBatch complete: {len(results)} file(s) processed.")
    for audit in results:
        print(f"  {audit['file']:<30} {audit['verification_status']}")


if __name__ == "__main__":
    main()
