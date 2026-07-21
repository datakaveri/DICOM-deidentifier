"""
run.py — main container entrypoint. Processes every *.dcm file found under
DATA_DIR through the full de-identification pipeline and writes, per input
file (named after the file's own basename) under OUTPUT_DIR/<name>/:

  data.json                     — snapshot of every original tag value,
                                   taken before anything runs.
  phi_tags.json                 — PHI-bearing tags identified (read-only;
                                   no tag value is ever modified for this).
  before_deidentification.dcm   — the DICOM after the full burned-in
                                   pixel/OCR redaction pipeline. Tag VALUES
                                   are still as they were (only private tags
                                   stripped, per that pipeline's own policy).
  after_deidentification.dcm    — that same pixel-redacted DICOM with every
                                   tag in tag_mapping.py additionally
                                   transformed per its assigned technique.
  pipeline_audit.json           — pixel redaction audit (regions redacted,
                                   verification status).
  tag_audit.json                — tag de-identification audit (technique
                                   applied per tag).

Also writes OUTPUT_DIR/manifest.json summarizing the whole run.

Run with:
    python -m de_identification.run

The tokenise/encrypt key material is persisted separately under
OUTPUT_DIR/keystore/*.json so tokens/ciphertext stay consistent across runs
and across files.
"""

import glob
import json
import os

import pydicom

from config import DATA_DIR, OUTPUT_DIR, KEYSTORE_DIR, log
from phi_tags import dump_original_tags, identify_phi_tags
from engines import check_gpu_available, initialize_engines
from pipeline import anonymize_dicom_file
from de_identification.deidentify import deidentify_dataset
from de_identification.keystore import KeyStore


def _find_input_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, "**", "*.dcm"), recursive=True))


def process_file(input_path, paddle_ocr, easy_ocr, analyzer, keystore):
    name = os.path.splitext(os.path.basename(input_path))[0]
    out_dir = os.path.join(OUTPUT_DIR, name)
    os.makedirs(out_dir, exist_ok=True)

    data_snapshot = os.path.join(out_dir, "data.json")
    phi_tags_snapshot = os.path.join(out_dir, "phi_tags.json")
    before_dcm = os.path.join(out_dir, "before_deidentification.dcm")
    after_dcm = os.path.join(out_dir, "after_deidentification.dcm")
    pipeline_audit_path = os.path.join(out_dir, "pipeline_audit.json")
    tag_audit_path = os.path.join(out_dir, "tag_audit.json")

    ds = pydicom.dcmread(input_path, force=True)

    # 1. Original tag snapshot + read-only PHI tag identification, taken
    #    before anything runs.
    dump_original_tags(ds, data_snapshot)
    phi_tags = identify_phi_tags(ds)
    with open(phi_tags_snapshot, "w") as f:
        json.dump(phi_tags, f, indent=2)

    # 2. Full burned-in pixel/OCR redaction pipeline -> before_deidentification.dcm
    #    (tags untouched beyond that pipeline's own private-tag stripping)
    pipeline_audit = anonymize_dicom_file(input_path, before_dcm, paddle_ocr, easy_ocr, analyzer)
    with open(pipeline_audit_path, "w") as f:
        json.dump(pipeline_audit, f, indent=2)

    # 3. Tag-level de-identification applied on top of the pixel-redacted file
    ds_pixel_redacted = pydicom.dcmread(before_dcm)
    tag_audit = deidentify_dataset(ds_pixel_redacted, keystore)
    ds_pixel_redacted.save_as(after_dcm, write_like_original=False)
    with open(tag_audit_path, "w") as f:
        json.dump(tag_audit, f, indent=2)

    by_technique = {}
    for entry in tag_audit:
        by_technique[entry["technique"]] = by_technique.get(entry["technique"], 0) + 1

    log.info(f"{name}: pixel redaction {pipeline_audit['verification_status']} "
             f"({len(pipeline_audit['redacted_regions'])} region(s)); "
             f"{len(tag_audit)} tag(s) touched -> {out_dir}")

    return {
        "file": os.path.basename(input_path),
        "output_dir": out_dir,
        "pixel_verification_status": pipeline_audit["verification_status"],
        "redacted_regions": len(pipeline_audit["redacted_regions"]),
        "tags_touched": len(tag_audit),
        "tags_by_technique": by_technique,
        "error": pipeline_audit.get("error"),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    input_files = _find_input_files()
    if not input_files:
        log.warning(f"No .dcm files found under {DATA_DIR} — nothing to process.")
        with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
            json.dump({"files": []}, f, indent=2)
        return

    log.info(f"Found {len(input_files)} DICOM file(s) under {DATA_DIR}")

    use_gpu = check_gpu_available()
    paddle_ocr, easy_ocr, analyzer = initialize_engines(use_gpu=use_gpu)
    keystore = KeyStore(KEYSTORE_DIR)

    results = []
    for input_path in input_files:
        try:
            results.append(process_file(input_path, paddle_ocr, easy_ocr, analyzer, keystore))
        except Exception as e:
            log.error(f"[ERROR] {input_path}: {e}", exc_info=True)
            results.append({"file": os.path.basename(input_path), "error": str(e)})

    keystore.save()

    with open(os.path.join(OUTPUT_DIR, "manifest.json"), "w") as f:
        json.dump({"files": results}, f, indent=2)

    log.info(f"Done. {len(results)} file(s) processed. Manifest -> "
              f"{os.path.join(OUTPUT_DIR, 'manifest.json')}")


if __name__ == "__main__":
    main()
