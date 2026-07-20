"""
run.py — standalone entry point that chains the two anonymization passes
and writes exactly three files to output/:

  1. data.json                     — snapshot of every original tag value,
                                      taken before anything runs.
  2. before_deidentification.dcm   — the DICOM after the full burned-in
                                      pixel/OCR redaction pipeline
                                      (pipeline.anonymize_dicom_file), with
                                      tag VALUES still as they were (only
                                      private tags stripped, per that
                                      pipeline's own policy). "Before" here
                                      means before TAG de-identification —
                                      the burned-in PHI in the image is
                                      already blocked out at this point.
  3. after_deidentification.dcm    — that same pixel-redacted DICOM with
                                      every tag in tag_mapping.py additionally
                                      transformed per its assigned technique.

Run from the `updated_code` directory with:
    python -m de_identification.run

Requires the pixel pipeline's dependencies (opencv-python, and at least one
of easyocr/paddleocr) to be installed, since step 2 runs OCR detection over
the image, same as main.py.

Every run overwrites these three files in place (no accumulation). The
tokenise/encrypt key material is persisted separately under
de_identification/keystore/*.json so tokens/ciphertext stay consistent
across runs and across files — that isn't one of the 3 requested output
files, so it lives next to this script rather than inside output/.
"""

import os

import pydicom

from config import INPUT_DCM
from phi_tags import dump_original_tags
from engines import check_gpu_available, initialize_engines
from pipeline import anonymize_dicom_file
from de_identification.deidentify import deidentify_dataset
from de_identification.keystore import KeyStore

DATA_SNAPSHOT = "output/data.json"
BEFORE_DCM = "output/before_deidentification.dcm"
AFTER_DCM = "output/after_deidentification.dcm"
KEYSTORE_DIR = "de_identification/keystore"


def main():
    os.makedirs("output", exist_ok=True)

    ds = pydicom.dcmread(INPUT_DCM)

    # 1. Original tag snapshot, taken before anything runs
    dump_original_tags(ds, DATA_SNAPSHOT)

    # 2. Full burned-in pixel/OCR redaction pipeline -> before_deidentification.dcm
    #    (tags untouched beyond that pipeline's own private-tag stripping)
    use_gpu = check_gpu_available()
    paddle_ocr, easy_ocr, analyzer = initialize_engines(use_gpu=use_gpu)
    pixel_audit = anonymize_dicom_file(INPUT_DCM, BEFORE_DCM, paddle_ocr, easy_ocr, analyzer)

    # 3. Tag-level de-identification applied on top of the pixel-redacted file
    ds_pixel_redacted = pydicom.dcmread(BEFORE_DCM)
    keystore = KeyStore(KEYSTORE_DIR)
    tag_audit = deidentify_dataset(ds_pixel_redacted, keystore)
    keystore.save()
    ds_pixel_redacted.save_as(AFTER_DCM, write_like_original=False)

    by_technique = {}
    for entry in tag_audit:
        by_technique[entry["technique"]] = by_technique.get(entry["technique"], 0) + 1

    print(f"Original tag snapshot saved -> {DATA_SNAPSHOT}")
    print(f"Pixel redaction status: {pixel_audit['verification_status']} "
          f"({len(pixel_audit['redacted_regions'])} region(s) redacted)")
    print(f"Before tag de-identification (pixel-redacted) DICOM saved -> {BEFORE_DCM}")
    print(f"After tag de-identification DICOM saved -> {AFTER_DCM}")
    print(f"{len(tag_audit)} tag(s) touched:")
    for technique, count in sorted(by_technique.items()):
        print(f"  {technique:<12} {count}")


if __name__ == "__main__":
    main()
