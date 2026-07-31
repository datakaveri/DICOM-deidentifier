"""
run.py — standalone entry point, kept for `python -m de_identification.run`.

main.py's pipeline now implements exactly this flow directly (pixel/OCR
redaction -> before_deidentification.dcm checkpoint, tag-level
de-identification via deidentify_dataset() -> after_deidentification.dcm),
so this just delegates to it instead of duplicating the orchestration.

Run from the `updated_code` directory with:
    python -m de_identification.run
"""

from main import main

if __name__ == "__main__":
    main()
