"""
run.py — standalone entry point, kept for `python -m de_identification.run`.

main.py's pipeline now implements exactly this flow directly (pixel/OCR
redaction -> before_deidentification.dcm checkpoint, tag-level
de-identification via deidentify_dataset() -> after_deidentification.dcm),
so this just delegates to it instead of duplicating the orchestration.

This is the container's entrypoint (see the Dockerfile's CMD). Run it from
inside `app/` with:
    python -m de_identification.run

Input/output contract, unchanged from the image's first release: every *.dcm
under DATA_DIR is processed recursively, and results land in
OUTPUT_DIR/<name>/ (data.json, phi_tags.json, before_deidentification.dcm,
after_deidentification.dcm, pipeline_audit.json, tag_audit.json) alongside a
run-level OUTPUT_DIR/manifest.json. See main.py for the full description.
"""

from main import main

if __name__ == "__main__":
    main()
