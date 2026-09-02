"""
main.py — batch entry point, and the container's entrypoint by way of
de_identification/run.py. Processes every *.dcm file found under INPUT_DIR
(recursively): for each, snapshots/identifies tags (read-only, Step 1), then
runs the burned-in pixel text anonymization pipeline (Step 2 + Step 3) and
reports the result.

Pipeline order (see pipeline.py): pixels are redacted first and checkpointed
to before_deidentification.dcm (tags still original at that point), then tag
de-identification (de_identification/deidentify.py — hash PatientID, mask
dates, suppress direct identifiers, etc. per tag_mapping.py) runs last, and
that final result is saved to after_deidentification.dcm.

Which tag policy runs is decided by the decrypted SPIDEr job config on the
config mount (de_identification/job_config.py):

  no `dicom_deidentify` block   — the original fixed policy, driven by
                                   tag_mapping.py. Unchanged, and what a
                                   pre-contract config still gets.
  a `dicom_deidentify` block    — the user's own per-tag choices, driven by
                                   its `tag_actions`. A tag it does not name
                                   is written through unchanged.

Writes, per input file, under OUTPUT_DIR/<name>/:

  after_deidentification.dcm    — the pixel-redacted DICOM with the tag
                                   policy applied.
  pipeline_audit.json           — pixel redaction audit (region geometry,
                                   verification status, timing). The
                                   recognised burned-in TEXT is stripped from
                                   it: emit_pixel_text_report is pinned false.
  tag_audit.json                — tag de-identification audit (technique
                                   applied per tag; no values, so it carries
                                   no PHI).

Three artefacts still hold the original, un-de-identified values:

  data.json                     — snapshot of every original tag value.
  phi_tags.json                 — the PHI-bearing tags identified, values
                                   included.
  before_deidentification.dcm   — pixels redacted, tag VALUES still original.

Under a job config those go to a scratch tree under TEMP_DIR (/tmp), never
OUTPUT_DIR, and are deleted when the batch ends — OUTPUT_DIR is the volume
that leaves the enclave. Without one they stay beside the output as before.

Also writes OUTPUT_DIR/manifest.json summarizing the whole run.

One KeyStore is shared across every file in the batch and saved once at the
end, so the same PatientID/AccessionNumber/etc. encrypts or tokenises to the
same value no matter which file it appears in. Under a job config it lives in
the scratch tree and dies with the run; otherwise it stays at
SECURED_KEYSTORE_FILE (OUTPUT_DIR/keystore/secured.json) and keeps correlating
across runs. The hashing_with_salt salt is per-run in both cases and is never
written anywhere.
"""

import os
import glob
import json
import shutil

import pydicom

from config import (
    INPUT_DIR, OUTPUT_DIR, TEMP_DIR, BEFORE_OUTPUT_NAME, FINAL_OUTPUT_NAME,
    DATA_SNAPSHOT_NAME, PHI_TAGS_SNAPSHOT_NAME, PIPELINE_AUDIT_SNAPSHOT_NAME,
    TAG_AUDIT_SNAPSHOT_NAME, MANIFEST_NAME, SECURED_KEYSTORE_FILE,
    EMIT_PIXEL_TEXT_REPORT, log,
)
from phi_tags import dump_original_tags, identify_phi_tags
from engines import check_gpu_available, initialize_engines
from pipeline import anonymize_dicom_file
from de_identification.keystore import KeyStore
from de_identification.job_config import load_job_config
from de_identification.actions import RunSecrets


import time

def process_file(input_path, out_dir, paddle_ocr, easy_ocr, analyzer, keystore,
                 deid_model=None, medical_ner=None, gliner_model=None,
                 job_config=None, secrets=None, scratch_dir=None):
    """Runs the full pipeline for one DICOM file; returns its pipeline audit dict.

    `scratch_dir`, when given, receives the three artefacts that still carry
    the original, un-de-identified values — data.json (every original tag
    value), phi_tags.json (the identified PHI, values included) and
    before_deidentification.dcm (pixels redacted, tags untouched). Under the
    job-config path that is TEMP_DIR, because any path beneath OUTPUT_DIR
    writes still-identified data onto the volume that leaves the enclave.
    Without one they stay beside the output, as they always have.
    """
    start_time = time.time()
    os.makedirs(out_dir, exist_ok=True)
    scratch_dir = scratch_dir or out_dir
    os.makedirs(scratch_dir, exist_ok=True)

    before_output = os.path.join(scratch_dir, BEFORE_OUTPUT_NAME)
    data_snapshot = os.path.join(scratch_dir, DATA_SNAPSHOT_NAME)
    phi_tags_snapshot = os.path.join(scratch_dir, PHI_TAGS_SNAPSHOT_NAME)

    final_output = os.path.join(out_dir, FINAL_OUTPUT_NAME)
    pipeline_audit_snapshot = os.path.join(out_dir, PIPELINE_AUDIT_SNAPSHOT_NAME)
    tag_audit_snapshot = os.path.join(out_dir, TAG_AUDIT_SNAPSHOT_NAME)

    print(f"\n{'#'*60}\n# {os.path.basename(input_path)}\n{'#'*60}")

    # force=True: a batch must not die on a file missing the DICM preamble —
    # the pipeline's own error handling reports it per file instead. Under a
    # job config, fail_on_unparsable turns that off inside the pipeline, which
    # is where the file is actually de-identified.
    ds = pydicom.dcmread(input_path, force=True)

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
        deid_model=deid_model, medical_ner=medical_ner, gliner_model=gliner_model,
        job_config=job_config, secrets=secrets
    )

    elapsed_sec = round(time.time() - start_time, 2)
    pipeline_audit["execution_time_seconds"] = elapsed_sec

    with open(pipeline_audit_snapshot, "w") as f:
        json.dump(_publishable_audit(pipeline_audit), f, indent=2)

    # The tag-level audit also gets its own file: it's the auditable record of
    # which technique (hash/tokenise/encrypt/mask/suppress) hit which tag, and
    # downstream consumers read it on its own, not buried in the pixel audit.
    with open(tag_audit_snapshot, "w") as f:
        json.dump(pipeline_audit["deidentified_tags"], f, indent=2)

    print(f"\nPixel anonymization status: {pipeline_audit['verification_status']}")
    print(f"Redacted regions: {len(pipeline_audit['redacted_regions'])}")
    print(f"Tags de-identified: {len(pipeline_audit['deidentified_tags'])}")
    print(f"Pipeline execution time: {elapsed_sec:.2f} seconds")
    print(f"Final anonymized DICOM saved -> {final_output}")
    if pipeline_audit.get("bbox_image_path"):
        print(f"Bounding-box visualization saved -> {pipeline_audit['bbox_image_path']}")
    print(f"Full pipeline audit saved -> {pipeline_audit_snapshot}")

    return pipeline_audit


def _publishable_audit(audit):
    """The pipeline audit as it may be written to the OUTPUT volume.

    `redacted_regions` carries the OCR-recognised burned-in text — patient
    names, MRNs, exactly the PHI the pass just blacked out of the pixels.
    Writing it beside the output is the plaintext sidecar that
    emit_pixel_text_report exists to forbid, so with that pinned false the text
    is dropped and only the geometry (and the count) survives.
    """
    if EMIT_PIXEL_TEXT_REPORT:
        return audit

    published = dict(audit)
    published["redacted_regions"] = [
        {"bbox": region.get("bbox")} for region in audit.get("redacted_regions", [])
    ]
    return published


def _clear_scratch(scratch_root):
    """Removes the scratch tree holding the still-identified intermediates.

    It lives under /tmp, inside the enclave, and nothing downstream reads it —
    leaving the original tag values and the pre-de-identification DICOM lying
    around for the next job serves no purpose.
    """
    if not scratch_root or not os.path.isdir(scratch_root):
        return
    shutil.rmtree(scratch_root, ignore_errors=True)
    log.info(f"Scratch (still-identified intermediates) cleared -> {scratch_root}")


def _find_input_files():
    """Every *.dcm under INPUT_DIR, at any depth — the mounted input volume is
    routinely a study/series tree, not a flat directory."""
    return sorted(glob.glob(os.path.join(INPUT_DIR, "**", "*.dcm"), recursive=True))


def _allocate_out_dir(input_path, taken):
    """
    Output subdirectory for `input_path`, named after the file's own basename.
    Because the scan is recursive, two files in different sub-directories can
    share a basename; the second one gets a `_2` suffix rather than silently
    overwriting the first. `taken` tracks names used in THIS run only, so a
    re-run against the same output volume overwrites in place as before
    instead of piling up copies.
    """
    stem = os.path.splitext(os.path.basename(input_path))[0]
    name = stem
    n = 1
    while name in taken:
        n += 1
        name = f"{stem}_{n}"
    taken.add(name)
    return os.path.join(OUTPUT_DIR, name)


def _write_manifest(results, job_config=None):
    """Writes OUTPUT_DIR/manifest.json — the `manifest` the response's
    `outputs.dicom` block carries. Every field is optional to the UI, so the
    shape only ever grows.

    It is written for every run, including a direct upload (ingest_mode
    "upload"), which historically reported only outputs.direct.outputBlobUrl
    with no manifest and so lost the manifest panel. It costs nothing to
    produce here, and the middleware can attach it either way.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUTPUT_DIR, MANIFEST_NAME)
    manifest = {"files": results}
    if job_config is not None:
        manifest["dataset_name"] = job_config.dataset_name
        manifest["ingest_mode"] = job_config.ingest_mode
        if job_config.unresolved_keywords:
            manifest["unresolved_tag_actions"] = job_config.unresolved_keywords
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest_path


def _summarize(audit, input_path, rel_path, out_dir):
    """One manifest entry for one processed file.

    `tags_by_technique` is keyed off the technique identifiers the run actually
    used — under a job config those are the six wire action names, so they line
    up exactly with what the UI sent. Only tags that were really changed are
    counted: an entry recorded because a tag was empty was not touched.
    """
    by_technique = {}
    touched = 0
    for entry in audit["deidentified_tags"]:
        if entry.get("changed", True) is False:
            continue
        touched += 1
        by_technique[entry["technique"]] = by_technique.get(entry["technique"], 0) + 1

    return {
        "file": os.path.basename(input_path),
        "input_path": rel_path,
        "output_dir": out_dir,
        "pixel_verification_status": audit["verification_status"],
        "redacted_regions": len(audit["redacted_regions"]),
        "tags_touched": touched,
        "tags_by_technique": by_technique,
        "execution_time_seconds": audit.get("execution_time_seconds"),
        "error": audit.get("error"),
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # The decrypted job config, if the middleware dropped one on the config
    # mount. None -> no `dicom_deidentify` block (or no config at all), which
    # means the original fixed-policy path runs unchanged. A config that IS
    # present but cannot be parsed is fatal: falling back to a different policy
    # than the user chose, silently, is the worst of the three outcomes.
    job_config = load_job_config()

    input_files = _find_input_files()
    if not input_files:
        log.warning(f"No .dcm files found under {INPUT_DIR} — nothing to process.")
        _write_manifest([])
        return

    print(f"Found {len(input_files)} file(s) under {INPUT_DIR}/ to process.")

    use_gpu = check_gpu_available()
    paddle_ocr, easy_ocr, analyzer, deid_model, medical_ner, gliner_model = initialize_engines(use_gpu=use_gpu)

    # Under a job config the keystore holds this run's encrypt/tokenise key
    # material and belongs in the scratch tree under /tmp, not on the output
    # volume: a key that leaves the enclave beside the data it protects is a
    # re-identification key, which is the same objection that keeps
    # emit_uid_crosswalk pinned false. Without a job config it stays where it
    # has always been, so batches keep correlating across runs.
    scratch_root = TEMP_DIR if job_config else None
    keystore_path = (
        os.path.join(scratch_root, "keystore", "secured.json")
        if scratch_root else SECURED_KEYSTORE_FILE
    )
    keystore = KeyStore(keystore_path)

    # The hashing_with_salt salt: minted once here, shared by every file in the
    # batch so equal values still hash equal and records join, and discarded at
    # process exit so the next run produces unrelated hashes.
    secrets = RunSecrets() if job_config else None

    results = []
    taken = set()
    for input_path in input_files:
        out_dir = _allocate_out_dir(input_path, taken)
        rel_path = os.path.relpath(input_path, INPUT_DIR)
        scratch_dir = (
            os.path.join(scratch_root, os.path.basename(out_dir))
            if scratch_root else None
        )
        try:
            audit = process_file(input_path, out_dir, paddle_ocr, easy_ocr, analyzer,
                                 keystore, deid_model=deid_model,
                                 medical_ner=medical_ner, gliner_model=gliner_model,
                                 job_config=job_config, secrets=secrets,
                                 scratch_dir=scratch_dir)
        except Exception as e:
            # One unreadable file must not take the rest of the batch with it.
            log.error(f"[ERROR] {input_path}: {e}", exc_info=True)
            results.append({
                "file": os.path.basename(input_path),
                "input_path": rel_path,
                "output_dir": out_dir,
                "error": str(e),
            })
            continue

        results.append(_summarize(audit, input_path, rel_path, out_dir))

    keystore.save()
    print(f"\nShared key/token material for this batch saved -> {keystore_path}")

    manifest_path = _write_manifest(results, job_config)
    _clear_scratch(scratch_root)

    print(f"\n{'='*60}\nBatch complete: {len(results)} file(s) processed.")
    for entry in results:
        status = entry.get("pixel_verification_status") or f"ERROR: {entry.get('error')}"
        print(f"  {entry['file']:<30} {status}")
    print(f"Manifest -> {manifest_path}")


if __name__ == "__main__":
    main()
