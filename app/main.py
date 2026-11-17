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

Parallelism (see runtime.py for how the machine is sized):
  - The models load once, in this process.
  - The pixel stage (OCR, classification, redaction, verification: all the
    model work) runs in worker processes forked from it, so the workers
    share the model weights copy-on-write instead of loading their own.
  - The tag stage runs here, one file at a time, in sorted input order,
    while the workers carry on with later files.

One KeyStore is used only by this process and saved once at the end. So the
same PatientID/AccessionNumber/etc. hashes or tokenises to the same value in
every file, and tokens are minted in the same order as a serial run.

Which policy runs is decided by the decrypted SPIDEr job config on the config
mount (de_identification/job_config.py):

  no file, or no `dicom_deidentify` block
      The fixed tag policy (tag_mapping.py) and inpainting. Intermediates stay
      beside the output; the keystore is SECURED_KEYSTORE_FILE and persists.
  a `dicom_deidentify` block
      The user's per-tag choices (`tag_actions`; unlisted tags are kept), a
      black fill and strict verification. The still-identified intermediates
      (data.json, phi_tags.json, before_deidentification.dcm) and the keystore
      go to a per-run scratch directory under TEMP_DIR (/tmp), deleted at the
      end: OUTPUT_DIR is the volume that leaves the enclave. The
      hashing_with_salt salt is per run, in memory only.

Per input file, OUTPUT_DIR/<name>/ gets after_deidentification.dcm,
after_preview.png, pipeline_audit.json (region geometry, status, timing; the
recognised text is stripped) and tag_audit.json (technique per tag, no
values). OUTPUT_DIR/manifest.json summarises the run; the enclave attaches it
to the job status and the UI shows its tags_by_technique.
"""

import glob
import json
import os
import re
import shutil
import sys
import tempfile
import time

import runtime
from config import (
    INPUT_DIR,
    OUTPUT_DIR,
    BEFORE_OUTPUT_NAME,
    FINAL_OUTPUT_NAME,
    DATA_SNAPSHOT_NAME,
    PHI_TAGS_SNAPSHOT_NAME,
    PIPELINE_AUDIT_SNAPSHOT_NAME,
    TAG_AUDIT_SNAPSHOT_NAME,
    MANIFEST_NAME,
    SECURED_KEYSTORE_FILE,
    EMIT_PIXEL_TEXT_REPORT,
    TEMP_DIR,
    log,
)

# Model handles for the pixel stage. Set in this process before the workers
# are forked; each worker inherits them.
_ENGINES = None
# Set when the worker pool failed and the batch finished serially.
_FALLBACK_NOTE = None
# The parsed job config (None: fixed policy), this run's scratch directory
# (job config only) and each input's output-directory name. Set before the
# workers are forked; each worker inherits them.
_JOB_CONFIG = None
_SCRATCH_DIR = None
_OUT_NAMES = {}


def _safe_stem(input_path):
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)[:64] or "file"


def _allocate_out_names(input_files):
    """One output-directory name per input. The scan can be recursive, so two
    files may share a basename: the second gets `_2` instead of overwriting
    the first's output."""
    names, taken = {}, set()
    for path in input_files:
        stem = name = _safe_stem(path)
        n = 1
        while name in taken:
            n += 1
            name = f"{stem}_{n}"
        taken.add(name)
        names[path] = name
    return names


def _paths(input_path):
    name = _OUT_NAMES.get(input_path) or _safe_stem(input_path)
    out_dir = os.path.join(OUTPUT_DIR, name)
    # Files that still hold original values go to the scratch tree under a
    # job config, never to the output volume.
    work_dir = os.path.join(_SCRATCH_DIR, name) if _SCRATCH_DIR else out_dir
    return {
        "out_dir": out_dir,
        "work_dir": work_dir,
        "before": os.path.join(work_dir, BEFORE_OUTPUT_NAME),
        "data": os.path.join(work_dir, DATA_SNAPSHOT_NAME),
        "phi_tags": os.path.join(work_dir, PHI_TAGS_SNAPSHOT_NAME),
        "final": os.path.join(out_dir, FINAL_OUTPUT_NAME),
        "audit": os.path.join(out_dir, PIPELINE_AUDIT_SNAPSHOT_NAME),
        "tag_audit": os.path.join(out_dir, TAG_AUDIT_SNAPSHOT_NAME),
    }


def _publishable_audit(audit):
    """The audit as written to the output volume: the recognised burned-in
    text (the PHI the pass just removed from the pixels) is dropped and only
    the geometry survives, since EMIT_PIXEL_TEXT_REPORT is pinned false."""
    if EMIT_PIXEL_TEXT_REPORT:
        return audit
    published = dict(audit)
    published["redacted_regions"] = [
        {"bbox": r.get("bbox")} for r in audit.get("redacted_regions", [])
    ]
    return published


def pixel_stage(input_path):
    """Step 1 and the pixel pipeline for one file. Safe to run in a worker."""
    from phi_tags import dump_original_tags, identify_phi_tags
    from pipeline import read_dicom, redact_file_pixels

    paddle_ocr, analyzer, deid_model, medical_ner, gliner_model = _ENGINES
    start_time = time.time()
    p = _paths(input_path)

    print(f"\n{'#'*60}\n# {os.path.basename(input_path)}\n{'#'*60}")

    try:
        os.makedirs(p["out_dir"], exist_ok=True)
        os.makedirs(p["work_dir"], exist_ok=True)
        ds = read_dicom(input_path, _JOB_CONFIG)

        # Snapshot every original tag value, for reference (no tag is ever modified)
        dump_original_tags(ds, p["data"])
        print(f"Original tags saved -> {p['data']}\n")

        # ── Step 1: identify which tags carry PHI (read-only, ds untouched) ────
        phi_tags = identify_phi_tags(ds)
        with open(p["phi_tags"], "w") as f:
            json.dump(phi_tags, f, indent=2)
        print(
            f"Identified {len(phi_tags)} PHI-bearing tag(s) (left unmodified). "
            f"Saved -> {p['phi_tags']}\n"
        )
        # Tag names and categories only: the values are the PHI itself.
        for t in phi_tags:
            print(f"{t['tag']:>14}  {t['field']:<28} {t['category']:<14}")

        # ── Step 2 (+ Step 3): burned-in pixel redaction ───────────────────────
        print(f"\nRunning pixel de-identification on {os.path.basename(input_path)}...")
        audit = redact_file_pixels(
            input_path,
            p["before"],
            p["final"],
            p["data"],
            paddle_ocr,
            analyzer,
            deid_model=deid_model,
            medical_ner=medical_ner,
            gliner_model=gliner_model,
            ds=ds,
            job_config=_JOB_CONFIG,
        )
    except Exception as e:
        log.error(f"  [ERROR] {os.path.basename(input_path)}: {e}", exc_info=True)
        audit = {
            "file": os.path.basename(input_path),
            "input_path": input_path,
            "before_output_path": p["before"],
            "output_path": p["final"],
            "redacted_regions": [],
            "deidentified_tags": [],
            "verification_status": "ERROR",
            "error": str(e),
        }

    audit["pixel_stage_seconds"] = round(time.time() - start_time, 2)
    return audit


def tag_stage(audit, keystore, secrets=None):
    """Tag de-identification and the audit files. Runs in the parent only."""
    from pipeline import deidentify_file_tags

    start_time = time.time()
    audit = deidentify_file_tags(audit, keystore, _JOB_CONFIG, secrets)
    audit.pop("pixel_stage_complete", None)
    pixel_seconds = audit.pop("pixel_stage_seconds", 0)
    audit["execution_time_seconds"] = round(pixel_seconds + time.time() - start_time, 2)

    if audit.get("error"):
        # A save that failed part-way can leave a truncated file behind;
        # never leave anything at the de-identified output path on failure.
        if os.path.isfile(audit["output_path"]):
            os.remove(audit["output_path"])
    elif not os.path.isfile(audit["output_path"]):
        audit["error"] = "pipeline finished but wrote no output file"
        audit["verification_status"] = "ERROR"

    p = _paths(audit["input_path"])
    os.makedirs(p["out_dir"], exist_ok=True)
    with open(p["audit"], "w") as f:
        json.dump(_publishable_audit(audit), f, indent=2)
    # Which technique hit which tag; no values, so no PHI.
    with open(p["tag_audit"], "w") as f:
        json.dump(audit.get("deidentified_tags", []), f, indent=2)

    print(f"\n[{audit['file']}]")
    print(f"Pixel anonymization status: {audit['verification_status']}")
    print(f"Redacted regions: {len(audit['redacted_regions'])}")
    print(f"Tags de-identified: {len(audit['deidentified_tags'])}")
    print(f"Pipeline execution time: {audit['execution_time_seconds']:.2f} seconds")
    if audit.get("error"):
        print(f"FAILED, no anonymized DICOM written: {audit['error']}", file=sys.stderr)
    else:
        print(f"Final anonymized DICOM saved -> {audit['output_path']}")
    if audit.get("bbox_image_path"):
        print(f"Bounding-box visualization saved -> {audit['bbox_image_path']}")
    print(f"Full pipeline audit saved -> {p['audit']}")
    return audit


def _worker_init(threads):
    runtime.apply_thread_limits(threads)
    # If memory runs out, let the kernel kill a worker rather than this
    # process: the batch then finishes serially instead of dying.
    try:
        with open("/proc/self/oom_score_adj", "w") as f:
            f.write("500")
    except OSError:
        pass


def _run_parallel(input_files, plan, keystore, secrets=None):
    """Yields finished audits in input order.

    Falls back to processing the remaining files serially in this process if
    the pool cannot start (no /dev/shm, seccomp), a worker dies (OOM killer),
    or a file takes longer than SKALD_FILE_TIMEOUT_S (a hung worker).
    """
    import multiprocessing
    import warnings
    from concurrent.futures import ProcessPoolExecutor
    from concurrent.futures import TimeoutError as FutureTimeout
    from concurrent.futures.process import BrokenProcessPool

    global _FALLBACK_NOTE

    timeout = runtime.file_timeout_s()
    done = 0
    pool = None
    try:
        # fork, not spawn: workers must inherit the loaded models. Python
        # warns about forking a process that has threads; the native pools
        # here are idle at fork time and capped again in _worker_init.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            pool = ProcessPoolExecutor(
                max_workers=plan.workers,
                mp_context=multiprocessing.get_context("fork"),
                initializer=_worker_init,
                initargs=(plan.threads,),
            )
            futures = [pool.submit(pixel_stage, f) for f in input_files]
        for fut in futures:
            audit = fut.result(timeout=timeout)
            yield tag_stage(audit, keystore, secrets)
            done += 1
        pool.shutdown(wait=True)
        return
    except (BrokenProcessPool, FutureTimeout, OSError) as e:
        reason = (
            f"no file finished within SKALD_FILE_TIMEOUT_S={timeout}s"
            if isinstance(e, FutureTimeout)
            else repr(e)
        )
        log.warning(
            f"Worker pool failed after {done} file(s) ({reason}); "
            "processing the remaining files serially in this process."
        )
        _FALLBACK_NOTE = f"pool failed after {done} file(s), rest ran serially"
    if pool is not None:
        # A hung worker would block a normal shutdown forever.
        for proc in list(getattr(pool, "_processes", {}).values()):
            proc.kill()
        pool.shutdown(wait=False, cancel_futures=True)

    runtime.apply_thread_limits(plan.cpus)
    for input_path in input_files[done:]:
        yield tag_stage(pixel_stage(input_path), keystore, secrets)


def _summarize(audit):
    """One manifest entry. tags_by_technique counts only tags really changed,
    keyed by the technique names the run used: under a job config, the UI's
    own action names."""
    by_technique, touched = {}, 0
    for entry in audit.get("deidentified_tags", []):
        if (
            entry.get("changed", True) is False
            or entry.get("action") == "skipped-empty"
        ):
            continue
        touched += 1
        by_technique[entry["technique"]] = by_technique.get(entry["technique"], 0) + 1
    return {
        "file": audit["file"],
        "input_path": os.path.relpath(audit["input_path"], INPUT_DIR),
        "output_dir": _paths(audit["input_path"])["out_dir"],
        "pixel_verification_status": audit.get("verification_status"),
        "redacted_regions": len(audit.get("redacted_regions", [])),
        "tags_touched": touched,
        "tags_by_technique": by_technique,
        "execution_time_seconds": audit.get("execution_time_seconds"),
        "error": audit.get("error"),
    }


def _write_manifest(results, job_config=None):
    """OUTPUT_DIR/manifest.json. The enclave attaches it (minus file names and
    paths) to the job status; the UI shows tags_by_technique."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest = {"files": [_summarize(a) for a in results]}
    if job_config is not None:
        manifest["dataset_name"] = job_config.dataset_name
        manifest["ingest_mode"] = job_config.ingest_mode
        if job_config.unresolved_keywords:
            manifest["unresolved_tag_actions"] = job_config.unresolved_keywords
    path = os.path.join(OUTPUT_DIR, MANIFEST_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def main():
    global _ENGINES, _JOB_CONFIG, _SCRATCH_DIR, _OUT_NAMES

    from de_identification.job_config import JobConfigError, load_job_config

    # A config that is present but cannot be understood is fatal: silently
    # running a different policy than the user chose is the worst outcome.
    try:
        _JOB_CONFIG = load_job_config()
    except JobConfigError as e:
        print(f"ERROR: invalid job config: {e}", file=sys.stderr)
        return 1
    policy = "job config (tag_actions)" if _JOB_CONFIG else "fixed policy"
    print(f"Tag policy: {policy}")

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.dcm")))
    if not input_files:
        input_files = sorted(
            glob.glob(os.path.join(INPUT_DIR, "**", "*.dcm"), recursive=True)
        )
    if not input_files:
        _write_manifest([], _JOB_CONFIG)
        print(f"ERROR: no .dcm files found in {INPUT_DIR}/", file=sys.stderr)
        return 1

    print(f"Found {len(input_files)} file(s) to process.")
    _OUT_NAMES = _allocate_out_names(input_files)

    # Size the run before numpy/torch/paddle/cv2 load: they read their thread
    # settings once, at import.
    plan = runtime.configure(len(input_files))
    print(f"Execution plan: {plan.describe()}")

    from engines import check_gpu_available, initialize_engines
    from de_identification.actions import RunSecrets
    from de_identification.keystore import KeyStore

    # GNU OpenMP (torch's CPU backend) is not fork-safe: if this process
    # starts an OpenMP thread pool (loading weights does), forked workers
    # inherit a pool whose threads do not exist and hang on their first
    # parallel op. With workers, this process never runs inference, so it
    # loads single-threaded and each worker sets its own count after the
    # fork (_worker_init). Paddle's per-predictor cpu_threads applies only
    # when a predictor runs, so it is set to the per-worker value here.
    runtime.apply_thread_limits(1 if plan.workers > 1 else plan.threads)
    use_gpu = check_gpu_available()
    _ENGINES = initialize_engines(
        use_gpu=use_gpu, cpu_threads=plan.threads, lazy_gliner=plan.workers == 1
    )

    # One KeyStore, owned by this process, saved once at the end, so every
    # file's hash/tokenise/encrypt values stay consistent with each other.
    # Under a job config it holds only this run's keys and lives in the
    # scratch directory: a key that leaves the enclave beside the data it
    # protects is a re-identification key. The hashing_with_salt salt is
    # minted once per run and never written anywhere.
    if _JOB_CONFIG is not None:
        os.makedirs(TEMP_DIR, mode=0o700, exist_ok=True)
        _SCRATCH_DIR = tempfile.mkdtemp(prefix="run-", dir=TEMP_DIR)
        keystore_path = os.path.join(_SCRATCH_DIR, "keystore", "secured.json")
        secrets = RunSecrets()
    else:
        keystore_path = SECURED_KEYSTORE_FILE
        secrets = None
    keystore = KeyStore(keystore_path)

    try:
        batch_start = time.time()
        if plan.workers > 1:
            results = list(_run_parallel(input_files, plan, keystore, secrets))
        else:
            results = [
                tag_stage(pixel_stage(f), keystore, secrets) for f in input_files
            ]
        wall = time.time() - batch_start

        keystore.save()
        print(f"\nShared key/token material for this batch saved -> {keystore_path}")
        manifest_path = _write_manifest(results, _JOB_CONFIG)
    finally:
        # The scratch tree holds original tag values, the un-de-identified
        # checkpoint and this run's keys; nothing reads it after the batch.
        if _SCRATCH_DIR:
            shutil.rmtree(_SCRATCH_DIR, ignore_errors=True)
            print(f"Scratch (still-identified intermediates) cleared -> {_SCRATCH_DIR}")

    total_file_time = sum(a.get("execution_time_seconds", 0) for a in results)
    avg_time = total_file_time / len(results) if results else 0

    print(f"\n{'='*75}")
    mode = f"{plan.workers} worker(s) x {plan.threads} thread(s)"
    if _FALLBACK_NOTE:
        mode += f"; {_FALLBACK_NOTE}"
    print(
        f"BATCH SUMMARY: {len(results)} file(s) in {wall:.2f}s wall clock "
        f"({mode}; {avg_time:.2f}s/file per worker)"
    )
    print(f"{'='*75}")
    print(f"{'File Name':<42} {'Status':<16} {'Regions':<8} {'Time (s)':<8}")
    print(f"{'-'*42} {'-'*16} {'-'*8} {'-'*8}")
    for audit in results:
        t_sec = audit.get("execution_time_seconds", 0)
        num_regions = len(audit.get("redacted_regions", []))
        print(
            f"{audit['file'][:40]:<42} {audit['verification_status']:<16} {num_regions:<8} {t_sec:>6.2f}s"
        )
    print(f"{'='*75}")
    print(f"Manifest -> {manifest_path}\n")

    # A batch with any failed file exits non-zero, with each file's real
    # error, so the job that ran the container reports the cause rather than
    # finding no output.
    failed = [a for a in results if a.get("error")]
    if failed:
        print(
            f"ERROR: {len(failed)} of {len(results)} file(s) failed:", file=sys.stderr
        )
        for audit in failed:
            print(f"  {audit['file']}: {audit['error']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
