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

One KeyStore is loaded from SECURED_KEYSTORE_FILE ("secured.json") and used
only by this process, then saved once at the end. So the same
PatientID/AccessionNumber/etc. hashes or tokenises to the same value in every
file, and tokens are minted in the same order as a serial run.
"""

import glob
import json
import os
import re
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
    SECURED_KEYSTORE_FILE,
    log,
)

# Model handles for the pixel stage. Set in this process before the workers
# are forked; each worker inherits them.
_ENGINES = None
# Set when the worker pool failed and the batch finished serially.
_FALLBACK_NOTE = None


def _paths(input_path):
    stem = os.path.splitext(os.path.basename(input_path))[0]
    safe_stem = re.sub(r"[^a-zA-Z0-9_\-]", "_", stem)[:64]
    out_dir = os.path.join(OUTPUT_DIR, safe_stem)
    return {
        "out_dir": out_dir,
        "before": os.path.join(out_dir, BEFORE_OUTPUT_NAME),
        "final": os.path.join(out_dir, FINAL_OUTPUT_NAME),
        "data": os.path.join(out_dir, DATA_SNAPSHOT_NAME),
        "phi_tags": os.path.join(out_dir, PHI_TAGS_SNAPSHOT_NAME),
        "audit": os.path.join(out_dir, PIPELINE_AUDIT_SNAPSHOT_NAME),
    }


def pixel_stage(input_path):
    """Step 1 and the pixel pipeline for one file. Safe to run in a worker."""
    import pydicom

    from phi_tags import dump_original_tags, identify_phi_tags
    from pipeline import redact_file_pixels

    paddle_ocr, analyzer, deid_model, medical_ner, gliner_model = _ENGINES
    start_time = time.time()
    p = _paths(input_path)
    os.makedirs(p["out_dir"], exist_ok=True)

    print(f"\n{'#'*60}\n# {os.path.basename(input_path)}\n{'#'*60}")

    try:
        ds = pydicom.dcmread(input_path, force=True)

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
        for t in phi_tags:
            print(
                f"{t['tag']:>14}  {t['field']:<28} {t['category']:<14} {t['value'][:40]!r}"
            )

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


def tag_stage(audit, keystore):
    """Tag de-identification and the audit file. Runs in the parent only."""
    from pipeline import deidentify_file_tags

    start_time = time.time()
    audit = deidentify_file_tags(audit, keystore)
    audit.pop("pixel_stage_complete", None)
    pixel_seconds = audit.pop("pixel_stage_seconds", 0)
    audit["execution_time_seconds"] = round(pixel_seconds + time.time() - start_time, 2)

    p = _paths(audit["input_path"])
    with open(p["audit"], "w") as f:
        json.dump(audit, f, indent=2)

    print(f"\n[{audit['file']}]")
    print(f"Pixel anonymization status: {audit['verification_status']}")
    print(f"Redacted regions: {len(audit['redacted_regions'])}")
    print(f"Tags de-identified: {len(audit['deidentified_tags'])}")
    print(f"Pipeline execution time: {audit['execution_time_seconds']:.2f} seconds")
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


def _run_parallel(input_files, plan, keystore):
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
            yield tag_stage(audit, keystore)
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
        yield tag_stage(pixel_stage(input_path), keystore)


def main():
    global _ENGINES

    input_files = sorted(glob.glob(os.path.join(INPUT_DIR, "*.dcm")))
    if not input_files:
        input_files = sorted(
            glob.glob(os.path.join(INPUT_DIR, "**", "*.dcm"), recursive=True)
        )
    if not input_files:
        print(f"No .dcm files found in {INPUT_DIR}/")
        return

    print(f"Found {len(input_files)} file(s) to process.")

    # Size the run before numpy/torch/paddle/cv2 load: they read their thread
    # settings once, at import.
    plan = runtime.configure(len(input_files))
    print(f"Execution plan: {plan.describe()}")

    from engines import check_gpu_available, initialize_engines
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
    keystore = KeyStore(SECURED_KEYSTORE_FILE)

    batch_start = time.time()
    if plan.workers > 1:
        results = list(_run_parallel(input_files, plan, keystore))
    else:
        results = [tag_stage(pixel_stage(f), keystore) for f in input_files]
    wall = time.time() - batch_start

    keystore.save()
    print(
        f"\nShared key/token material for this batch saved -> {SECURED_KEYSTORE_FILE}"
    )

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
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
