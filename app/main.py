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

import datetime
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
    REPORT_DIR,
    OUTPUT_LAYOUT,
    BEFORE_OUTPUT_NAME,
    FINAL_OUTPUT_NAME,
    DATA_SNAPSHOT_NAME,
    PHI_TAGS_SNAPSHOT_NAME,
    PIPELINE_AUDIT_SNAPSHOT_NAME,
    TAG_AUDIT_SNAPSHOT_NAME,
    MANIFEST_NAME,
    PREVIEW_NAME,
    BENCHMARK_FILES_NAME,
    BENCHMARK_SUMMARY_NAME,
    SECURED_KEYSTORE_FILE,
    EMIT_PIXEL_TEXT_REPORT,
    TEMP_DIR,
    PRESERVE_TAGS,
    DISCARD_INTERMEDIATES,
    RESUME,
    KEYSTORE_SAVE_EVERY,
    log,
)

# Model handles for the pixel stage. Set in this process before the workers
# are forked; each worker inherits them.
_ENGINES = None
# Set when the worker pool failed and the batch finished serially.
_FALLBACK_NOTE = None
# The parsed job config (None: fixed policy), this run's scratch directory
# (job config or SKALD_DISCARD_INTERMEDIATES) and each input's output name.
# Set before the workers are forked; each worker inherits them.
_JOB_CONFIG = None
_SCRATCH_DIR = None
_OUT_NAMES = {}
# Batch progress, kept by the parent for progress lines, periodic keystore
# saves and the benchmark log.
_PROGRESS = {"done": 0, "total": 0, "t0": None}


def _mem_mb():
    """(current RSS, peak RSS) of this process in MB, from /proc."""
    rss = hwm = None
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) // 1024
                elif line.startswith("VmHWM:"):
                    hwm = int(line.split()[1]) // 1024
    except OSError:
        pass
    return rss, hwm


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
    """Where each artefact of `input_path` goes.

    per-file  OUTPUT_DIR/<name>/after_deidentification.dcm, audits beside it.
    mirror    OUTPUT_DIR/<relative path>, same file name; audits and preview
              under REPORT_DIR/<relative path without extension>/.
    Files that still hold original values go to the scratch tree when there
    is one (job config, or SKALD_DISCARD_INTERMEDIATES), else to the report
    directory.
    """
    if OUTPUT_LAYOUT == "mirror":
        rel = os.path.relpath(input_path, INPUT_DIR)
        final = os.path.join(OUTPUT_DIR, rel)
        name = os.path.splitext(rel)[0]
        report_dir = os.path.join(REPORT_DIR, name)
    else:
        name = _OUT_NAMES.get(input_path) or _safe_stem(input_path)
        report_dir = os.path.join(OUTPUT_DIR, name)
        final = os.path.join(report_dir, FINAL_OUTPUT_NAME)
    work_dir = os.path.join(_SCRATCH_DIR, name) if _SCRATCH_DIR else report_dir
    return {
        "out_dir": os.path.dirname(final),
        "report_dir": report_dir,
        "work_dir": work_dir,
        "before": os.path.join(work_dir, BEFORE_OUTPUT_NAME),
        "data": os.path.join(work_dir, DATA_SNAPSHOT_NAME),
        "phi_tags": os.path.join(work_dir, PHI_TAGS_SNAPSHOT_NAME),
        "final": final,
        "audit": os.path.join(report_dir, PIPELINE_AUDIT_SNAPSHOT_NAME),
        "tag_audit": os.path.join(report_dir, TAG_AUDIT_SNAPSHOT_NAME),
        "preview": os.path.join(report_dir, PREVIEW_NAME),
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
        os.makedirs(p["report_dir"], exist_ok=True)
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
            preview_path=p["preview"],
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
    rss, hwm = _mem_mb()
    audit["worker"] = {"pid": os.getpid(), "rss_mb": rss, "peak_rss_mb": hwm}
    return audit


def _benchmark_record(audit, tag_seconds):
    try:
        size = os.path.getsize(audit["input_path"])
    except OSError:
        size = None
    return {
        "file": os.path.relpath(audit["input_path"], INPUT_DIR),
        "size_bytes": size,
        "image_size": audit.get("image_size"),
        "bits_allocated": audit.get("bits_allocated"),
        "source_transfer_syntax": audit.get("source_transfer_syntax"),
        "status": audit.get("verification_status"),
        "error": audit.get("error"),
        "ocr_detections": audit.get("ocr_detections"),
        "tag_matched_regions": audit.get("tag_matched_regions"),
        "redacted_regions": len(audit.get("redacted_regions", [])),
        "tags_changed": sum(
            1 for e in audit.get("deidentified_tags", []) if e.get("changed", True)
        ),
        "seconds": {
            **audit.get("timings", {}),
            "pixel_stage": audit.get("pixel_stage_seconds"),
            "tag_stage": round(tag_seconds, 4),
            "total": audit.get("execution_time_seconds"),
        },
        "worker": audit.get("worker"),
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def tag_stage(audit, keystore, secrets=None):
    """Tag de-identification, the audit files and the benchmark record. Runs
    in the parent only."""
    from pipeline import deidentify_file_tags

    start_time = time.time()
    audit = deidentify_file_tags(audit, keystore, _JOB_CONFIG, secrets)
    audit.pop("pixel_stage_complete", None)
    pixel_seconds = audit.pop("pixel_stage_seconds", 0)
    tag_seconds = time.time() - start_time
    audit["pixel_stage_seconds"] = pixel_seconds
    audit["execution_time_seconds"] = round(pixel_seconds + tag_seconds, 2)

    if audit.get("error"):
        # A save that failed part-way can leave a truncated file behind;
        # never leave anything at the de-identified output path on failure.
        if os.path.isfile(audit["output_path"]):
            os.remove(audit["output_path"])
    elif not os.path.isfile(audit["output_path"]):
        audit["error"] = "pipeline finished but wrote no output file"
        audit["verification_status"] = "ERROR"

    p = _paths(audit["input_path"])
    os.makedirs(p["report_dir"], exist_ok=True)
    with open(p["audit"], "w") as f:
        json.dump(_publishable_audit(audit), f, indent=2)
    # Which technique hit which tag; no values, so no PHI.
    with open(p["tag_audit"], "w") as f:
        json.dump(audit.get("deidentified_tags", []), f, indent=2)
    # The still-identified intermediates are not needed once the file is
    # done; with a scratch tree they are deleted now, not at the end of the
    # batch, so scratch never holds more than the files in flight.
    if _SCRATCH_DIR:
        shutil.rmtree(p["work_dir"], ignore_errors=True)

    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, BENCHMARK_FILES_NAME), "a") as f:
        f.write(json.dumps(_benchmark_record(audit, tag_seconds)) + "\n")

    _PROGRESS["done"] += 1
    if KEYSTORE_SAVE_EVERY > 0 and _PROGRESS["done"] % KEYSTORE_SAVE_EVERY == 0:
        keystore.save()

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

    done, total = _PROGRESS["done"], _PROGRESS["total"]
    if _PROGRESS["t0"] and total:
        elapsed = time.time() - _PROGRESS["t0"]
        eta = elapsed / done * (total - done)
        rss, hwm = _mem_mb()
        print(
            f"PROGRESS {done}/{total} ({100 * done / total:.1f}%) "
            f"elapsed {elapsed / 60:.1f} min, eta {eta / 60:.1f} min, "
            f"parent rss {rss} MB peak {hwm} MB",
            flush=True,
        )
    return audit


def _already_done(input_path):
    """The audit of an earlier successful run of `input_path`, or None."""
    p = _paths(input_path)
    if not (os.path.isfile(p["final"]) and os.path.isfile(p["audit"])):
        return None
    try:
        with open(p["audit"]) as f:
            audit = json.load(f)
        with open(p["tag_audit"]) as f:
            audit["deidentified_tags"] = json.load(f)
    except (OSError, ValueError):
        return None
    if audit.get("error") or audit.get("input_path") != input_path:
        return None
    audit["resumed"] = True
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
        if entry.get("changed", True) is False or entry.get("action") in (
            "skipped-empty",
            "retained",
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
    """manifest.json in REPORT_DIR (OUTPUT_DIR unless set). The enclave
    attaches it (minus file names and paths) to the job status; the UI shows
    tags_by_technique."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    manifest = {"files": [_summarize(a) for a in results]}
    if job_config is not None:
        manifest["dataset_name"] = job_config.dataset_name
        manifest["ingest_mode"] = job_config.ingest_mode
        if job_config.unresolved_keywords:
            manifest["unresolved_tag_actions"] = job_config.unresolved_keywords
    path = os.path.join(REPORT_DIR, MANIFEST_NAME)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)
    return path


def _stats(values):
    vals = sorted(v for v in values if isinstance(v, (int, float)))
    if not vals:
        return None

    def pct(q):
        return round(vals[min(len(vals) - 1, int(round(q * (len(vals) - 1))))], 3)

    return {
        "n": len(vals),
        "mean": round(sum(vals) / len(vals), 3),
        "p50": pct(0.5),
        "p90": pct(0.9),
        "p99": pct(0.99),
        "max": round(vals[-1], 3),
        "sum": round(sum(vals), 1),
    }


def _read_first(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def _host_info():
    cpu_model = None
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    mem_total = None
    try:
        with open("/proc/meminfo") as f:
            mem_total = int(f.readline().split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return {
        "cpu_model": cpu_model,
        "host_cpus": os.cpu_count(),
        "host_memory_mb": mem_total,
    }


def _write_benchmark_summary(results, plan, times, resumed):
    """benchmark_summary.json in REPORT_DIR: the run's timings, throughput,
    per-stage statistics and memory peaks. Per-file detail is in
    benchmark_files.jsonl beside it."""
    import platform

    version_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "__init__.py"
    )
    m = re.search(r'__version__ = "([^"]+)"', _read_first(version_file) or "")
    __version__ = m.group(1) if m else None

    ran = [a for a in results if not a.get("resumed")]
    stage_keys = sorted({k for a in ran for k in a.get("timings", {})})
    per_stage = {
        k: _stats([a.get("timings", {}).get(k) for a in ran]) for k in stage_keys
    }
    per_stage["pixel_stage"] = _stats([a.get("pixel_stage_seconds") for a in ran])
    per_stage["file_total"] = _stats([a.get("execution_time_seconds") for a in ran])
    statuses = {}
    for a in results:
        statuses[a.get("verification_status")] = (
            statuses.get(a.get("verification_status"), 0) + 1
        )
    worker_peaks = [(a.get("worker") or {}).get("peak_rss_mb") for a in ran]
    parent_rss, parent_hwm = _mem_mb()
    cg_peak = _read_first("/sys/fs/cgroup/memory.peak")
    cg_max = _read_first("/sys/fs/cgroup/memory.max")
    wall = times["batch_end"] - times["batch_start"]
    summary = {
        "version": __version__,
        "python": platform.python_version(),
        "host": _host_info(),
        "plan": {
            "workers": plan.workers,
            "threads_per_worker": plan.threads,
            "cpus": plan.cpus,
            "memory_mb": plan.memory_mb,
            "reason": plan.reason,
            "fallback": _FALLBACK_NOTE,
        },
        "settings": {
            "policy": "job config" if _JOB_CONFIG else "fixed",
            "output_layout": OUTPUT_LAYOUT,
            "preserve_tags": list(PRESERVE_TAGS),
            "discard_intermediates": bool(_SCRATCH_DIR),
            "resume": RESUME,
        },
        "started_at": times["started_at"],
        "finished_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "seconds": {
            "startup_to_models_loaded": round(
                times["models_loaded"] - times["start"], 2
            ),
            "model_load": round(times["models_loaded"] - times["model_load_start"], 2),
            "batch_wall": round(wall, 2),
            "process_total": round(time.time() - times["start"], 2),
        },
        "files": {
            "found": len(results),
            "processed_this_run": len(ran),
            "skipped_already_done": resumed,
            "failed": sum(1 for a in results if a.get("error")),
            "by_status": statuses,
            "redacted_regions_total": sum(
                len(a.get("redacted_regions", [])) for a in ran
            ),
            "tag_matched_regions_total": sum(
                a.get("tag_matched_regions") or 0 for a in ran
            ),
            "input_bytes_total": sum(
                os.path.getsize(a["input_path"])
                for a in ran
                if os.path.exists(a["input_path"])
            ),
        },
        "throughput": {
            "files_per_minute": round(len(ran) / wall * 60, 2) if wall > 0 else None,
            "seconds_per_file_wall": round(wall / len(ran), 3) if ran else None,
        },
        "per_file_seconds": per_stage,
        "memory_mb": {
            "parent_peak_rss": parent_hwm,
            "parent_rss_at_end": parent_rss,
            "worker_peak_rss_max": max((w for w in worker_peaks if w), default=None),
            "container_peak": (
                int(cg_peak) // 2**20 if cg_peak and cg_peak.isdigit() else None
            ),
            "container_limit": (
                int(cg_max) // 2**20 if cg_max and cg_max.isdigit() else cg_max
            ),
        },
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, BENCHMARK_SUMMARY_NAME)
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    return path, summary


def main():
    global _ENGINES, _JOB_CONFIG, _SCRATCH_DIR, _OUT_NAMES

    times = {
        "start": time.time(),
        "started_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    from de_identification.job_config import JobConfigError, load_job_config
    from pydicom.datadict import tag_for_keyword

    unknown = [k for k in PRESERVE_TAGS if tag_for_keyword(k) is None]
    if unknown:
        print(
            f"ERROR: SKALD_PRESERVE_TAGS names unknown keyword(s): {unknown}",
            file=sys.stderr,
        )
        return 1

    # A config that is present but cannot be understood is fatal: silently
    # running a different policy than the user chose is the worst outcome.
    try:
        _JOB_CONFIG = load_job_config()
    except JobConfigError as e:
        print(f"ERROR: invalid job config: {e}", file=sys.stderr)
        return 1
    policy = "job config (tag_actions)" if _JOB_CONFIG else "fixed policy"
    print(f"Tag policy: {policy}; output layout: {OUTPUT_LAYOUT}")
    if PRESERVE_TAGS:
        print(f"Preserved tags (left untouched): {', '.join(PRESERVE_TAGS)}")

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

    # An interrupted batch: keep what an earlier run already finished.
    done_before = {}
    if RESUME:
        for f in input_files:
            audit = _already_done(f)
            if audit is not None:
                done_before[f] = audit
        print(f"Resume: {len(done_before)} file(s) already done, skipping them.")
    todo = [f for f in input_files if f not in done_before]

    # Size the run before numpy/torch/paddle/cv2 load: they read their thread
    # settings once, at import.
    plan = runtime.configure(max(1, len(todo)))
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
    times["model_load_start"] = time.time()
    runtime.apply_thread_limits(1 if plan.workers > 1 else plan.threads)
    use_gpu = check_gpu_available()
    _ENGINES = initialize_engines(
        use_gpu=use_gpu, cpu_threads=plan.threads, lazy_gliner=plan.workers == 1
    )
    times["models_loaded"] = time.time()
    print(f"Models loaded in {times['models_loaded'] - times['model_load_start']:.1f}s")

    # One KeyStore, owned by this process, so every file's hash/tokenise/
    # encrypt values stay consistent with each other; saved every
    # KEYSTORE_SAVE_EVERY files and at the end. Under a job config it holds
    # only this run's keys and lives in the scratch directory: a key that
    # leaves the enclave beside the data it protects is a re-identification
    # key. The hashing_with_salt salt is minted once per run and never
    # written anywhere.
    if _JOB_CONFIG is not None or DISCARD_INTERMEDIATES:
        os.makedirs(TEMP_DIR, mode=0o700, exist_ok=True)
        _SCRATCH_DIR = tempfile.mkdtemp(prefix="run-", dir=TEMP_DIR)
    if _JOB_CONFIG is not None:
        keystore_path = os.path.join(_SCRATCH_DIR, "keystore", "secured.json")
        secrets = RunSecrets()
    else:
        keystore_path = SECURED_KEYSTORE_FILE
        secrets = None
    keystore = KeyStore(keystore_path)

    _PROGRESS.update(done=0, total=len(todo), t0=time.time())
    try:
        times["batch_start"] = time.time()
        if not todo:
            results = []
        elif plan.workers > 1:
            results = list(_run_parallel(todo, plan, keystore, secrets))
        else:
            results = [tag_stage(pixel_stage(f), keystore, secrets) for f in todo]
        times["batch_end"] = time.time()
        wall = times["batch_end"] - times["batch_start"]

        keystore.save()
        print(f"\nShared key/token material for this batch saved -> {keystore_path}")
        by_path = {a["input_path"]: a for a in results}
        by_path.update(done_before)
        results = [by_path[f] for f in input_files if f in by_path]
        manifest_path = _write_manifest(results, _JOB_CONFIG)
    finally:
        # The scratch tree holds original tag values, the un-de-identified
        # checkpoint and this run's keys; nothing reads it after the batch.
        if _SCRATCH_DIR:
            shutil.rmtree(_SCRATCH_DIR, ignore_errors=True)
            print(f"Scratch (still-identified intermediates) cleared -> {_SCRATCH_DIR}")

    ran = [a for a in results if not a.get("resumed")]
    total_file_time = sum(a.get("execution_time_seconds", 0) for a in ran)
    avg_time = total_file_time / len(ran) if ran else 0

    print(f"\n{'='*75}")
    mode = f"{plan.workers} worker(s) x {plan.threads} thread(s)"
    if _FALLBACK_NOTE:
        mode += f"; {_FALLBACK_NOTE}"
    print(
        f"BATCH SUMMARY: {len(ran)} file(s) in {wall:.2f}s wall clock "
        f"({mode}; {avg_time:.2f}s/file per worker)"
        + (f"; {len(done_before)} skipped (already done)" if done_before else "")
    )
    print(f"{'='*75}")
    print(f"{'File Name':<42} {'Status':<16} {'Regions':<8} {'Time (s)':<8}")
    print(f"{'-'*42} {'-'*16} {'-'*8} {'-'*8}")
    for audit in ran:
        t_sec = audit.get("execution_time_seconds", 0)
        num_regions = len(audit.get("redacted_regions", []))
        print(
            f"{audit['file'][:40]:<42} {audit['verification_status']:<16} {num_regions:<8} {t_sec:>6.2f}s"
        )
    print(f"{'='*75}")
    print(f"Manifest -> {manifest_path}")
    bench_path, bench = _write_benchmark_summary(results, plan, times, len(done_before))
    print(
        f"Benchmark -> {bench_path}: model load {bench['seconds']['model_load']}s, "
        f"{bench['throughput']['files_per_minute']} files/min, "
        f"peak memory {bench['memory_mb']['container_peak'] or bench['memory_mb']['parent_peak_rss']} MB\n"
    )

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
