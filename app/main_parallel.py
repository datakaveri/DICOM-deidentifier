"""
main_parallel.py — High-throughput parallel DICOM de-identification batch runner.
Optimized for multi-core VMs (e.g. 16 CPU Cores / 125 GB RAM).

Key Features:
  - Process-level parallelism with persistent worker engines (models loaded once per worker).
  - Model cache pre-warming: all 5 AI model weights downloaded once in the main process
    before spawning workers, so workers load from local disk cache (60-80% faster init).
  - PaddleOCR runs on text-region crops only (not full image) — major per-file speedup.
  - Paddle MKLDNN disabled and internal thread pool capped — fixes VM crashes & thrashing.
  - forkserver context on Linux (imports happen once, then workers fork from server).
  - Dynamic worker auto-tuning based on available CPU cores and RAM.
  - Safe error handling & full audit snapshot generation.
"""

# ── 0. Paddle environment hardening (MUST be first) ──────────────────────────
import paddle_env  # noqa: F401 — sets env vars before any Paddle/PyTorch import

import os
import sys
import glob
import time
import json
import re
import argparse
import multiprocessing as mp

# ── 1. Determine Worker & Thread Allocation ──────────────────────────────────
# Optimal config for a 16-core CPU-only VM:
#   - 6 workers × 2 threads each = 12 math threads + PaddleOCR internal ~2 = ~24
#   - Staggered I/O keeps effective utilization at ~16 cores.
#   - More workers = better pipeline overlap between I/O-bound and CPU-bound stages.
#
# Why NOT 2 workers × 8 threads:
#   PaddleOCR internally spawns its own threads (via Paddle/OpenMP/MKL).
#   With 8 threads per worker, total real threads = 2 × (8 + ~8) = ~32 → severe thrashing.

TOTAL_CPUS = os.cpu_count() or 4


def _auto_tune():
    """
    Returns (workers, threads_per_worker) tuned for the current machine.

    Strategy:
      - Each PaddleOCR worker internally uses ~2 extra threads on top of the
        explicit thread cap, so effective_threads_per_worker ≈ threads + 2.
      - We aim for total_effective ≈ TOTAL_CPUS × 1.25  (slight oversubscription
        keeps cores busy during I/O gaps, but not so much that we thrash).
      - Each worker needs ~3.2 GB RAM for models + ~0.2 GB for data buffers.
    """
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        available_gb = 125.0  # fallback for VMs without psutil

    threads = 2  # sweet spot: prevents internal thread explosion
    paddle_internal_threads = 2  # PaddleOCR's own threads on top of our cap
    effective_per_worker = threads + paddle_internal_threads
    ram_per_worker_gb = 3.5  # ~3.2 GB models + 0.3 GB data buffers

    # Max workers by CPU (allow ~1.25× oversubscription for I/O overlap)
    max_by_cpu = max(1, int(TOTAL_CPUS * 1.25 / effective_per_worker))
    # Max workers by RAM (leave 8 GB for OS + main process)
    max_by_ram = max(1, int((available_gb - 8) / ram_per_worker_gb))
    # Final: take the smaller limit, but at least 2 workers
    workers = max(2, min(max_by_cpu, max_by_ram, 10))

    return workers, threads


AUTO_WORKERS, AUTO_THREADS = _auto_tune()

parser = argparse.ArgumentParser(description="Parallel DICOM De-Identification Pipeline")
parser.add_argument("--workers", "-w", type=int,
                    default=int(os.getenv("SKALD_NUM_WORKERS", AUTO_WORKERS)),
                    help=f"Number of parallel worker processes (auto-tuned default: {AUTO_WORKERS})")
parser.add_argument("--threads", "-t", type=int,
                    default=int(os.getenv("SKALD_THREADS_PER_WORKER", AUTO_THREADS)),
                    help=f"CPU threads per worker for PyTorch/Paddle (auto-tuned default: {AUTO_THREADS})")
parser.add_argument("--input-dir", "-i", type=str, default=None,
                    help="Custom input directory containing .dcm files")
parser.add_argument("--output-dir", "-o", type=str, default=None,
                    help="Custom output directory to store de-identified files and logs")
parser.add_argument("--skip-warmup", action="store_true",
                    help="Skip model cache pre-warming (use if models are already cached)")
args, _ = parser.parse_known_args()

NUM_WORKERS = args.workers
THREADS_PER_WORKER = str(args.threads)

# Set thread caps BEFORE importing deep learning libraries
os.environ["SKALD_THREADS_PER_WORKER"] = THREADS_PER_WORKER
os.environ["SKALD_PADDLE_THREADS"] = THREADS_PER_WORKER
os.environ["OMP_NUM_THREADS"] = THREADS_PER_WORKER
os.environ["MKL_NUM_THREADS"] = THREADS_PER_WORKER
os.environ["OPENBLAS_NUM_THREADS"] = THREADS_PER_WORKER
os.environ["VECLIB_MAXIMUM_THREADS"] = THREADS_PER_WORKER
os.environ["NUMEXPR_NUM_THREADS"] = THREADS_PER_WORKER

import torch
import cv2
import pydicom

torch.set_num_threads(int(THREADS_PER_WORKER))
cv2.setNumThreads(int(THREADS_PER_WORKER))

from config import (
    INPUT_DIR, OUTPUT_DIR, BEFORE_OUTPUT_NAME, FINAL_OUTPUT_NAME,
    DATA_SNAPSHOT_NAME, PHI_TAGS_SNAPSHOT_NAME, PIPELINE_AUDIT_SNAPSHOT_NAME,
    SECURED_KEYSTORE_FILE,
)
from phi_tags import dump_original_tags, identify_phi_tags
from engines import check_gpu_available, initialize_engines, pre_warm_model_cache
from pipeline import anonymize_dicom_file
from de_identification.keystore import KeyStore

# Global variables within each worker process
_engines = None
_keystore = None


def init_worker(threads_per_worker):
    """
    Initializes all 5 AI/OCR models ONCE when each worker process starts.
    Subsequent files processed by this worker incur zero model load overhead.

    Models load from LOCAL DISK CACHE (populated by pre_warm_model_cache()
    in the main process) — no re-download needed.
    """
    global _engines, _keystore

    # Re-apply thread caps in the child process (spawn/forkserver start fresh)
    os.environ["SKALD_THREADS_PER_WORKER"] = str(threads_per_worker)
    os.environ["SKALD_PADDLE_THREADS"] = str(threads_per_worker)
    os.environ["OMP_NUM_THREADS"] = str(threads_per_worker)
    os.environ["MKL_NUM_THREADS"] = str(threads_per_worker)
    os.environ["OPENBLAS_NUM_THREADS"] = str(threads_per_worker)

    torch.set_num_threads(int(threads_per_worker))
    cv2.setNumThreads(int(threads_per_worker))

    t0 = time.time()
    use_gpu = check_gpu_available()
    _engines = initialize_engines(use_gpu=use_gpu)
    _keystore = KeyStore(SECURED_KEYSTORE_FILE)
    elapsed = round(time.time() - t0, 1)
    print(f"  [Worker {os.getpid()}] Ready in {elapsed}s")


def process_single_file(input_path):
    """
    Processes one DICOM file through the full de-identification pipeline.
    """
    global _engines, _keystore
    if _engines is None:
        init_worker(THREADS_PER_WORKER)

    paddle_ocr, analyzer, deid_model, medical_ner, gliner_model = _engines
    start_time = time.time()

    stem = os.path.splitext(os.path.basename(input_path))[0]
    safe_stem = re.sub(r'[^a-zA-Z0-9_\-]', '_', stem)[:64]
    target_out_dir = args.output_dir or OUTPUT_DIR
    out_dir = os.path.join(target_out_dir, safe_stem)
    os.makedirs(out_dir, exist_ok=True)

    before_output = os.path.join(out_dir, BEFORE_OUTPUT_NAME)
    final_output = os.path.join(out_dir, FINAL_OUTPUT_NAME)
    data_snapshot = os.path.join(out_dir, DATA_SNAPSHOT_NAME)
    phi_tags_snapshot = os.path.join(out_dir, PHI_TAGS_SNAPSHOT_NAME)
    pipeline_audit_snapshot = os.path.join(out_dir, PIPELINE_AUDIT_SNAPSHOT_NAME)

    try:
        ds = pydicom.dcmread(input_path, force=True)
        if "TransferSyntaxUID" not in getattr(ds, "file_meta", {}):
            ds.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

        dump_original_tags(ds, data_snapshot)

        phi_tags = identify_phi_tags(ds)
        with open(phi_tags_snapshot, "w") as f:
            json.dump(phi_tags, f, indent=2)

        pipeline_audit = anonymize_dicom_file(
            input_path, before_output, final_output, data_snapshot,
            paddle_ocr, analyzer, _keystore,
            deid_model=deid_model, medical_ner=medical_ner, gliner_model=gliner_model
        )

        elapsed_sec = round(time.time() - start_time, 2)
        pipeline_audit["execution_time_seconds"] = elapsed_sec

        with open(pipeline_audit_snapshot, "w") as f:
            json.dump(pipeline_audit, f, indent=2)

        print(f"  [DONE] {os.path.basename(input_path)} -> {pipeline_audit['verification_status']} ({elapsed_sec:.2f}s)")
        return pipeline_audit

    except Exception as e:
        elapsed_sec = round(time.time() - start_time, 2)
        print(f"  [ERROR] {os.path.basename(input_path)} failed: {e}")
        return {
            "file": os.path.basename(input_path),
            "verification_status": f"FAILED ({e})",
            "redacted_regions": [],
            "deidentified_tags": [],
            "execution_time_seconds": elapsed_sec,
            "error": str(e)
        }


def _choose_mp_context():
    """
    Picks the fastest safe multiprocessing start method:
      - forkserver (Linux): imports happen once in the server process,
        then each worker forks from the server. Much faster than spawn.
      - spawn (Windows/macOS): full re-import per worker (no fork).
    """
    if sys.platform.startswith("linux"):
        try:
            return mp.get_context("forkserver")
        except ValueError:
            pass
    return mp.get_context("spawn")


def main():
    target_dir = args.input_dir or INPUT_DIR
    input_files = sorted(glob.glob(os.path.join(target_dir, "*.dcm")))
    if not input_files:
        input_files = sorted(glob.glob(os.path.join(target_dir, "**", "*.dcm"), recursive=True))
    if not input_files:
        print(f"No .dcm files found in {target_dir}/")
        return

    target_out = args.output_dir or OUTPUT_DIR

    print(f"\n{'='*75}")
    print(f"PARALLEL DICOM DE-IDENTIFICATION PIPELINE")
    print(f"  Input Files       : {len(input_files)}")
    print(f"  Parallel Workers  : {NUM_WORKERS}")
    print(f"  Threads / Worker  : {THREADS_PER_WORKER}")
    print(f"  CPU Cores         : {TOTAL_CPUS}")
    print(f"  Target Output     : {target_out}")
    print(f"  MP Context        : {'forkserver' if sys.platform.startswith('linux') else 'spawn'}")
    print(f"{'='*75}\n")

    # ── Pre-warm model cache ONCE in main process ─────────────────────────────
    # Downloads all model weights to local disk cache so workers load from
    # cache instead of re-downloading. Cuts worker init from ~120s to ~15s.
    if not args.skip_warmup:
        use_gpu = check_gpu_available()
        pre_warm_model_cache(use_gpu=use_gpu)
    else:
        print("[SKIP] Model cache pre-warming skipped (--skip-warmup).\n")

    batch_start = time.time()

    # ── Launch worker pool ────────────────────────────────────────────────────
    ctx = _choose_mp_context()
    print(f"Spawning {NUM_WORKERS} workers (each loading 5 AI models from cache)...\n")

    with ctx.Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(THREADS_PER_WORKER,)) as pool:
        results = []
        # imap_unordered for maximum throughput (process files as workers free up)
        for i, audit in enumerate(pool.imap_unordered(process_single_file, input_files), 1):
            results.append(audit)
            elapsed_so_far = time.time() - batch_start
            avg_per_file = elapsed_so_far / i
            remaining = (len(input_files) - i) * avg_per_file
            print(f"  Progress: {i}/{len(input_files)} files | "
                  f"Elapsed: {elapsed_so_far:.0f}s | "
                  f"ETA: {remaining:.0f}s | "
                  f"Avg: {avg_per_file:.1f}s/file")

    # Save shared keystore
    keystore = KeyStore(SECURED_KEYSTORE_FILE)
    keystore.save()

    total_batch_time = round(time.time() - batch_start, 2)
    avg_time = total_batch_time / len(results) if results else 0

    print(f"\n{'='*75}")
    print(f"PARALLEL BATCH COMPLETE: {len(results)} file(s) in {total_batch_time:.2f}s (Throughput: {avg_time:.2f}s/file)")
    print(f"{'='*75}")
    print(f"{'File Name':<42} {'Status':<18} {'Regions':<8} {'Time (s)':<8}")
    print(f"{'-'*42} {'-'*18} {'-'*8} {'-'*8}")
    for audit in sorted(results, key=lambda x: x.get("file", "")):
        t_sec = audit.get("execution_time_seconds", 0)
        num_regions = len(audit.get("redacted_regions", []))
        status = audit.get("verification_status", "UNKNOWN")
        print(f"{audit['file'][:40]:<42} {status:<18} {num_regions:<8} {t_sec:>6.2f}s")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
