"""
main_parallel.py — High-throughput parallel DICOM de-identification batch runner.
Optimized for multi-core VMs (e.g. 16 CPU Cores / 125 GB RAM).

Key Features:
  - Process-level parallelism with persistent worker engines (models loaded once per worker).
  - Intra-op thread capping per worker (prevents CPU core thrashing / over-subscription).
  - Dynamic worker auto-tuning based on available CPU cores.
  - Safe error handling & full audit snapshot generation.
"""

import os
import glob
import time
import json
import re
import argparse
import multiprocessing as mp

# ── 1. Determine Worker & Thread Allocation ──────────────────────────────────
# Default: 4 workers on a 16-core machine (4 threads per worker)
TOTAL_CPUS = os.cpu_count() or 4
DEFAULT_WORKERS = max(1, min(TOTAL_CPUS // 4, 8))
DEFAULT_THREADS = max(1, TOTAL_CPUS // DEFAULT_WORKERS)

parser = argparse.ArgumentParser(description="Parallel DICOM De-Identification Pipeline")
parser.add_argument("--workers", "-w", type=int, default=int(os.getenv("SKALD_NUM_WORKERS", DEFAULT_WORKERS)),
                    help=f"Number of parallel worker processes (default: {DEFAULT_WORKERS})")
parser.add_argument("--threads", "-t", type=int, default=int(os.getenv("SKALD_THREADS_PER_WORKER", DEFAULT_THREADS)),
                    help=f"CPU threads per worker for PyTorch/Paddle (default: {DEFAULT_THREADS})")
parser.add_argument("--input-dir", "-i", type=str, default=None,
                    help="Custom input directory containing .dcm files")
args, _ = parser.parse_known_args()

NUM_WORKERS = args.workers
THREADS_PER_WORKER = str(args.threads)

# Set thread caps BEFORE importing deep learning libraries
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
from engines import check_gpu_available, initialize_engines
from pipeline import anonymize_dicom_file
from de_identification.keystore import KeyStore

# Global variables within each worker process
_engines = None
_keystore = None


def init_worker(threads_per_worker):
    """
    Initializes all 5 AI/OCR models ONCE when each worker process starts.
    Subsequent files processed by this worker incur zero model load overhead.
    """
    global _engines, _keystore
    torch.set_num_threads(int(threads_per_worker))
    cv2.setNumThreads(int(threads_per_worker))

    use_gpu = check_gpu_available()
    _engines = initialize_engines(use_gpu=use_gpu)
    _keystore = KeyStore(SECURED_KEYSTORE_FILE)


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
    out_dir = os.path.join(OUTPUT_DIR, safe_stem)
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


def main():
    target_dir = args.input_dir or INPUT_DIR
    input_files = sorted(glob.glob(os.path.join(target_dir, "*.dcm")))
    if not input_files:
        input_files = sorted(glob.glob(os.path.join(target_dir, "**", "*.dcm"), recursive=True))
    if not input_files:
        print(f"No .dcm files found in {target_dir}/")
        return

    print(f"\n{'='*75}")
    print(f"PARALLEL DICOM DE-IDENTIFICATION PIPELINE")
    print(f"  Input Files       : {len(input_files)}")
    print(f"  Parallel Workers  : {NUM_WORKERS}")
    print(f"  Threads / Worker  : {THREADS_PER_WORKER}")
    print(f"  Target Output     : {OUTPUT_DIR}")
    print(f"{'='*75}\n")

    batch_start = time.time()

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=NUM_WORKERS, initializer=init_worker, initargs=(THREADS_PER_WORKER,)) as pool:
        results = list(pool.imap_unordered(process_single_file, input_files))

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
