"""
paddle_env.py — PaddlePaddle environment hardening for CPU-only VMs.

MUST be imported BEFORE any PaddlePaddle or PaddleOCR import.

Fixes:
  - MKLDNN / oneDNN segfaults on VMs without AVX2/AVX512.
  - Paddle's internal thread explosion that causes CPU thrashing
    when running under multiprocessing workers.
  - Verbose Paddle logging noise (FLAGS_log_level, GLOG).

Usage:
    import paddle_env          # <-- first line in any entry point
    from paddleocr import PaddleOCR   # now safe
"""

import os

# ── 1. Disable MKLDNN / oneDNN (fixes VM segfaults) ─────────────────────────
# These MUST be set before paddle is imported — once paddle reads them
# they are baked into the runtime and cannot be changed.
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["FLAGS_use_mkl_packed_mha"] = "0"

# ── 2. Cap Paddle's internal thread pool ─────────────────────────────────────
# Without these, each PaddleOCR call can spawn dozens of threads
# internally (via Eigen / PaddlePaddle threadpool), over-subscribing
# the CPU when running inside a multiprocessing worker.
_PADDLE_THREADS = os.environ.get("SKALD_PADDLE_THREADS", "2")
os.environ["FLAGS_num_threads"] = _PADDLE_THREADS
os.environ["FLAGS_inner_op_parallelism"] = _PADDLE_THREADS
os.environ.setdefault("CPU_NUM", _PADDLE_THREADS)

# ── 3. Cap all numerical library thread pools ────────────────────────────────
# These are belt-and-suspenders on top of what main_parallel.py sets,
# because config.py / engines.py may be imported before main_parallel.py
# in some code paths (e.g. tests, standalone runs).
_MATH_THREADS = os.environ.get("SKALD_THREADS_PER_WORKER", _PADDLE_THREADS)
for var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(var, _MATH_THREADS)

# ── 4. Silence Paddle / GLOG noise ──────────────────────────────────────────
os.environ.setdefault("GLOG_minloglevel", "2")          # suppress INFO/WARNING
os.environ.setdefault("FLAGS_log_level", "3")            # Paddle native log off
os.environ.setdefault("DISABLE_AUTO_LOGGING_CONFIG", "1")
os.environ.setdefault("PADDLE_LOG_LEVEL", "ERROR")

# ── 5. Disable Paddle's memory pre-allocation ───────────────────────────────
# In CPU mode, Paddle can pre-allocate a large memory arena. Disable it
# so multiple workers don't each grab gigabytes up front.
os.environ.setdefault("FLAGS_initial_cpu_memory_in_mb", "0")
os.environ.setdefault("FLAGS_fraction_of_cpu_memory_to_use", "0.1")
