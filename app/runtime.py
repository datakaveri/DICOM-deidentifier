"""
runtime.py — size the batch to the machine it actually runs on.

The pipeline is deployed into TEE VMs (confidential VMs) whose shape is not
known in advance, usually inside a container. This module works out the CPU
and memory budget the process really has, then picks how many worker
processes to run and how many math-library threads each one gets.

Sources, smallest wins:
  CPUs    os.sched_getaffinity (cpusets/taskset), then the cgroup v2 cpu.max
          or v1 cfs quota (docker --cpus, Kubernetes limits).
  Memory  /proc/meminfo MemAvailable, then the cgroup v2 memory.max or v1
          limit minus current usage (docker --memory).

Overrides (all optional):
  SKALD_WORKERS             fixed worker count (1 = serial, no subprocesses).
                            Auto: one worker per ~8 CPUs, capped by memory
                            and by the number of files.
  SKALD_THREADS_PER_WORKER  fixed threads per worker
  SKALD_MODEL_MEMORY_MB     RSS of the parent once every model is loaded
  SKALD_WORKER_MEMORY_MB    extra memory each forked worker needs
  SKALD_FILE_TIMEOUT_S      per-file limit before a worker counts as hung
                            (default 900); the batch then finishes serially

The memory defaults were measured on the CPU image (see docs/performance.md).
Workers are forked from the parent after the models load, so they share the
model weights copy-on-write. A worker only pays for its own activations and
image buffers, not another copy of the models.

This module imports nothing heavy. configure() must run before numpy, torch,
paddle or cv2 are imported, because those read the thread variables once,
when they load.
"""

import math
import os
from dataclasses import dataclass

# Measured on the CPU image (docs/performance.md): one process with every
# model loaded peaks at ~4.2 GB, and each forked worker adds ~1 GB. A
# single-process run loads GLiNER lazily and peaks at ~1.2 GB.
DEFAULT_MODEL_MEMORY_MB = 4500
DEFAULT_WORKER_MEMORY_MB = 1500
# One process scales well up to about this many threads; past it, extra
# cores are better spent on another worker. On 12 CPUs, 1 x 12 matched
# 3 x 4 in throughput, and the single process needs a quarter of the memory
# and no eager GLiNER load.
TARGET_THREADS_PER_WORKER = 8
# Kept free for the kernel, page cache and the parent's own spikes.
RESERVED_MEMORY_MB = 512

_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    # Paddle's own pools (read when paddle is imported).
    "FLAGS_num_threads",
    "FLAGS_inner_op_parallelism",
    "CPU_NUM",
)


def _read(path):
    try:
        with open(path, "r", encoding="ascii") as f:
            return f.read().strip()
    except OSError:
        return None


def _cgroup_dirs():
    """The process's own cgroup directory (v2) plus the mount root.

    Inside a container with a cgroup namespace the root is the container's
    cgroup; on a bare VM the process's cgroup is a subdirectory of it.
    """
    dirs = ["/sys/fs/cgroup"]
    for line in (_read("/proc/self/cgroup") or "").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3 and parts[0] == "0" and parts[2] not in ("", "/"):
            dirs.insert(0, "/sys/fs/cgroup" + parts[2])
    return dirs


def cpu_budget():
    """CPUs this process may actually use (at least 1)."""
    try:
        cpus = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpus = os.cpu_count() or 1

    for d in _cgroup_dirs():
        quota = _read(os.path.join(d, "cpu.max"))  # v2: "<quota|max> <period>"
        if quota:
            q, _, p = quota.partition(" ")
            if q != "max" and p:
                cpus = min(cpus, max(1, math.ceil(int(q) / int(p))))
            break
    else:
        q = _read("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")  # v1
        p = _read("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q and p and int(q) > 0 and int(p) > 0:
            cpus = min(cpus, max(1, math.ceil(int(q) / int(p))))
    return max(1, cpus)


def memory_budget_mb():
    """Memory still available to this process in MB, or None if unknown."""
    candidates = []

    for line in (_read("/proc/meminfo") or "").splitlines():
        if line.startswith("MemAvailable:"):
            candidates.append(int(line.split()[1]) // 1024)
            break

    for d in _cgroup_dirs():
        limit = _read(os.path.join(d, "memory.max"))  # v2
        if limit:
            if limit != "max":
                used = int(_read(os.path.join(d, "memory.current")) or 0)
                candidates.append((int(limit) - used) // (1024 * 1024))
            break
    else:
        limit = _read("/sys/fs/cgroup/memory/memory.limit_in_bytes")  # v1
        # v1 reports "no limit" as a huge page-aligned number.
        if limit and int(limit) < (1 << 60):
            used = int(_read("/sys/fs/cgroup/memory/memory.usage_in_bytes") or 0)
            candidates.append((int(limit) - used) // (1024 * 1024))

    return min(candidates) if candidates else None


def _env_int(name):
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise SystemExit(f"{name} must be an integer, got {raw!r}")
    if value < 1:
        raise SystemExit(f"{name} must be >= 1, got {value}")
    return value


@dataclass(frozen=True)
class Plan:
    cpus: int
    memory_mb: object  # int, or None when it could not be read
    workers: int
    threads: int
    reason: str

    def describe(self):
        mem = f"{self.memory_mb} MB" if self.memory_mb is not None else "unknown"
        return (
            f"{self.workers} worker(s) x {self.threads} thread(s) "
            f"[cpus={self.cpus}, available memory={mem}; {self.reason}]"
        )


def plan(n_files):
    """Pick workers and threads for a batch of n_files on this machine."""
    cpus = cpu_budget()
    memory_mb = memory_budget_mb()
    forced_workers = _env_int("SKALD_WORKERS")
    forced_threads = _env_int("SKALD_THREADS_PER_WORKER")
    model_mb = _env_int("SKALD_MODEL_MEMORY_MB") or DEFAULT_MODEL_MEMORY_MB
    worker_mb = _env_int("SKALD_WORKER_MEMORY_MB") or DEFAULT_WORKER_MEMORY_MB

    if forced_workers:
        workers, reason = forced_workers, "SKALD_WORKERS"
    else:
        by_cpu = max(1, cpus // TARGET_THREADS_PER_WORKER)
        if memory_mb is None:
            # Unknown memory: one process is the only safe choice.
            by_mem, mem_note = 1, "memory unknown"
        else:
            spare = memory_mb - model_mb - RESERVED_MEMORY_MB
            by_mem = max(1, spare // worker_mb) if spare > 0 else 1
            mem_note = f"memory allows {by_mem}"
        workers = max(1, min(by_cpu, by_mem, max(1, n_files)))
        reason = f"auto: cpu allows {by_cpu}, {mem_note}, {n_files} file(s)"

    threads = forced_threads or max(1, cpus // workers)
    return Plan(cpus, memory_mb, workers, threads, reason)


def apply_thread_env(threads):
    """Cap every math-library thread pool. Call before importing numpy/torch/paddle/cv2."""
    for var in _THREAD_VARS:
        os.environ[var] = str(threads)
    # Rust tokenizers would otherwise start their own pool, and warn (then
    # disable it) after every fork.
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def apply_thread_limits(threads):
    """Cap pools that ignore the environment. Call after the heavy imports,
    and again in each worker process."""
    try:
        import torch

        torch.set_num_threads(threads)
    except Exception:
        pass
    try:
        import torch

        # Only settable before the first parallel op; later calls raise.
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(threads)
    except Exception:
        pass


def file_timeout_s():
    """Seconds to wait for any one file before treating a worker as hung."""
    return _env_int("SKALD_FILE_TIMEOUT_S") or 900


def configure(n_files):
    """Plan the batch and apply its thread caps. Returns the Plan.

    With several workers the environment caps are 1: the parent only loads
    models, and an OpenMP pool started before fork() deadlocks the children
    (see main.py). Workers raise their own limits after the fork.
    """
    p = plan(n_files)
    apply_thread_env(1 if p.workers > 1 else p.threads)
    return p
