# Performance

The pipeline is deployed into TEE (confidential) VMs whose CPU count and
memory are not known in advance. It sizes itself at start-up and prints the
plan:

```
Execution plan: 1 worker(s) x 12 thread(s) [cpus=12, available memory=7150 MB; auto: ...]
```

## How a run is sized (`app/runtime.py`)

| Input | Source (smallest wins) |
|---|---|
| CPUs | `sched_getaffinity`, cgroup v2 `cpu.max` or v1 CFS quota (`docker --cpus`, Kubernetes limits) |
| Memory | `/proc/meminfo` MemAvailable, cgroup v2 `memory.max` or v1 limit minus usage (`docker --memory`) |

- **Workers:** `min(CPUs ÷ 8, memory allows, number of files)`.
  - "Memory allows" is `(available − 4500 MB models − 512 MB reserve) ÷ 1500 MB per worker`.
  - If memory cannot be read, the run uses one process.
- **Threads per worker:** CPUs ÷ workers.
- **Overrides:** `SKALD_WORKERS`, `SKALD_THREADS_PER_WORKER`,
  `SKALD_MODEL_MEMORY_MB`, `SKALD_WORKER_MEMORY_MB` and `SKALD_FILE_TIMEOUT_S`.
  The README lists their defaults.

### Single process (the common case)

- Every model runs in one process, with all available threads.
- GLiNER, the largest model, loads only when a string needs it.

### Several workers (large machines)

- **How workers start:** the parent loads every model, then forks the
  workers, which share the weights copy-on-write. Each extra worker costs
  about 1 GB, not another 4 GB.
- **Parent stays single-threaded:** the parent loads the models with one
  thread. GNU OpenMP is not fork-safe: a thread pool started before `fork()`
  makes the children hang on their first torch operation.
- **Tag stage:** tag de-identification runs in the parent, one file at a time,
  in sorted input order. That keeps token numbers and per-column keys
  identical to a serial run.
- **Fallback:** if the pool cannot start, a worker is OOM-killed (workers set
  `oom_score_adj` so the kernel picks them before the parent), or a file
  exceeds `SKALD_FILE_TIMEOUT_S`, the parent kills the workers and finishes the
  remaining files itself.

## What changed and why (measured 2026-11-17, 12-CPU host, CPU image)

**Profile before the change:** one process, files one after another.

| Stage | Share of time |
|---|---|
| GLiNER | ~80% (~1.4 s per OCR string) |
| PaddleOCR | ~7% |
| All else | ~13% |

`classify_phi` sent every OCR string through four NER models. GLiNER and the
biomedical model only feed the "medical" signal, and that signal only matters
when no stronger PHI signal has fired. The classifier now evaluates the
signals in cost order and stops once the decision is fixed. The decision
matrix is unchanged; the `_decide` docstring in `classify.py` sets it out.

**Results on the 6-file sample set:**

| Measure | Before | After |
|---|---|---|
| Classification per file | 16–50 s | 0.0–0.4 s |
| Whole batch, including model loading | 236 s | 20 s |
| Per-file processing | ~31 s | ~1.4 s |
| Peak memory, single process | 4.2 GB | 1.2 GB |

On these samples GLiNER was never needed, so it never loaded.

**24-file batch** (the 6 samples × 4, with a 7 GB container limit):

| Plan | Batch time | Peak memory |
|---|---|---|
| 1 × 12 threads | 28.7 s | 1.2 GB |
| 3 × 4 threads | 28.6 s (+11 s start-up for eager GLiNER) | 7.1 GB |
| 1 × 4 threads (`--cpus 4`) | 37.8 s | 1.9 GB |

One process using all threads matched three workers at 12 CPUs, with a
quarter of the memory. That is why auto mode adds a worker only per ~8 CPUs.

### Equivalence checks

- **Classifier:** old and new `classify_phi` ran against the real models on
  166 detections: every OCR string from the samples, plus 45 hand-picked
  dates, IDs, names, markers and exposure values. They ran in both full and
  verify (Presidio-only) modes, with 0 differences.
- **Whole pipeline:** both versions ran with the same key store.
  - Pixel data, redacted regions and verification status were identical.
  - Every tag that is deterministic between two runs of the old version was
    identical too.
  - The token vault and hash keys were identical.
- **Parallel vs serial:** both ran with fresh key stores. Token vaults were
  identical, so tokens were minted in the same order. All 24 outputs had the
  same pixels and regions.
- **Failure:** 3 workers in a 4.5 GB container. The kernel killed a worker,
  and the batch finished serially, with 24/24 files PASSED.

## Not done, and why

- **Region-only OCR** (`run_on_regions=True`, as tried on the old
  `parallel-pipeline-final` branch). It misses text the shape detector misses,
  and OCR is no longer the bottleneck it was assumed to be.
- **Paddle MKL-DNN.** No gain on this CPU (6.3 s vs 5.8 s for 6 frames), and
  it was disabled originally because it crashed on some VMs.
- **Dropping or downscaling models, or skipping verification.** Each would
  change what gets redacted.
- **Further speed-ups**, ordered by expected gain:
  1. Run PaddleOCR text recognition through ONNX Runtime. Recognition is
     now the largest remaining cost, ~1 s per frame.
  2. The PaddleOCR 3 migration.
  3. Pipelining file reads.

  Each needs the same before/after equivalence check.
