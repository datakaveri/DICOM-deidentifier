# DICOM De-Identification Pipeline

De-identification pipeline for radiology DICOM files. Removes Protected Health Information (PHI) at two levels in a single unified pass:

1. **Burned-in Pixel Text Redaction** — OCR (**PaddleOCR**) detects burned-in text. An ensemble classification stack (**Stanford De-ID**, **Biomedical NER**, **GLiNER-BioMed**, **Microsoft Presidio**, and clinical allowlists) distinguishes patient PHI from medical findings and anatomy markers. Redaction uses **character stroke & drop-shadow segmentation** combined with **Navier-Stokes neighbor inpainting**, eliminating dark silhouettes while preserving underlying bone and tissue textures. A two-tier verification gate ensures complete anonymization with near-zero latency overhead.
2. **DICOM Tag De-Identification** — Every header tag is processed through a per-tag technique (cryptographic hash, tokenisation, format-preserving encryption, date masking, suppression, or retention) defined in [`app/de_identification/tag_mapping.py`](app/de_identification/tag_mapping.py), alongside private/vendor tag stripping and UID regeneration.

The output is always a valid `.dcm` file (with optional high-resolution `.png` visual preview for audits).

---

## Project Layout

```
├── Dockerfile                  # Multi-stage, non-root, offline-ready production image
├── pyproject.toml              # Version and dependency ranges (the source of truth)
├── requirements.lock           # Hashed, fully pinned runtime lock (what the image installs)
├── requirements-dev.lock       # Runtime lock + test/lint/audit tooling
├── requirements.in             # Lock inputs pyproject.toml cannot express (spaCy model)
├── scripts/lock.sh             # Regenerates both locks in the pinned base image
├── app/
│   ├── main.py                 # Main batch pipeline entry point
│   ├── config.py               # Paths, environment variables, clinical allowlist, PII regex
│   ├── engines.py              # PaddleOCR, Presidio, GLiNER, Stanford De-ID model initializers
│   ├── pipeline.py             # 7-stage anonymization orchestrator
│   ├── ocr_detect.py           # PaddleOCR full-image and region-based detection
│   ├── classify.py             # Multi-model PHI vs clinical classification matrix
│   ├── masking.py              # Character stroke & drop-shadow isolation + Navier-Stokes inpainting
│   ├── verify.py               # Two-tier verification (Tier 1: Stroke check <1ms, Tier 2: OCR fallback)
│   ├── dicom_io.py             # Pixel write-back and DICOM descriptor header updates
│   ├── phi_tags.py             # PHI tag scanner and burned-in tag value cross-checker
│   ├── bbox_visualize.py       # Visual bbox preview overlay generator
│   └── de_identification/      # Header tag de-identification engine & keystore
├── data/                       # Input DICOM files (*.dcm)
└── output/                     # Anonymized DICOMs, previews, and JSON audit logs
```

---

## Running Locally

```bash
# 1. Create a Python 3.12 virtual environment
python3.12 -m venv venv && source venv/bin/activate

# 2. Install the locked dependencies (includes the spaCy model and CPU torch)
pip install --require-hashes --no-deps -r requirements-dev.lock

# 3. Run the batch pipeline (keys go to SKALD_KEYS_DIR, default: the CWD).
#    LD_PRELOAD works around paddlepaddle 2.6's bundled zlib, which otherwise
#    segfaults on `import paddleocr`; the image sets it too (see Dockerfile).
cd app
LD_PRELOAD=/lib/x86_64-linux-gnu/libz.so.1 SKALD_KEYS_DIR=../keys python main.py
```

### Running Tests

```bash
ruff check . && black --check .
pytest -q
```

### Outputs Generated (per DICOM file)
Under `<output>/<sample_name>/`:
- `after_deidentification.dcm` — Final anonymized DICOM (pixel-redacted + tag de-identified).
- `after_preview.png` — Preview of the redacted pixels.
- `pipeline_audit.json` — Redacted region boxes (no recognised text), verification status, timing, errors.
- `tag_audit.json` — Which technique was applied to which tag (no values).
- `before_deidentification.dcm`, `data.json`, `phi_tags.json` — Still hold the original values. Without a job config they sit beside the output; with one they go to a scratch directory under `SKALD_TEMP_DIR` that is deleted at the end of the batch.
- `bbox_regions.png` — Only with `SKALD_SAVE_BBOX_PREVIEW=1`: the frame before redaction with PHI boxes drawn on it, so it shows the PHI.

`<output>/manifest.json` summarises the run per file (status, region count, `tags_by_technique`, error). The SPIDEr enclave attaches it to the job status.

### Job config (SPIDEr UI)

If `config.json` in the config directory has a `dicom_deidentify` block, its `tag_actions` decide what happens to each tag, as the user chose in the UI. The contract is in the UI repo (`docs/dicom-deidentify-config-contract.md`).
- **Actions:** `suppress` (blank, tag kept), `hashing_with_salt` (per-run salt, never stored), `encrypt`, `masking`, `tokenization` and `charcloak`.
- **Tags not listed** are kept unchanged. UIDs are still remapped and private tags are still removed.
- **Fixed policy:** burned-in text is filled black, verification is strict and can report `FAILED`, and an unparsable file is an error. Values sent for the fixed fields (`temp_dir`, `pixel`, `fail_on_unparsable`, ...) are logged and ignored.
- **Invalid config:** the run exits 1 before reading any file.

With no config file, or no `dicom_deidentify` block, the fixed policy in `app/de_identification/tag_mapping.py` runs. It removes or transforms every PHI tag, including unlisted date tags, and redacts with inpainting.

---

## Running with Docker

```bash
# Released images (see "Versioning and releases" below)
docker pull ghcr.io/datakaveri/skald-dicom:2.0.0

# Or build locally (model weights are baked in during the build)
docker build -t skald-dicom:dev .

# The process runs as uid 65532: output/ and keys/ must be writable by it
mkdir -p output keys && sudo chown 65532:65532 output keys

docker run --rm \
  -v /path/to/input/dicoms:/app/data:ro \
  -v /path/to/config:/app/config:ro \
  -v "$PWD/output":/app/output \
  -v "$PWD/keys":/app/keys \
  ghcr.io/datakaveri/skald-dicom:2.0.0
```

The container processes every `.dcm` file in `/app/data` and writes anonymized outputs and logs to `/app/output`. Key material (`secured.json`: hash keys and the reverse token vault) goes to `/app/keys`: **never ship `keys/` with a release**, and mount and reuse the same `keys/` across batches so pseudonyms stay consistent. Without a mount, the keys live and die with the container. Model weights are embedded at build time and the runtime refuses network model downloads, so the image runs fully offline.

If the input files and output directory belong to root and cannot be re-owned (the SPIDEr enclaves stage inputs as root, mode `0600`), run the container as root instead: `--user 0:0`, or `user: "0:0"` in compose.

**Compressed input.** JPEG Lossless (the usual PACS export), JPEG baseline/extended, JPEG-LS, JPEG 2000 and RLE are decoded on read with pylibjpeg. The redacted output is always uncompressed Explicit VR Little Endian (`1.2.840.10008.1.2.1`), so a compressed input grows by about 1.5×: an 8-bit 4248 × 3480 PACS X-ray went from 9.0 MB (JPEG Lossless) to 14.8 MB. Colour images (`SamplesPerPixel` > 1) are rejected; the redaction stages handle single-channel images only.

**Exit status.** `0` when every file was de-identified. `1` when any file failed (its error is printed, and nothing is left at its `after_deidentification.dcm`) or when `/app/data` has no `.dcm` files. Files that succeeded in a failed batch are still written.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `SKALD_DATA_DIR` | `/app/data` | Input directory containing DICOM files (`*.dcm`) |
| `SKALD_CONFIG_DIR` | `/app/config` | Configuration directory |
| `SKALD_OUTPUT_DIR` | `/app/output` | Destination for anonymized files and audit logs |
| `SKALD_KEYS_DIR` | `/app/keys` (image), CWD (local) | Where `secured.json` (keys + token vault) is read and written |
| `SKALD_WORKERS` | auto | Worker processes. Auto: one per ~8 CPUs, capped by free memory and file count; `1` runs everything in one process |
| `SKALD_THREADS_PER_WORKER` | auto | Math-library threads per worker (default: CPUs ÷ workers) |
| `SKALD_FILE_TIMEOUT_S` | `900` | Seconds one file may take before a worker counts as hung; the batch then finishes serially |
| `SKALD_MODEL_MEMORY_MB` / `SKALD_WORKER_MEMORY_MB` | `4500` / `1500` | Memory estimates used to size the worker pool |
| `SKALD_JOB_CONFIG` | `$SKALD_CONFIG_DIR/config.json` | The decrypted SPIDEr job config (see "Job config") |
| `SKALD_TEMP_DIR` | `/tmp/skald-dicom` | Scratch for still-identified intermediates and the run's keys under a job config; must be under `/tmp` |
| `SKALD_SAVE_BBOX_PREVIEW` | `0` | `1` writes `bbox_regions.png` (shows PHI; local tuning only) |

The image sizes itself to the machine it lands on (CPU affinity, cgroup CPU and memory limits, available memory), so the same image runs on an unknown TEE VM without tuning. See [`docs/performance.md`](docs/performance.md).

---

## Versioning and releases

The project follows [Semantic Versioning](https://semver.org/). The version lives in `pyproject.toml` and `app/__init__.py` (a unit test keeps them equal), and every change is recorded in [`CHANGELOG.md`](CHANGELOG.md).

A release is an immutable tag `vMAJOR.MINOR.PATCH` on `main`. Pushing it runs [`release.yml`](.github/workflows/release.yml), which refuses a tag that does not match the version or has no dated CHANGELOG section, then publishes `ghcr.io/datakaveri/skald-dicom:<version>` (also tagged `v<version>`) and a GitHub Release with that CHANGELOG section and the image digest. `:latest` moves to each new highest release, and deployments run `:latest`; a patch to an older line never moves it. The full procedure is in [`docs/releasing.md`](docs/releasing.md).
