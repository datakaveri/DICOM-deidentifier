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
Under `app/output/<sample_name>/`:
- `after_deidentification.dcm` — Final anonymized DICOM (pixel-redacted + tag de-identified).
- `after_preview.png` — Normalized visual inspection preview image.
- `before_deidentification.dcm` — Intermediate checkpoint (pixel-redacted, original tags).
- `bbox_regions.png` — Bounding boxes of detected PHI annotations.
- `pipeline_audit.json` — Detailed audit log (redacted regions, execution time, status).
- `data.json` & `phi_tags.json` — Pre-deidentification tag snapshots.

---

## Running with Docker

```bash
# Released images (see "Versioning and releases" below)
docker pull ghcr.io/datakaveri/skald-dicom:1.0.0

# Or build locally (model weights are baked in during the build)
docker build -t skald-dicom:dev .

# The process runs as uid 65532: output/ and keys/ must be writable by it
mkdir -p output keys && sudo chown 65532:65532 output keys

docker run --rm \
  -v /path/to/input/dicoms:/app/data:ro \
  -v /path/to/config:/app/config:ro \
  -v "$PWD/output":/app/output \
  -v "$PWD/keys":/app/keys \
  ghcr.io/datakaveri/skald-dicom:1.0.0
```

The container processes every `.dcm` file in `/app/data` and writes anonymized outputs and logs to `/app/output`. Key material (`secured.json`: hash keys and the reverse token vault) goes to `/app/keys`, a separate volume: **never ship `keys/` with a release**, and reuse the same `keys/` across batches so pseudonyms stay consistent. Model weights are embedded at build time and the runtime refuses network model downloads, so the image runs fully offline.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `SKALD_DATA_DIR` | `/app/data` | Input directory containing DICOM files (`*.dcm`) |
| `SKALD_CONFIG_DIR` | `/app/config` | Configuration directory |
| `SKALD_OUTPUT_DIR` | `/app/output` | Destination for anonymized files and audit logs |
| `SKALD_KEYS_DIR` | `/app/keys` (image), CWD (local) | Where `secured.json` (keys + token vault) is read and written |

---

## Versioning and releases

The project follows [Semantic Versioning](https://semver.org/). The version lives in `pyproject.toml` and `app/__init__.py` (a unit test keeps them equal), and every change is recorded in [`CHANGELOG.md`](CHANGELOG.md).

A release is an immutable tag `vMAJOR.MINOR.PATCH` on `main`. Pushing it runs [`release.yml`](.github/workflows/release.yml), which refuses a tag that does not match the version or has no dated CHANGELOG section, then publishes `ghcr.io/datakaveri/skald-dicom:<version>` (also tagged `v<version>`) and a GitHub Release with that CHANGELOG section and the image digest. `:latest` moves to each new highest release, and deployments run `:latest`; a patch to an older line never moves it. The full procedure is in [`docs/releasing.md`](docs/releasing.md).
