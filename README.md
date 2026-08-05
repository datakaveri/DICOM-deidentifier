# SKALD-DICOM

De-identification pipeline for radiology DICOM files, built for the SPIDEr platform. Removes Protected Health Information (PHI) at two levels in a single pass:

1. **Burned-in pixel text** — OCR (EasyOCR) finds names, IDs, dates, phone numbers, and other PHI printed directly into the image, classifies it against a clinical allowlist and NLP (Presidio), and redacts it with structure-preserving inpainting (no quality loss to the underlying anatomy). PaddleOCR is supported as an optional second engine in `engines.py`/`ocr_detect.py` for local, non-containerized use, but is deliberately left out of the Docker image — running PaddlePaddle and PyTorch in the same process corrupts the heap once both load real model weights.
2. **DICOM tag values** — every header tag is passed through a per-tag technique (hash, tokenise, format-preserving encrypt, suppress, or retain) defined in [`app/de_identification/tag_mapping.py`](app/de_identification/tag_mapping.py), plus private/vendor tag stripping and UID regeneration.

The output is always a valid `.dcm` file — never a PNG/JPEG export — so downstream DICOM tooling keeps working.

## Project layout

```
app/
  main.py                  # debug entrypoint: pixel redaction only, single file (data/input.dcm)
  de_identification/run.py # main entrypoint: full pipeline, batches every *.dcm under DATA_DIR
  config.py                # paths (env-var overridable), PII patterns, clinical allowlist
  pipeline.py               # 7-stage pixel redaction pipeline
  de_identification/        # tag-level de-identification (hashing, tokenisation, FPE, keystore)
  config/                   # local dev mirror of the container's mounted config volume
data/                       # input DICOM files (local dev mirror of the container's mounted volume)
output/                     # results, audit logs, keystore (local dev mirror of the container's mounted volume)
tests/                      # pytest suite
Dockerfile
requirements.txt
```

## Running locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Drop one or more .dcm files into data/, then:
cd app
python -m de_identification.run
```

Results land under `output/<filename>/`: `data.json` (original tag snapshot), `phi_tags.json` (identified PHI tags), `before_deidentification.dcm` (pixel-redacted), `after_deidentification.dcm` (pixel-redacted + tag de-identified), `pipeline_audit.json`, and `tag_audit.json`. A run-level `output/manifest.json` summarizes every file processed. Tokenisation/encryption keys persist in `output/keystore/` across runs.

## Running with Docker

```bash
docker build -t skald-dicom .

docker run --rm \
  -v /path/to/input/dicoms:/data \
  -v /path/to/config:/app/config \
  -v /path/to/output:/output \
  skald-dicom
```

The container processes every `.dcm` file found under `/data` (recursively) and writes results to `/output`, in the same layout described above. OCR/NLP model weights are baked into the image at build time, so the container needs no outbound network access at runtime — this matters for air-gapped/TEE deployments.

Environment variables (already set in the image, override if needed):

| Variable | Default | Purpose |
|---|---|---|
| `SKALD_DATA_DIR` | `/data` | Input DICOM files (recursively scanned for `*.dcm`) |
| `SKALD_CONFIG_DIR` | `/app/config` | Reserved for future run-time config overrides |
| `SKALD_OUTPUT_DIR` | `/output` | Per-file results, audit logs, and the persistent keystore |

## Testing

```bash
pytest
```

## Policy notes

- Tag values are never modified by the pixel pipeline — only private/vendor tags are stripped and UIDs regenerated, so a file can't be linked back to the original study before tag-level de-identification runs.
- The tag-level de-identification technique per field is explicit and auditable in `tag_mapping.py` — nothing is inferred at runtime.
- `output/keystore/` (tokenisation/encryption key material) is never committed to git — it's runtime state, mounted or persisted via `/output` in the container.
