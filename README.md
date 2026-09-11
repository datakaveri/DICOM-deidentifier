# DICOM De-Identification Pipeline

De-identification pipeline for radiology DICOM files. Removes Protected Health Information (PHI) at two levels in a single unified pass:

1. **Burned-in Pixel Text Redaction** — OCR (**PaddleOCR**) detects burned-in text. An ensemble classification stack (**Stanford De-ID**, **Biomedical NER**, **GLiNER-BioMed**, **Microsoft Presidio**, and clinical allowlists) distinguishes patient PHI from medical findings and anatomy markers. Redaction uses **character stroke & drop-shadow segmentation** combined with **Navier-Stokes neighbor inpainting**, eliminating dark silhouettes while preserving underlying bone and tissue textures. A two-tier verification gate ensures complete anonymization with near-zero latency overhead.
2. **DICOM Tag De-Identification** — Every header tag is processed through a per-tag technique (cryptographic hash, tokenisation, format-preserving encryption, date masking, suppression, or retention) defined in [`app/de_identification/tag_mapping.py`](app/de_identification/tag_mapping.py), alongside private/vendor tag stripping and UID regeneration.

The output is always a valid `.dcm` file (with optional high-resolution `.png` visual preview for audits).

---

## Project Layout

```
├── Dockerfile                  # Production container definition (air-gapped & offline-ready)
├── requirements.txt            # Pinned dependencies (PaddleOCR, PyTorch, Presidio, GLiNER)
├── secured.json                # Shared cryptographic key and token material
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
# 1. Activate virtual environment
python -m venv venv && source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Run the batch pipeline (Sequential)
cd app
python main.py

# 4. Run the batch pipeline in Parallel (High-Throughput on Multi-Core VM)
python main_parallel.py --workers 4 --threads 4
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
# Build the production image (model weights are pre-cached during build)
docker build -t dicom-deidentifier .

# Run the batch pipeline
docker run --rm \
  -v /path/to/input/dicoms:/app/data \
  -v /path/to/config:/app/config \
  -v /path/to/output:/app/output \
  dicom-deidentifier
```

The container processes every `.dcm` file located in `/app/data` and saves anonymized outputs and logs to `/app/output`. Model weights are embedded into the image during `docker build`, enabling fully offline, air-gapped deployment in secure clinical environments.

---

## Environment Variables

| Variable | Default | Purpose |
|---|---|---|
| `SKALD_DATA_DIR` | `/app/data` | Input directory containing DICOM files (`*.dcm`) |
| `SKALD_CONFIG_DIR` | `/app/config` | Configuration directory |
| `SKALD_OUTPUT_DIR` | `/app/output` | Destination for anonymized files, audit logs, and keys |
