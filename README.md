# DICOM-deidentifier

Flask microservice that de-identifies PHI from DICOM tag metadata.
Part of the **SPIDEr** pipeline at IUDX / Data Kaveri.

> **Scope — tags only.**  
> Pixel-level redaction (burned-in text) is handled by a separate service
> that integrates with this one. See [Integration](#integration) below.

---

## What it does

For each uploaded DICOM file the service:

| Field type | Action |
|------------|--------|
| Patient name, address, phone, comments, physician names, institution | Masked → `*` |
| Patient ID, Accession number | SHA-256 pseudonymised (12-char hex) |
| Dates (birth, study, series, acquisition) | Generalised to year → `YYYY0101` |
| Study / Series / SOP Instance UIDs | Replaced with fresh generated UIDs |

Policy aligns with **DICOM PS3.15 Attribute Confidentiality Profiles**.

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/test_DICOM_deidentifier` | Health check |
| `POST` | `/inspect_DICOM` | List PHI tags found in the file |
| `POST` | `/process_DICOM` | De-identify tags; return cleaned DICOM |

### POST `/process_DICOM`

**Request** — `multipart/form-data`

| Field | Type | Description |
|-------|------|-------------|
| `file` | file | DICOM `.dcm` file |

**Response** — JSON

```json
{
  "status": "success",
  "fields_processed": 10,
  "audit": [
    {"field": "PatientName",    "old_value": "SMITH^JANE", "action": "masked"},
    {"field": "PatientID",      "old_value": "MRN789012",  "action": "hashed"},
    {"field": "PatientBirthDate","old_value": "19850322",  "action": "date_generalised"}
  ],
  "deidentified_dicom": "<base64-encoded .dcm bytes>"
}
```

### POST `/inspect_DICOM`

**Request** — `multipart/form-data` with `file`

**Response** — JSON

```json
{
  "status": "success",
  "count": 8,
  "phi_tags": [
    {"field": "PatientName",  "value": "SMITH^JANE"},
    {"field": "InstitutionName", "value": "CITY EYE CLINIC"}
  ]
}
```

---

## Run locally

```bash
git clone https://github.com/datakaveri/DICOM-deidentifier.git
cd DICOM-deidentifier
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python server.py
```

Test with curl:
```bash
curl -F "file=@sample_data/retinal_phi.dcm" http://localhost:5001/inspect_DICOM
curl -F "file=@sample_data/retinal_phi.dcm" http://localhost:5001/process_DICOM
```

---

## Run with Docker

```bash
docker build -t dicom-deidentifier .
docker run -p 5001:5001 dicom-deidentifier
```

Or with compose:
```bash
docker-compose up --build
```

---

## Run tests

```bash
pytest tests/ -v --cov=deidentifier
```

---

## Integration

### SPIDEr pipeline

This service handles **Step 1 — tag de-identification**.  
A second service handles **Step 2 — pixel de-identification** (burned-in text redaction).

Typical integration flow:
```
[DICOM input]
    │
    ▼
POST /process_DICOM          ← this service (tags)
    │  deidentified_dicom (base64)
    ▼
POST /process_DICOM_pixels   ← pixel de-identification service
    │  pixel-redacted DICOM
    ▼
[Clean DICOM output]
```

### Adding pixel de-identification

The pixel service should:
1. Accept a DICOM file (multipart or base64 JSON body)
2. OCR-detect and inpaint burned-in text
3. Return cleaned DICOM bytes

Suggested endpoint to add to this repo (or a separate service):
```
POST /process_DICOM_pixels
```

It can consume the `deidentified_dicom` base64 field from `/process_DICOM`
directly — no intermediate file needed.

---

## Project structure

```
DICOM-deidentifier/
├── deidentifier.py      # core de-identification logic (import this)
├── server.py            # Flask REST API
├── detect_pii.py        # optional: Presidio-based PII scanner
├── server_config.cfg    # host / port
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── sample_data/
│   ├── retinal_phi.dcm      # synthetic retinal fundus with PHI
│   └── synthetic_phi.dcm    # synthetic chest CT with PHI
└── tests/
    └── test_deidentifier.py
```
