# System Architecture — DICOM De-Identification Pipeline

## Overview
The **DICOM De-Identification Pipeline** processes medical radiology DICOM (`.dcm`) images to strip Protected Health Information (PHI) at two decoupled layers in a single unified execution pass:
1. **DICOM Header Metadata De-Identification**: Header attribute tokenization, cryptographic hashing, format-preserving encryption (FPE), date-shifting, and suppression compliant with DICOM PS 3.15 Profile.
2. **Burned-In Pixel Text Redaction**: OCR and neural entity recognition to detect burned-in text on image planes, stroke segmentation, and Navier-Stokes neighbor reconstruction to preserve underlying anatomical structures.

```
                          ┌───────────────────────────┐
                          │    Input DICOM (.dcm)     │
                          └─────────────┬─────────────┘
                                        │
                         ┌──────────────┴──────────────┐
                         ▼                             ▼
              [ Header Metadata ]              [ Pixel Data Array ]
                         │                             │
        ┌────────────────────────────┐    ┌────────────────────────────┐
        │ PS 3.15 Tag De-ID Engine   │    │ Stage 1-2: Text Detection  │
        │ - SHA-256 / Tokenisation   │    │ (PaddleOCR + Shape Gate)   │
        │ - FPE (AES-CBC Format)     │    └────────────┬───────────────┘
        │ - UID Regeneration        │                 │
        │ - Keystore (secured.json)  │    ┌────────────▼───────────────┐
        └─────────────┬──────────────┘    │ Stage 3-4: Classification  │
                      │                   │ (Presidio + GLiNER + NLP   │
                      │                   │  vs Clinical Allowlist)    │
                      │                   └────────────┬───────────────┘
                      │                                │
                      │                   ┌────────────▼───────────────┐
                      │                   │ Stage 5: Inpainting        │
                      │                   │ (Stroke / Shadow Masking + │
                      │                   │  Navier-Stokes Inpaint)    │
                      │                   └────────────┬───────────────┘
                      │                                │
                      │                   ┌────────────▼───────────────┐
                      │                   │ Stage 6: Two-Tier Verify   │
                      │                   │ (Tier 1: Stroke check      │
                      │                   │  Tier 2: Fast OCR Gate)    │
                      │                   └────────────┬───────────────┘
                      │                                │
                      └──────────────┬─────────────────┘
                                     ▼
                      ┌───────────────────────────────┐
                      │ Stage 7: Pixel Write-Back     │
                      │ Output: after_deidentification│
                      └───────────────────────────────┘
```

## Pipeline Components

### 1. Optical Character Recognition (OCR) Engine
- **Primary Engine**: PaddleOCR (PP-OCRv4) optimized for Latin text and numerals.
- Runs on region-of-interest proposals to maximize throughput.

### 2. Multi-Model Entity Classification Ensemble
- **Clinical Allowlist**: Filters anatomy markers (`AP`, `LAT`, `PA VIEW`, `CHEST`), preserving diagnostic indicators.
- **PII Pattern Regex**: Catches structured identifiers (Aadhaar, ABHA, UHID/MRN, phone numbers, accession numbers).
- **Stanford De-ID & Biomedical NER**: Transformer-based token classification for clinician names and locations.
- **GLiNER**: Zero-shot named entity recognition for contextual medical phrases.

### 3. Stroke Segmentation & Navier-Stokes Inpainting
- Rather than drawing opaque solid black boxes over pixels (which distorts windowing and destroys bone margins), the pipeline computes morphological top-hat masks around text strokes and fills using Navier-Stokes image neighbor reconstruction.

### 4. Two-Tier Verification Gate
- **Tier 1**: Sub-millisecond connected component stroke density evaluation.
- **Tier 2**: Fallback OCR scan over redacted bounding boxes.

### 5. Keystore & Deterministic Mapping
- Cryptographic keys and token tables are maintained in `secured.json`.
- A patient ID across multiple series maps deterministically to the same pseudonymized identifier across the study batch.
