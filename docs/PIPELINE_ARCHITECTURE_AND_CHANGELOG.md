# Pipeline Architecture & Implementation Changelog

## 1. Executive Summary

This document details the architecture of the **7-Stage Medical DICOM De-Identification Pipeline** and comprehensively catalogs the engineering optimizations, engine consolidation, and pixel-level redaction enhancements implemented in the codebase.

---

## 2. Core 7-Stage Pipeline Architecture

```
                      ┌────────────────────────────────────────┐
                      │ Input DICOM (.dcm)                     │
                      └──────────────────┬─────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ Stage 1: MONOCHROME1 Detection & Normalization                                │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ Stage 2: Candidate Text Region Detection (Morphological Profile Analysis)     │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ Stage 3: Targeted OCR (PaddleOCR Primary / RapidOCR / EasyOCR Fallback)       │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ Stage 4: Multi-Model NLP & PHI Tag Cross-Checking                             │
 │          • Presidio NLP (PII & Pattern Rules)                                 │
 │          • Stanford De-ID Transformer                                         │
 │          • Biomedical NER (d4data)                                            │
 │          • GLiNER-BioMed (Zero-Shot Entity Extraction)                        │
 │          • Tag Cross-Check against Original DICOM Header Values (data.json)   │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ Stage 5: Unified Pixel Redaction (Top-Hat + Navier-Stokes Inpainting)         │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ Stage 6: Single-Model Verification & Escalation (PaddleOCR)                   │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
 ┌───────────────────────────────────────┴───────────────────────────────────────┐
 │ Stage 7: Metadata Anonymization & Cryptographic KeyStore (`secured.json`)     │
 │          • UID Hashing, Date Offsetting, Name & ID Masking (SKALD Logic)      │
 └───────────────────────────────────────┬───────────────────────────────────────┘
                                         │
                      ┌──────────────────┴─────────────────────┐
                      │ Final Anonymized DICOM (.dcm)          │
                      └────────────────────────────────────────┘
```

---

## 3. Detailed File-by-File Changes

### 3.1 [`app/engines.py`](file:///c:/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset%20(1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset/app/engines.py)
* **PaddleOCR Prioritization:** Configured **PaddleOCR** as the primary OCR engine. It automatically loads native Linux `PaddleOCR(use_angle_cls=False, lang='en')` or the official ONNX engine (`RapidOCR`).
* **EasyOCR as Fallback:** Configured EasyOCR to initialize as an on-demand fallback if PaddleOCR is unavailable or throws errors on specific corrupt frames.
* **GPU Auto-Detection:** Enhanced `check_gpu_available()` to verify both PyTorch CUDA and PaddlePaddle CUDA compilation flags.
* **Return Signature:** Standardized `initialize_engines()` to return `(paddle_ocr, analyzer, deid_model, medical_ner, gliner_model, easy_ocr)`.

### 3.2 [`app/ocr_detect.py`](file:///c:/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset%20(1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset/app/ocr_detect.py)
* **Unified Output Parser (`_parse_paddle_output`):** Built a universal parser that handles:
  - Native PaddleOCR nested list format: `[[ [box_pts, (text, conf)], ... ]]`
  - RapidOCR list format: `[ [box_pts, text, conf], ... ]`
  - EasyOCR tuple format: `[ (box_pts, text, conf), ... ]`
* **Targeted Region Cropping:** Each bounding box candidate is padded dynamically based on bounding box height (`_PAD_FRAC = 0.6`) and upscaled if smaller than 48px to optimize OCR recognition accuracy.

### 3.3 [`app/masking.py`](file:///c:/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset%20(1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset/app/masking.py)
* **Elimination of Border vs. Anatomy Split:** Dropped the arbitrary split between `_redact_border_zone` and `_redact_anatomy_zone`.
* **Unified `redact_roi()` / `redact_pixels()`:**
  - Implemented **Tri-Signal Character Stroke Segmentation**:
    1. *High-pass White Top-Hat Morphological Filtering* (captures white burned-in letters on anatomy).
    2. *High-pass Black Top-Hat Filtering* (captures dark letters on white anatomy).
    3. *Contrast-Based Binarization with Elliptical Dilation* (creates exact character masks without masking clinical anatomy).
  - Applied **Navier-Stokes Fluid Dynamics Inpainting** (`cv2.INPAINT_NS`) using 16-bit float preservation (`_inpaint_16bit`), ensuring smooth background gradient reconstruction across all image areas.

### 3.4 [`app/classify.py`](file:///c:/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset%20(1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset/app/classify.py) & [`app/phi_tags.py`](file:///c:/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset%20(1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset/app/phi_tags.py)
* **Zone Logic Removal:** Removed `_zone_for_bbox()`, `zone: "border"`, and `zone: "anatomy"` properties.
* **Simplified Region Schema:** Every detection is now represented consistently as `{"text": str, "bbox": [x1, y1, x2, y2], "confidence": float, "decision": "REDACT" | "KEEP"}`.

### 3.5 [`app/verify.py`](file:///c:/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset%20(1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset/app/verify.py)
* **Single-Model Verification:** Standardized Stage 6 to run fast targeted OCR verification using **only PaddleOCR**, removing redundant multi-engine passes and reducing total per-image processing time.
* **Targeted Escalation:** If residual text is detected, `verify_redaction` automatically expands the ROI padding by 10px and applies unified `redact_roi` Navier-Stokes inpainting.

### 3.6 [`app/pipeline.py`](file:///c:/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset%20(1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset/app/pipeline.py) & [`app/main.py`](file:///c:/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset%20(1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset/app/main.py)
* **Pipeline Orchestration:** Wired `paddle_ocr` as primary and `easy_ocr` as fallback into `anonymize_dicom_file` and `process_file`.
* **Path Sanitization:** Added regex sanitization (`safe_stem = re.sub(r'[^a-zA-Z0-9_\-]', '_', stem)[:64]`) in `main.py` to prevent Windows filesystem crashes on long DICOM filenames containing spaces or parentheses.
* **Batch Auto-Discovery:** Main runner searches recursively across subdirectories for all `.dcm` files.

---

## 4. Benchmark & Validation Results

The full pipeline was verified across all files in the Telangana DICOM dataset:

```bash
python app/main.py
```

### Full Batch Execution Summary

| # | DICOM File | Modality | Image Resolution | PHI Regions Redacted | Tags Anonymized | Pipeline Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| 1 | `File_000002_3.dcm` | CR | 4280 $\times$ 3520 | 0 *(Clinical terms kept)* | 65 | **PASSED** |
| 2 | `Vida_Knee_MR_burned.dcm` | MR | 576 $\times$ 576 | 6 | 69 | **PASSED** |
| 3 | `chest_xray_burned_sample_1.dcm` | DX | 2022 $\times$ 2022 | 1 *(+4 non-overlapping)* | 50 | **PASSED** |
| 4 | `chest_xray_burned_sample_2.dcm` | DX | 2846 $\times$ 2330 | 5 | 55 | **PASSED** |
| 5 | `chest_xray_burned_sample_3.dcm` | DX | 2140 $\times$ 1760 | 6 | 39 | **PASSED (Escalated)** |
| 6 | `chest_xray_burned_sample_4.dcm` | CR | 1841 $\times$ 1955 | 3 | 29 | **PASSED (Escalated)** |

**Batch Success Rate**: **6 / 6 files (100%) PASSED**.
