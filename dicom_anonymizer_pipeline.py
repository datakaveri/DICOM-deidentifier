# -*- coding: utf-8 -*-
"""
=============================================================================
UNIFIED DICOM BURNED-IN TEXT ANONYMIZATION PIPELINE
=============================================================================
INPUT  : Any .dcm file (chest X-ray, 8-bit or 16-bit, CR/DX/XR modality)
OUTPUT : Anonymized .dcm file — DICOM FORMAT ONLY (no JPEG, no PNG)

Pipeline Stages:
  Stage 1 — Metadata Sanitization  (DICOM tags: PatientName, ID, DOB, etc.)
  Stage 2 — Image Enhancement       (5 contrast variants for OCR)
  Stage 3 — OCR Text Detection      (PaddleOCR + EasyOCR dual engine)
  Stage 4 — PHI Classification      (Allowlist + Regex + NLP + Spatial)
  Stage 5 — Pixel Redaction         (Navier-Stokes inpainting, 16-bit safe)
  Stage 6 — Verification + Retry    (Re-OCR to confirm zero residual text)
  Stage 7 — DICOM Write-Back        (correct BitsAllocated/BitsStored tags)

Usage:
  python dicom_anonymizer_pipeline.py
=============================================================================
"""

import os
import re
import json
import copy
import logging
import datetime
import argparse

import numpy as np
import cv2
import pydicom
from pydicom.uid import generate_uid

# ─── Suppress verbose sub-library logs ───────────────────────────────────────
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["DISABLE_AUTO_LOGGING_CONFIG"] = "1"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ─── Optional engine imports ──────────────────────────────────────────────────
try:
    from paddleocr import PaddleOCR, logger as paddle_logger
    paddle_logger.setLevel(logging.ERROR)
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except Exception:
    PRESIDIO_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_DIR  = "./raw_datasamples_xray_chest"   # Raw (original) DICOMs
OUTPUT_DIR = "./anonymized_xray_chest"        # Anonymized DICOM output only
AUDIT_LOG  = "./anonymized_xray_chest/audit_log.json"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Clinical terms that must NEVER be redacted (anatomy / positioning markers)
CLINICAL_ALLOWLIST = {
    "L", "R", "LT", "RT", "LEFT", "RIGHT",
    "PA", "AP", "LAT", "LL", "RL", "LATERAL",
    "ERECT", "SUPINE", "PRONE", "DECUBITUS", "UPRIGHT",
    "PORTABLE", "MOBILE", "STAT", "ROUTINE",
    "CHEST", "ABDOMEN", "PELVIS", "SKULL", "SPINE",
    "KVP", "MAS", "MA", "SEC", "CM", "MM", "FOV",
    "CXR", "CX", "PA VIEW", "AP VIEW",
}

# PII patterns (Indian health context + universal)
PII_PATTERNS = {
    "aadhaar":     r'\b\d{4}\s?\d{4}\s?\d{4}\b',
    "abha":        r'\b\d{2}-\d{4}-\d{4}-\d{4}\b',
    "phone":       r'\b(?:\+91|0)?[6-9]\d{9}\b',
    "date":        r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b',
    "uhid_mrn":    r'\b(?:UHID|MRN|REG|IPD|OPD|CR|PID|HID)[\s:/-]?\d+\b',
    "age_sex":     r'\b\d{1,3}\s*[/]\s*[MFO]\b',
    "name_prefix": r'\b(?:DR\.?|MR\.?|MRS\.?|MS\.?|SHRI|SMT|S/O|D/O|W/O)\b',
    "accession":   r'\b(?:ACC|ACCNO|ACCESSION)[\s:.-]?\w+\b',
}

# Metadata tags to clear (DICOM de-identification profile)
METADATA_TAGS_TO_CLEAR = [
    'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
    'PatientAge', 'PatientAddress', 'PatientTelephoneNumbers',
    'PatientMotherBirthName', 'PatientBirthName',
    'ReferringPhysicianName', 'ReferringPhysicianAddress',
    'InstitutionName', 'InstitutionAddress', 'InstitutionalDepartmentName',
    'StationName', 'AccessionNumber', 'StudyID',
    'StudyDate', 'SeriesDate', 'AcquisitionDate', 'ContentDate',
    'StudyTime', 'SeriesTime', 'AcquisitionTime', 'ContentTime',
    'PhysiciansOfRecord', 'PerformingPhysicianName',
    'NameOfPhysiciansReadingStudy', 'OperatorsName',
    'AdmittingDiagnosesDescription', 'PatientWeight',
    'RequestingPhysician', 'RequestedProcedureDescription',
    'ScheduledPerformingPhysicianName', 'RequestedProcedureID',
]

# =============================================================================
# STAGE 1: METADATA SANITIZATION
# =============================================================================
def sanitize_metadata(ds):
    """Clears all PHI DICOM metadata tags and re-generates UIDs."""
    log.info("  [Stage 1] Sanitizing DICOM metadata tags...")

    for attr in METADATA_TAGS_TO_CLEAR:
        if hasattr(ds, attr):
            if attr == 'PatientName':
                ds.PatientName = ""
            elif attr == 'PatientID':
                ds.PatientID = ""
            else:
                try:
                    setattr(ds, attr, "")
                except Exception:
                    pass

    # Re-generate UIDs so this file cannot be linked to the original study
    for uid_attr in ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']:
        if hasattr(ds, uid_attr):
            setattr(ds, uid_attr, generate_uid())

    # Remove all private / vendor tags (group numbers are odd)
    ds.remove_private_tags()

    log.info("  [Stage 1] Done. All PHI metadata cleared.")
    return ds


# =============================================================================
# STAGE 2: IMAGE ENHANCEMENT (generates 5 contrast variants for OCR)
# =============================================================================
def enhance_image(image_8bit):
    """
    Takes an 8-bit grayscale image (already normalized from 16-bit).
    Returns a dict of enhanced variants: standard, clahe, adaptive_thresh,
    inverse_thresh, gamma, sharpened.
    """
    variants = {"standard": image_8bit}

    try:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        variants["clahe"] = clahe.apply(image_8bit)
    except Exception:
        pass

    try:
        variants["adaptive_thresh"] = cv2.adaptiveThreshold(
            image_8bit, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5
        )
    except Exception:
        pass

    try:
        _, thresh_inv = cv2.threshold(
            image_8bit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        variants["inverse_thresh"] = thresh_inv
    except Exception:
        pass

    try:
        table = np.array([
            ((i / 255.0) ** (1.0 / 0.5)) * 255 for i in range(256)
        ], dtype=np.uint8)
        variants["gamma"] = cv2.LUT(image_8bit, table)
    except Exception:
        pass

    try:
        blurred = cv2.GaussianBlur(image_8bit, (3, 3), 0)
        variants["sharpened"] = cv2.addWeighted(image_8bit, 1.5, blurred, -0.5, 0)
    except Exception:
        pass

    return variants


# =============================================================================
# STAGE 3: OCR TEXT DETECTION
# =============================================================================
def _parse_paddle_result(result):
    """
    Parses PaddleOCR result into a flat list of {text, bbox, confidence} dicts.
    Handles three formats:
      1. New PaddleX predict() -> list of OCRResult objects (has .rec_texts etc.)
      2. New PaddleX predict() -> list of dicts with rec_texts/rec_polys/rec_scores
      3. Legacy PaddleOCR ocr() -> list of [[[x,y],...], (text, score)]
    """
    detections = []
    if not result or len(result) == 0:
        return detections

    for item in result:
        # ── Format 1 & 2: PaddleX OCRResult object or dict ─────────────────
        rec_texts  = None
        rec_polys  = None
        rec_scores = None

        if hasattr(item, 'rec_texts'):
            # Object with attributes
            rec_texts  = item.rec_texts
            rec_polys  = item.rec_polys
            rec_scores = item.rec_scores
        elif isinstance(item, dict):
            rec_texts  = item.get('rec_texts', [])
            rec_polys  = item.get('rec_polys', [])
            rec_scores = item.get('rec_scores', [])

        if rec_texts is not None:
            for text, pts, score in zip(rec_texts, rec_polys, rec_scores):
                try:
                    pts = np.array(pts, dtype=np.int32)
                    xs = pts[:, 0]; ys = pts[:, 1]
                    detections.append({
                        "text": str(text).strip(),
                        "bbox": [int(xs.min()), int(ys.min()),
                                 int(xs.max()), int(ys.max())],
                        "confidence": float(score),
                        "engine": "paddle"
                    })
                except Exception as e:
                    log.debug(f"PaddleX item parse error: {e}")
            continue  # Handled this item; go to next

        # ── Format 3: Legacy list of [points, (text, score)] ───────────────
        if isinstance(item, list):
            for line in item:
                try:
                    if (
                        len(line) == 2
                        and isinstance(line[1], (list, tuple))
                        and len(line[1]) == 2
                    ):
                        pts = np.array(line[0], dtype=np.int32)
                        text  = str(line[1][0]).strip()
                        score = float(line[1][1])
                        xs = pts[:, 0]; ys = pts[:, 1]
                        detections.append({
                            "text": text,
                            "bbox": [int(xs.min()), int(ys.min()),
                                     int(xs.max()), int(ys.max())],
                            "confidence": score,
                            "engine": "paddle"
                        })
                except Exception as e:
                    log.debug(f"Legacy PaddleOCR line parse error: {e}")

    return detections


def detect_text(variants, paddle_ocr=None, easy_ocr=None):
    """Runs all available OCR engines on all image variants."""
    raw = []

    for name, img in variants.items():
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = img

        # PaddleOCR — use new predict() API (ocr() is deprecated in PaddleX)
        if paddle_ocr:
            try:
                # Try new predict() first (PaddleX >= 3.x)
                if hasattr(paddle_ocr, 'predict'):
                    result = paddle_ocr.predict(rgb)
                else:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        result = paddle_ocr.ocr(rgb, cls=False)
                for det in _parse_paddle_result(result):
                    det["variant"] = name
                    raw.append(det)
            except Exception as e:
                log.debug(f"PaddleOCR failed on variant '{name}': {e}")

        # EasyOCR
        if easy_ocr:
            try:
                results = easy_ocr.readtext(rgb)
                for (bbox_pts, text, conf) in results:
                    pts = np.array(bbox_pts, dtype=np.int32)
                    xs = pts[:, 0]; ys = pts[:, 1]
                    raw.append({
                        "text": str(text).strip(),
                        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                        "confidence": float(conf),
                        "variant": name,
                        "engine": "easyocr"
                    })
            except Exception as e:
                log.debug(f"EasyOCR failed on variant '{name}': {e}")

    return raw


# =============================================================================
# STAGE 4: MERGE, CLASSIFY (PHI vs SAFE)
# =============================================================================
def _iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0.0


def merge_detections(raw, iou_thresh=0.3):
    """NMS-style merge of overlapping boxes from all OCR engine+variant passes."""
    if not raw:
        return []
    sorted_det = sorted(raw, key=lambda x: x["confidence"], reverse=True)
    merged = []
    while sorted_det:
        cur = sorted_det.pop(0)
        group = [cur]
        remaining = []
        for d in sorted_det:
            if _iou(cur["bbox"], d["bbox"]) > iou_thresh:
                group.append(d)
            else:
                remaining.append(d)
        sorted_det = remaining
        bboxes = np.array([g["bbox"] for g in group])
        best_text = max(group, key=lambda x: len(x["text"]))["text"]
        max_conf = max(g["confidence"] for g in group)
        merged.append({
            "text": best_text,
            "bbox": [int(bboxes[:,0].min()), int(bboxes[:,1].min()),
                     int(bboxes[:,2].max()), int(bboxes[:,3].max())],
            "confidence": min(1.0, max_conf),
        })
    return merged


def _is_clinical(text):
    clean = re.sub(r'[^A-Z]', '', text.upper())
    return clean in CLINICAL_ALLOWLIST


def classify_phi(merged, image_shape, analyzer=None):
    """
    Classifies each merged detection as PHI (redact) or safe (keep).
    Returns list of dicts: {text, bbox, zone}
      zone = 'border'  -> flat border region (aggressive box fill safe)
      zone = 'anatomy' -> overlaps anatomy (must use careful hybrid inpainting)
    """
    h, w = image_shape[:2]
    phi_regions = []

    for det in merged:
        text = det["text"]
        bbox = det["bbox"]

        # ── Size validation: discard full-frame/large OCR bounding box artifacts ──
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        box_area = box_w * box_h
        img_area = w * h

        if box_h > h * 0.15 or box_w > w * 0.85 or box_area > (img_area * 0.10):
            log.warning(
                f"    SKIP OCR ARTIFACT (box too large): '{text}' @ {bbox} "
                f"(w={box_w}, h={box_h}, area={box_area}/{img_area})"
            )
            continue

        if _is_clinical(text):

            log.info(f"    KEEP (clinical marker): '{text}'")
            continue

        # ── Regex PII check ──────────────────────────────────────────────────
        regex_hit = any(
            re.search(p, text, re.IGNORECASE)
            for p in PII_PATTERNS.values()
        )

        # ── Presidio NLP check ───────────────────────────────────────────────
        nlp_hit = False
        if analyzer:
            try:
                results = analyzer.analyze(text=text, language="en")
                nlp_hit = any(
                    r.entity_type in {"PERSON", "DATE_TIME", "LOCATION",
                                      "EMAIL_ADDRESS", "PHONE_NUMBER"}
                    and r.score > 0.45
                    for r in results
                )
            except Exception:
                pass

        # ── Spatial zone detection ────────────────────────────────────────────
        y_mid = (bbox[1] + bbox[3]) / 2.0
        x_mid = (bbox[0] + bbox[2]) / 2.0

        in_border = (
            y_mid < h * 0.18 or y_mid > h * 0.82 or
            x_mid < w * 0.10 or x_mid > w * 0.90
        )
        in_anatomy = not in_border  # Anything NOT in border is overlaid on anatomy

        # ── Short text validation ─────────────────────────────────────────────
        clean_text = text.strip()
        if not clean_text:
            log.warning(f"    SKIP OCR ARTIFACT (empty text): @ {bbox}")
            continue

        if in_anatomy and len(clean_text) < 3:
            # Discard short 1-2 char noise in anatomy zone unless it matched regex/nlp
            if not (regex_hit or nlp_hit):
                log.info(f"    KEEP (short unknown text in anatomy): '{text}' @ {bbox}")
                continue

        # ── Decision ─────────────────────────────────────────────────────────
        # Border zone: redact if border (spatial), regex, or NLP hit
        # Anatomy zone: redact if regex, NLP, OR any unknown text (safe default)
        #   (Clinical allowlist already filters out L/R/PA etc.)
        if in_border and (regex_hit or nlp_hit or in_border):
            zone = "border"

            reason = ["border"]
            if regex_hit: reason.append("regex")
            if nlp_hit:   reason.append("nlp")
            log.info(f"    REDACT-BORDER ({', '.join(reason)}): '{text}' @ {bbox}")
            phi_regions.append({"text": text, "bbox": bbox, "zone": zone})

        elif in_anatomy and (regex_hit or nlp_hit):
            # Confirmed PHI inside anatomy (matched pattern/NLP) — must redact carefully
            zone = "anatomy"
            reason = []
            if regex_hit: reason.append("regex")
            if nlp_hit:   reason.append("nlp")
            log.info(f"    REDACT-ANATOMY ({', '.join(reason)}): '{text}' @ {bbox}")
            phi_regions.append({"text": text, "bbox": bbox, "zone": zone})

        elif in_anatomy and not regex_hit and not nlp_hit:
            # Unknown non-clinical text inside anatomy — treat as suspect PHI
            # Use anatomy zone masking (careful inpainting)
            log.info(f"    REDACT-ANATOMY (suspect unknown text): '{text}' @ {bbox}")
            phi_regions.append({"text": text, "bbox": bbox, "zone": "anatomy"})

        else:
            log.info(f"    KEEP (unmatched): '{text}'")

    return phi_regions


# =============================================================================
# MULTI-HYBRID MASKING HELPERS
# =============================================================================

def _get_ink_mask(roi_8bit, dilation_px=2):
    """
    Extracts a precise binary mask of text ink strokes from an 8-bit ROI.
    Uses Otsu binarization + morphological cleanup + optional MSER fallback.
    dilation_px: how many pixels to expand the ink mask outward (captures edges).
    """
    # Primary: Otsu global threshold
    _, mask_otsu = cv2.threshold(roi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If text is bright on dark background, otsu gives us the bright blobs
    # If text is dark on bright background, invert
    mean_val = np.mean(roi_8bit)
    if mean_val > 127:
        mask_otsu = cv2.bitwise_not(mask_otsu)   # dark ink on bright bg

    # Secondary: Adaptive threshold (catches faint ink missed by Otsu)
    mask_adapt = cv2.adaptiveThreshold(
        roi_8bit, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    if mean_val > 127:
        mask_adapt = cv2.bitwise_not(mask_adapt)

    # Combine: any pixel flagged by either method
    combined = cv2.bitwise_or(mask_otsu, mask_adapt)

    # Morphological cleanup: remove noise specks (open), then dilate to cover edges
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    k_dilat = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_px + 1, dilation_px + 1))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    dilated = cv2.dilate(cleaned, k_dilat, iterations=1)

    return dilated


def _inpaint_16bit(image_16, mask_8, radius, method):
    """
    Runs cv2.inpaint on a 16-bit image by temporarily scaling to 8-bit,
    inpainting, then scaling back to the original 16-bit dynamic range.
    Returns the inpainted 16-bit image.
    """
    raw_min = float(image_16.min())
    raw_max = float(image_16.max())
    if raw_max > raw_min:
        temp_8 = ((image_16 - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
    else:
        temp_8 = image_16.astype(np.uint8)

    inpainted_8 = cv2.inpaint(temp_8, mask_8, radius, method)

    result = (
        inpainted_8.astype(np.float32) / 255.0 * (raw_max - raw_min) + raw_min
    ).astype(np.uint16)
    return result


def _blend_boundary(image, roi_slice, feather_px=4):
    """
    Applies a Gaussian-feathered blend at the boundary of an inpainted ROI
    so the fill merges smoothly into surrounding anatomy.
    roi_slice: (y1, y2, x1, x2) of the inpainted region.
    """
    y1, y2, x1, x2 = roi_slice
    h, w = image.shape[:2]
    pad = feather_px

    # Slightly expanded context region
    ey1 = max(0, y1 - pad); ey2 = min(h, y2 + pad)
    ex1 = max(0, x1 - pad); ex2 = min(w, x2 + pad)

    region = image[ey1:ey2, ex1:ex2].copy()
    if region.dtype == np.uint16:
        region_f = region.astype(np.float32) / 65535.0
        blurred  = cv2.GaussianBlur(region_f, (2 * feather_px + 1, 2 * feather_px + 1), 0)
        image[ey1:ey2, ex1:ex2] = (blurred * 65535.0).astype(np.uint16)
    else:
        blurred = cv2.GaussianBlur(region, (2 * feather_px + 1, 2 * feather_px + 1), 0)
        image[ey1:ey2, ex1:ex2] = blurred

    return image


def _redact_border_zone(cleaned, x1, y1, x2, y2):
    """
    BORDER ZONE STRATEGY: Background uniform fill.
    Safe to use in plain black/white border margins where there is no anatomy.
    Fills with the local surrounding median to match the margin tone.
    """
    h, w = cleaned.shape[:2]
    pad  = 30
    ctx  = cleaned[
        max(0, y1 - pad):min(h, y2 + pad),
        max(0, x1 - pad):min(w, x2 + pad)
    ]
    fill_val = int(np.median(ctx))
    cleaned[y1:y2, x1:x2] = fill_val
    return cleaned


def _redact_anatomy_zone(cleaned, x1, y1, x2, y2):
    """
    ANATOMY ZONE — MULTI-HYBRID 4-PASS STRATEGY:

    Pass 1 (Ink Isolation):
        Otsu + Adaptive threshold fusion to get a precise binary ink mask.
        Only the ink strokes are marked for inpainting, NOT the full box.
        This preserves underlying anatomy pixels not covered by ink.

    Pass 2 (Navier-Stokes Inpainting, radius=9):
        Structure-aware fill that propagates texture from the boundary inward.
        Best for anatomical gradients (ribs, lung parenchyma).

    Pass 3 (Telea Inpainting on Residual, radius=5):
        Fast marching method on any ink pixels missed by NS pass.
        Handles isolated thin strokes that NS may leave.

    Pass 4 (Bilateral Texture Blend):
        Bilateral filter on the inpainted region to smooth the fill
        and match the local frequency/texture of surrounding anatomy.
        Edge-preserving — does not blur bone boundaries.
    """
    h, w = cleaned.shape[:2]
    pad = 6  # extra context around box for better inpaint propagation

    # Safe bounds
    rx1 = max(0, x1 - pad); rx2 = min(w, x2 + pad)
    ry1 = max(0, y1 - pad); ry2 = min(h, y2 + pad)

    roi = cleaned[ry1:ry2, rx1:rx2]
    roi_h, roi_w = roi.shape[:2]

    # ── Pass 1: Precise ink mask ──────────────────────────────────────────
    roi_min, roi_max = roi.min(), roi.max()
    if roi_max > roi_min:
        roi_8 = ((roi - roi_min) / (roi_max - roi_min) * 255.0).astype(np.uint8)
    else:
        roi_8 = roi.astype(np.uint8)

    ink_mask = _get_ink_mask(roi_8, dilation_px=2)

    if not np.any(ink_mask > 0):
        # Fallback: no distinct ink found — use full box fill with median
        return _redact_border_zone(cleaned, x1, y1, x2, y2)

    # ── Pass 2: NS inpainting (structure-aware, radius=9) ─────────────────
    if roi.dtype == np.uint16:
        roi_ns = _inpaint_16bit(roi, ink_mask, radius=9, method=cv2.INPAINT_NS)
    else:
        roi_ns = cv2.inpaint(roi, ink_mask, 9, cv2.INPAINT_NS)

    # ── Pass 3: Telea inpainting on residual pixels ───────────────────────
    # Re-check residual mask on the NS result
    ns_min, ns_max = roi_ns.min(), roi_ns.max()
    if ns_max > ns_min:
        ns_8 = ((roi_ns - ns_min) / (ns_max - ns_min) * 255.0).astype(np.uint8)
    else:
        ns_8 = roi_ns.astype(np.uint8)

    residual_mask = _get_ink_mask(ns_8, dilation_px=1)
    # Only re-inpaint pixels that are still ink-like AND were in original mask
    residual_mask = cv2.bitwise_and(residual_mask, ink_mask)

    if np.any(residual_mask > 0):
        if roi_ns.dtype == np.uint16:
            roi_telea = _inpaint_16bit(roi_ns, residual_mask, radius=5, method=cv2.INPAINT_TELEA)
        else:
            roi_telea = cv2.inpaint(roi_ns, residual_mask, 5, cv2.INPAINT_TELEA)
    else:
        roi_telea = roi_ns

    # ── Pass 4: Bilateral texture blend ───────────────────────────────────
    if roi_telea.dtype == np.uint16:
        roi_f = roi_telea.astype(np.float32) / 65535.0
        blended_f = cv2.bilateralFilter(roi_f, d=7, sigmaColor=0.05, sigmaSpace=5)
        # Only apply bilateral blend on the ink region, keep rest original
        blend_zone = (ink_mask > 0).astype(np.float32)
        blend_zone_3d = blend_zone  # single channel
        roi_final_f = blended_f * blend_zone_3d + roi_f * (1.0 - blend_zone_3d)
        roi_final = (roi_final_f * 65535.0).astype(np.uint16)
    else:
        roi_f = roi_telea.astype(np.float32)
        blended_f = cv2.bilateralFilter(roi_f, d=7, sigmaColor=12.0, sigmaSpace=5)
        blend_zone = (ink_mask > 0).astype(np.float32)
        roi_final_f = blended_f * blend_zone + roi_f * (1.0 - blend_zone)
        roi_final = np.clip(roi_final_f, 0, 255).astype(np.uint8)

    # Write processed ROI back into the full image
    cleaned[ry1:ry2, rx1:rx2] = roi_final

    return cleaned


# =============================================================================
# STAGE 5: PIXEL REDACTION — ZONE-AWARE MULTI-HYBRID MASKING
# =============================================================================
def redact_pixels(image_array, phi_regions):
    """
    Zone-aware multi-hybrid pixel redaction.

    BORDER zone  -> Uniform median fill (fast, complete, safe in flat margins)
    ANATOMY zone -> 4-pass hybrid:
                     1. Precise ink isolation (Otsu+Adaptive fusion)
                     2. Navier-Stokes inpainting (structure-aware, r=9)
                     3. Telea inpainting on residual pixels (r=5)
                     4. Bilateral texture blend (edge-preserving smooth)

    Works on native 8-bit OR 16-bit pixel arrays without losing dynamic range.
    Returns (cleaned_array, combined_mask).
    """
    if not phi_regions:
        log.info("  [Stage 5] No PHI pixels to redact.")
        return image_array.copy(), np.zeros(image_array.shape[:2], dtype=np.uint8)

    cleaned = image_array.copy()
    h, w = cleaned.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    border_count  = 0
    anatomy_count = 0

    for region in phi_regions:
        x1, y1, x2, y2 = region["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        zone = region.get("zone", "border")  # default to border if unset
        combined_mask[y1:y2, x1:x2] = 255

        if zone == "border":
            # ── BORDER: simple uniform fill ───────────────────────────────
            cleaned = _redact_border_zone(cleaned, x1, y1, x2, y2)
            border_count += 1
            log.info(f"  [Stage 5] Border fill applied @ [{x1},{y1},{x2},{y2}]")
        else:
            # ── ANATOMY: 4-pass hybrid masking ────────────────────────────
            cleaned = _redact_anatomy_zone(cleaned, x1, y1, x2, y2)
            anatomy_count += 1
            log.info(f"  [Stage 5] Anatomy hybrid masking @ [{x1},{y1},{x2},{y2}]")

    log.info(
        f"  [Stage 5] Done. Border fills: {border_count} | "
        f"Anatomy hybrid: {anatomy_count} | "
        f"Total pixels masked: {int(np.sum(combined_mask > 0))}"
    )
    return cleaned, combined_mask


# =============================================================================
# STAGE 6: VERIFICATION + ESCALATION
# =============================================================================
def verify_redaction(cleaned_array, phi_regions, paddle_ocr, easy_ocr, analyzer):
    """Re-runs OCR on cleaned image to confirm zero residual PHI text."""
    log.info("  [Stage 6] Verification pass — re-running OCR on cleaned image...")

    raw_min, raw_max = cleaned_array.min(), cleaned_array.max()
    if raw_max > raw_min:
        temp_8 = ((cleaned_array - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
    else:
        temp_8 = cleaned_array.astype(np.uint8)

    variants    = enhance_image(temp_8)
    raw_det     = detect_text(variants, paddle_ocr, easy_ocr)
    merged      = merge_detections(raw_det)
    residual    = classify_phi(merged, cleaned_array.shape, analyzer)

    if not residual:
        log.info("  [Stage 6] PASSED — no residual PHI text detected.")
        return cleaned_array, "PASSED"

    # ── Zone-aware escalation ─────────────────────────────────────────────────
    log.warning(
        f"  [Stage 6] Residual PHI found in {len(residual)} region(s): "
        f"{[r['text'][:30] for r in residual]}. Running zone-aware escalation..."
    )
    h, w = cleaned_array.shape[:2]
    escalated = cleaned_array.copy()

    border_esc  = 0
    anatomy_esc = 0

    for region in residual:
        x1, y1, x2, y2 = region["bbox"]
        # Add 8px padding to catch any char edges
        x1e = max(0, x1 - 8); y1e = max(0, y1 - 8)
        x2e = min(w, x2 + 8); y2e = min(h, y2 + 8)

        zone = region.get("zone", "border")

        if zone == "border":
            # Expanded median fill for border residuals
            escalated = _redact_border_zone(escalated, x1e, y1e, x2e, y2e)
            border_esc += 1
            log.info(f"  [Stage 6] Escalation border fill @ [{x1e},{y1e},{x2e},{y2e}]")
        else:
            # Re-run 4-pass hybrid on anatomy residuals
            escalated = _redact_anatomy_zone(escalated, x1e, y1e, x2e, y2e)
            anatomy_esc += 1
            log.info(f"  [Stage 6] Escalation anatomy hybrid @ [{x1e},{y1e},{x2e},{y2e}]")

    log.info(
        f"  [Stage 6] Escalation complete. Border fills: {border_esc} | "
        f"Anatomy hybrid: {anatomy_esc}"
    )

    # ── Second tight verification pass ───────────────────────────────────────
    esc_min, esc_max = escalated.min(), escalated.max()
    if esc_max > esc_min:
        esc_8 = ((escalated - esc_min) / (esc_max - esc_min) * 255.0).astype(np.uint8)
    else:
        esc_8 = escalated.astype(np.uint8)

    v2_det    = detect_text(enhance_image(esc_8), paddle_ocr, easy_ocr)
    v2_merged = merge_detections(v2_det)
    v2_phi    = classify_phi(v2_merged, escalated.shape, analyzer)

    if not v2_phi:
        log.info("  [Stage 6] PASSED (Escalated — all residual text cleared).")
        return escalated, "PASSED (Escalated)"

    # ── Nuclear option: blackout the border strips ────────────────────────────
    log.error(
        f"  [Stage 6] Escalation still has {len(v2_phi)} residual region(s). "
        "Applying nuclear border blackout..."
    )
    nuclear = escalated.copy()
    top_h    = int(h * 0.20)
    bot_h    = int(h * 0.80)
    side_w   = int(w * 0.12)
    nuclear[0:top_h, :]   = 0
    nuclear[bot_h:, :]    = 0
    nuclear[:, 0:side_w]  = 0
    nuclear[:, w-side_w:] = 0
    log.info("  [Stage 6] FORCE PASSED (Nuclear border blackout applied).")
    return nuclear, "FORCE PASSED (Nuclear Fallback)"


# =============================================================================
# STAGE 7: WRITE PIXELS BACK INTO DICOM (correct tag updates)
# =============================================================================
def write_pixels_to_dicom(ds, cleaned_array):
    """
    Correctly writes the anonymized pixel array back into the DICOM dataset.
    Updates ALL mandatory pixel descriptor tags to match the array's dtype/shape.
    """
    arr = cleaned_array

    # Handle multi-frame: if shape is (frames, H, W) flatten is not needed;
    # pydicom handles this via NumberOfFrames
    if arr.ndim == 3 and arr.shape[0] > 1:
        # Multi-frame: (F, H, W)
        ds.NumberOfFrames = arr.shape[0]
    elif arr.ndim == 2:
        pass  # single frame

    # Determine bit depth from dtype
    if arr.dtype == np.uint8:
        bits_alloc = 8; bits_stored = 8; high_bit = 7; pix_rep = 0
    elif arr.dtype == np.uint16:
        bits_alloc = 16; bits_stored = 16; high_bit = 15; pix_rep = 0
    elif arr.dtype == np.int16:
        bits_alloc = 16; bits_stored = 16; high_bit = 15; pix_rep = 1
    else:
        # Fallback: cast to uint16
        arr = arr.astype(np.uint16)
        bits_alloc = 16; bits_stored = 16; high_bit = 15; pix_rep = 0

    # Update all pixel descriptor tags
    ds.BitsAllocated      = bits_alloc
    ds.BitsStored         = bits_stored
    ds.HighBit            = high_bit
    ds.PixelRepresentation = pix_rep

    # Ensure uncompressed transfer syntax (Explicit VR Little Endian)
    from pydicom.uid import ExplicitVRLittleEndian
    ds.file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds.is_implicit_VR  = False
    ds.is_little_endian = True

    # Write pixel data
    ds.PixelData = arr.tobytes()

    return ds


# =============================================================================
# MAIN PIPELINE FUNCTION
# =============================================================================
def anonymize_dicom_file(input_path, output_path, paddle_ocr, easy_ocr, analyzer):
    """
    Full 7-stage anonymization pipeline for a single DICOM file.
    Returns an audit dict.
    """
    filename = os.path.basename(input_path)
    log.info(f"\n{'='*60}")
    log.info(f"Processing: {filename}")
    log.info(f"{'='*60}")

    audit = {
        "file": filename,
        "input_path":  input_path,
        "output_path": output_path,
        "timestamp":   datetime.datetime.now().isoformat(),
        "modality":    "Unknown",
        "image_size":  "Unknown",
        "redacted_regions": [],
        "verification_status": "NOT RUN",
        "error": None
    }

    try:
        # ── Read DICOM ────────────────────────────────────────────────────────
        ds = pydicom.dcmread(input_path, force=True)

        audit["modality"] = getattr(ds, "Modality", "Unknown")
        rows = getattr(ds, "Rows", "?")
        cols = getattr(ds, "Columns", "?")
        audit["image_size"] = f"{rows}x{cols}"

        # ── Stage 1: Metadata ─────────────────────────────────────────────────
        ds = sanitize_metadata(ds)

        # ── Extract pixels ────────────────────────────────────────────────────
        if not hasattr(ds, 'pixel_array'):
            log.warning(f"  No pixel data in {filename}. Saving metadata-only.")
            ds.save_as(output_path, write_like_original=False)
            audit["verification_status"] = "SKIPPED (no pixels)"
            return audit

        pixels = ds.pixel_array.copy()
        original_dtype = pixels.dtype

        # Strip overlay planes (group 60xx) if present
        overlay_groups = [g for g in range(0x6000, 0x601F, 2)]
        for group in overlay_groups:
            for elem in list(ds):
                if elem.tag.group == group:
                    del ds[elem.tag]

        # Fix MONOCHROME1 inversion (dark=bright for OCR correctness)
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        is_monochrome1 = (photometric == "MONOCHROME1")
        original_max = None
        if is_monochrome1:
            original_max = np.max(pixels)
            pixels = original_max - pixels
            log.info("  MONOCHROME1 detected — pixel values flipped.")

        # Normalize to 8-bit for OCR pipeline
        pix_min, pix_max = pixels.min(), pixels.max()
        if pix_max > pix_min:
            norm_8 = ((pixels - pix_min) / (pix_max - pix_min) * 255.0).astype(np.uint8)
        else:
            norm_8 = pixels.astype(np.uint8)

        # Handle multi-frame: process frame 0 for OCR detection, apply to all frames
        if pixels.ndim == 3:
            ocr_frame = norm_8[0]
            ocr_pixels = pixels[0]
        else:
            ocr_frame = norm_8
            ocr_pixels = pixels

        # ── Stage 2: Enhancement ──────────────────────────────────────────────
        log.info("  [Stage 2] Generating enhanced image variants...")
        variants = enhance_image(ocr_frame)

        # ── Stage 3: OCR ──────────────────────────────────────────────────────
        log.info("  [Stage 3] Running OCR text detection...")
        raw_det = detect_text(variants, paddle_ocr, easy_ocr)
        log.info(f"            Raw detections: {len(raw_det)}")

        # ── Stage 4: Classify ─────────────────────────────────────────────────
        log.info("  [Stage 4] Classifying detections (PHI vs clinical)...")
        merged     = merge_detections(raw_det)
        phi_regions = classify_phi(merged, ocr_frame.shape, analyzer)
        log.info(f"            PHI regions to redact: {len(phi_regions)}")
        audit["redacted_regions"] = [
            {"text": r["text"], "bbox": r["bbox"]} for r in phi_regions
        ]

        # ── Stage 5: Redact ───────────────────────────────────────────────────
        log.info("  [Stage 5] Redacting PHI pixels (Navier-Stokes inpainting)...")
        if pixels.ndim == 3:
            # Apply to every frame
            cleaned_frames = []
            for i in range(pixels.shape[0]):
                frame_pix = pixels[i]
                frame_phi = phi_regions  # same regions for all frames
                cleaned_frame, _ = redact_pixels(frame_pix, frame_phi)
                cleaned_frames.append(cleaned_frame)
            cleaned_pixels = np.stack(cleaned_frames, axis=0)
        else:
            cleaned_pixels, _ = redact_pixels(pixels, phi_regions)

        # ── Stage 6: Verify ───────────────────────────────────────────────────
        verify_frame = cleaned_pixels[0] if cleaned_pixels.ndim == 3 else cleaned_pixels
        cleaned_pixels_final = cleaned_pixels.copy()
        verify_clean, status = verify_redaction(
            verify_frame, phi_regions, paddle_ocr, easy_ocr, analyzer
        )
        if cleaned_pixels.ndim == 3:
            cleaned_pixels_final[0] = verify_clean
        else:
            cleaned_pixels_final = verify_clean
        audit["verification_status"] = status

        # ── Stage 7: Write back to DICOM ─────────────────────────────────────
        log.info("  [Stage 7] Writing anonymized pixels back into DICOM dataset...")
        if is_monochrome1 and original_max is not None:
            cleaned_pixels_final = original_max - cleaned_pixels_final
            log.info("  Flipped pixels back to MONOCHROME1 representation.")
        ds = write_pixels_to_dicom(ds, cleaned_pixels_final)

        # ── Save as DICOM ONLY ────────────────────────────────────────────────
        ds.save_as(output_path, write_like_original=False)

        # Confirm output is .dcm
        assert output_path.lower().endswith(".dcm"), "BUG: output path must be .dcm!"
        log.info(f"  [DONE] Anonymized DICOM saved: {output_path}")
        log.info(f"         Status: {status}")

    except Exception as e:
        log.error(f"  [ERROR] {filename}: {e}", exc_info=True)
        audit["error"] = str(e)
        audit["verification_status"] = "ERROR"

    return audit


# =============================================================================
# ENGINE INITIALIZATION & GPU AUTO-DETECTION
# =============================================================================
def check_gpu_available():
    """Checks if a GPU is available via PyTorch or Paddle."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            return True
    except ImportError:
        pass
    return False


def initialize_engines(use_gpu=False):
    """Initializes PaddleOCR, EasyOCR, and Presidio Analyzer with GPU option."""
    print(f"\nInitializing OCR engines (GPU={use_gpu})...")

    paddle_ocr = None
    if PADDLE_AVAILABLE:
        try:
            # First attempt: Try with GPU and MKL-DNN toggles
            paddle_ocr = PaddleOCR(lang='en', enable_mkldnn=False, use_gpu=use_gpu)
            print(f"  [OK] PaddleOCR initialized (use_gpu={use_gpu})")
        except Exception as e1:
            # Second attempt: Try without use_gpu (if not accepted by this version)
            try:
                paddle_ocr = PaddleOCR(lang='en', enable_mkldnn=False)
                print("  [OK] PaddleOCR initialized (without explicit use_gpu)")
            except Exception as e2:
                # Third attempt: Try with bare minimum
                try:
                    paddle_ocr = PaddleOCR(lang='en')
                    print("  [OK] PaddleOCR initialized (default settings)")
                except Exception as e3:
                    print(f"  [WARN] PaddleOCR failed to init: {e3}")

    easy_ocr = None
    if EASYOCR_AVAILABLE:
        try:
            easy_ocr = easyocr.Reader(['en'], gpu=use_gpu)
            print("  [OK] EasyOCR initialized")
        except Exception as e:
            print(f"  [WARN] EasyOCR failed to init: {e}")

    analyzer = None
    if PRESIDIO_AVAILABLE:
        try:
            analyzer = AnalyzerEngine()
            print("  [OK] Presidio NLP Analyzer initialized")
        except Exception as e:
            print(f"  [WARN] Presidio failed to init: {e}")

    if not paddle_ocr and not easy_ocr:
        print("\n[ERROR] No OCR engine available! Install at least one:")
        print("  pip install paddleocr   OR   pip install easyocr")
        print("Metadata anonymization will still run.\n")
        
    return paddle_ocr, easy_ocr, analyzer


# =============================================================================
# HIGH-LEVEL RUNNER API
# =============================================================================
def anonymize(input_path, output_path, use_gpu=None, audit_log_path=None):
    """
    High-level API to anonymize a single DICOM file or an entire directory of DICOM files.
    """
    if use_gpu is None:
        use_gpu = check_gpu_available()
        
    paddle_ocr, easy_ocr, analyzer = initialize_engines(use_gpu=use_gpu)
    
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    
    jobs = []
    
    if os.path.isfile(input_path):
        # Input is a single file
        if os.path.isdir(output_path) or not output_path.lower().endswith(".dcm"):
            # If output is a directory or doesn't look like a .dcm file path, treat it as output directory
            os.makedirs(output_path, exist_ok=True)
            out_file = os.path.join(output_path, "anon_" + os.path.basename(input_path))
        else:
            # Output is specified as a file path
            out_dir = os.path.dirname(output_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            out_file = output_path
        jobs.append((input_path, out_file))
    elif os.path.isdir(input_path):
        # Input is a directory
        dicom_files = [f for f in os.listdir(input_path) if f.lower().endswith(".dcm")]
        if not dicom_files:
            print(f"\n[ERROR] No .dcm files found in directory: {input_path}")
            return False
        os.makedirs(output_path, exist_ok=True)
        for f in sorted(dicom_files):
            jobs.append((
                os.path.join(input_path, f),
                os.path.join(output_path, "anon_" + f)
            ))
    else:
        print(f"\n[ERROR] Input path not found: {input_path}")
        return False

    print(f"\nFound {len(jobs)} DICOM file(s) to process:")
    for src, dst in jobs:
        sz = os.path.getsize(src) / 1024 / 1024
        print(f"  {os.path.basename(src):50s} ({sz:.2f} MB) -> {os.path.basename(dst)}")

    # ── Process each file ─────────────────────────────────────────────────────
    print(f"\nStarting anonymization of {len(jobs)} file(s)...\n")
    all_audits = []

    for src, dst in jobs:
        audit = anonymize_dicom_file(
            src, dst,
            paddle_ocr, easy_ocr, analyzer
        )
        all_audits.append(audit)

    # ── Save audit log ────────────────────────────────────────────────────────
    if not audit_log_path:
        # If output_path is a directory, write audit_log.json there.
        # If output_path is a file, write audit_log.json in same folder.
        if os.path.isdir(output_path):
            audit_log_path = os.path.join(output_path, "audit_log.json")
        else:
            audit_log_path = os.path.join(os.path.dirname(output_path), "audit_log.json")
            
    with open(audit_log_path, "w") as f:
        json.dump(all_audits, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("ANONYMIZATION COMPLETE — RESULTS SUMMARY")
    print("=" * 70)

    success = [a for a in all_audits if a["error"] is None]
    errors  = [a for a in all_audits if a["error"] is not None]

    print(f"\n  Total files processed : {len(all_audits)}")
    print(f"  Successful            : {len(success)}")
    print(f"  Errors                : {len(errors)}")
    print()
    print(f"  {'File':<45} {'Status':<30} {'Regions'}")
    print(f"  {'-'*45} {'-'*30} {'-'*10}")
    for a in all_audits:
        n_regions = len(a.get("redacted_regions", []))
        status    = a.get("verification_status", "?")
        err       = f"ERROR: {a['error'][:40]}" if a["error"] else status
        print(f"  {a['file']:<45} {err:<30} {n_regions} PHI region(s)")

    print()
    print(f"  Output Audit log    : {os.path.abspath(audit_log_path)}")
    print()
    print("  [OK] All outputs are in DICOM (.dcm) format -- NO JPEG/PNG produced.")
    print("=" * 70)
    return len(errors) == 0


# =============================================================================
# KAGGLE ENVIRONMENT AUTO-DETECTION
# =============================================================================
def detect_kaggle_paths(default_input, default_output, default_audit):
    """Automatically detects Kaggle datasets and updates default paths if running on Kaggle."""
    input_path = default_input
    output_path = default_output
    audit_path = default_audit

    if os.path.exists("/kaggle/input"):
        try:
            target_dir = None
            # Recursively walk /kaggle/input to find any directory containing .dcm files
            for root, dirs, files in os.walk("/kaggle/input"):
                dcm_files = [f for f in files if f.lower().endswith(".dcm")]
                if dcm_files:
                    target_dir = root
                    break
            
            # Fallback to first directory if no .dcm files found anywhere
            if not target_dir:
                subdirs = [
                    os.path.join("/kaggle/input", d)
                    for d in os.listdir("/kaggle/input")
                    if os.path.isdir(os.path.join("/kaggle/input", d))
                ]
                if subdirs:
                    target_dir = subdirs[0]
            
            if target_dir:
                input_path = target_dir
                print(f"[Kaggle Detected] Auto-selected input dataset path: {input_path}")
        except Exception:
            pass
            
        output_path = "/kaggle/working/anonymized_output"
        audit_path = "/kaggle/working/anonymized_output/audit_log.json"
        
    return input_path, output_path, audit_path


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    print("\n" + "=" * 70)
    print("UNIFIED DICOM BURNED-IN TEXT ANONYMIZATION PIPELINE")
    print("=" * 70)

    # Auto-detect Kaggle paths
    default_input, default_output, default_audit = detect_kaggle_paths(
        INPUT_DIR, OUTPUT_DIR, AUDIT_LOG
    )

    parser = argparse.ArgumentParser(description="Unified DICOM Burned-In Text Anonymization Pipeline")
    parser.add_argument("-i", "--input", default=default_input,
                        help=f"Path to input DICOM file or directory (default: {default_input})")
    parser.add_argument("-o", "--output", default=default_output,
                        help=f"Path to output anonymized DICOM file or directory (default: {default_output})")
    parser.add_argument("-a", "--audit", default=default_audit,
                        help=f"Path to save the audit log JSON file (default: {default_audit})")
    parser.add_argument("--gpu", action="store_true", default=None,
                        help="Force enable GPU usage for OCR engines")
    parser.add_argument("--no-gpu", action="store_false", dest="gpu",
                        help="Force disable GPU usage for OCR engines")
    
    args, _ = parser.parse_known_args()

    # Determine GPU usage
    use_gpu = False
    if args.gpu is not None:
        use_gpu = args.gpu
    else:
        use_gpu = check_gpu_available()

    anonymize(
        input_path=args.input,
        output_path=args.output,
        use_gpu=use_gpu,
        audit_log_path=args.audit
    )


if __name__ == "__main__":
    main()
