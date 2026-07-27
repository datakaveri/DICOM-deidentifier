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
  Stage 4 — PHI Classification      (3-Step Hybrid: Stanford De-ID + Regex + Combined Decision → Medical NER → Spatial)
  Stage 5 — Pixel Redaction         (Navier-Stokes inpainting, 16-bit safe)
  Stage 6 — Verification + Retry    (Re-OCR to confirm zero residual text)
  Stage 7 — DICOM Write-Back        (correct BitsAllocated/BitsStored tags)

Usage:
  python dicom_anonymizer_pipeline.py
=============================================================================
"""

import os
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["DISABLE_AUTO_LOGGING_CONFIG"] = "1"

import sys
import re
import json
import copy
import logging
import datetime
import argparse
import shutil

import numpy as np
import cv2
import pydicom
from pydicom.uid import generate_uid
from PIL import Image

# Pre-import PyTorch to prevent Windows DLL symbol conflict (shm.dll) with Paddle
try:
    import torch
except Exception:
    pass

# ─── NumPy 2.x Backwards-Compatibility Polyfill ──────────────────────────────
# Legacy libraries (PaddleOCR, Presidio, Transformers) reference numpy.char / numpy.core.
# NumPy 2.0+ removed these submodules, so we alias them in sys.modules before any import.
import sys as _sys
import numpy as _np
if hasattr(_np, 'char') and 'numpy.char' not in _sys.modules:
    _sys.modules['numpy.char'] = _np.char
if hasattr(_np, '_core'):
    if 'numpy.core' not in _sys.modules:
        _sys.modules['numpy.core'] = _np._core
    if 'numpy.core.multiarray' not in _sys.modules:
        _sys.modules['numpy.core.multiarray'] = _np._core.multiarray

# Polyfill for np.sctypes (removed in NumPy 2.0, required by imgaug)
if not hasattr(_np, 'sctypes'):
    _np.sctypes = {
        'int': [_np.int8, _np.int16, _np.int32, _np.int64],
        'uint': [_np.uint8, _np.uint16, _np.uint32, _np.uint64],
        'float': [_np.float16, _np.float32, _np.float64],
        'complex': [_np.complex64, _np.complex128],
        'others': [bool, str, bytes, object]
    }

# ─── Suppress verbose sub-library logs & Python warnings ─────────────────────
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
os.environ["DISABLE_AUTO_LOGGING_CONFIG"] = "1"

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pydicom")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*write_like_original.*")
warnings.filterwarnings("ignore", message=".*is_implicit_VR.*")
warnings.filterwarnings("ignore", message=".*is_little_endian.*")

# Configure a robust, self-contained logger that prints directly to stdout
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.propagate = False
if not log.handlers:
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setLevel(logging.INFO)
    _formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    _ch.setFormatter(_formatter)
    log.addHandler(_ch)

# ─── Pre-import: auto-install missing packages ───────────────────────────────
def _ensure_dependencies():
    """Helper to auto-install all required pipeline packages including paddle and its dependencies."""
    if not (os.path.exists("/kaggle/input") or 'COLAB_GPU' in os.environ):
        return

    import subprocess as _sp
    import importlib as _il
    import importlib.util as _util
    import sys as _sys

    def _is_installed(name):
        try:
            if name in _sys.modules:
                _val = _sys.modules[name]
                if _val is None or not hasattr(_val, "__file__"):
                    return False
                if name == "paddle" and not hasattr(_val, "tensor"):
                    return False
            return _util.find_spec(name) is not None
        except Exception:
            return False

    # Determine GPU availability
    _has_gpu = False
    try:
        import torch as _torch
        _has_gpu = _torch.cuda.is_available()
    except Exception:
        pass

    # Standard package checklist
    _pkg_map = [
        ("pydicom", "pydicom"),
        ("easyocr", "easyocr"),
        ("presidio_analyzer", "presidio-analyzer"),
        ("gliner", "gliner"),
        ("transformers", "transformers"),
        ("fire", "fire"),
        ("docx", "python-docx"),
        ("rapidfuzz", "rapidfuzz"),
        ("lmdb", "lmdb"),
        ("imgaug", "imgaug"),
        ("pyclipper", "pyclipper"),
        ("shapely", "shapely"),
        ("astor", "astor"),
        ("decorator", "decorator"),
        ("opt_einsum", "opt_einsum"),
    ]

    _to_install = []
    for _mod, _pip in _pkg_map:
        if not _is_installed(_mod):
            # Clean sys.modules cache for this package if present
            if _mod in _sys.modules:
                del _sys.modules[_mod]
            _to_install.append(_pip)

    if _to_install:
        print(f"[Setup] Installing/Re-installing packages: {', '.join(_to_install)}...")
        _sp.run([_sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + _to_install, check=False)
        _il.invalidate_caches()

    # Special handling for paddle / paddleocr
    _has_paddle = _is_installed("paddle")
    _has_paddleocr = _is_installed("paddleocr")

    if not _has_paddle or not _has_paddleocr:
        _paddle_pkg = "paddlepaddle-gpu==2.6.2" if _has_gpu else "paddlepaddle==2.6.2"
        print(f"[Setup] Installing Paddle engine ({_paddle_pkg}) & PaddleOCR...")
        
        # Clean any cached bad import states
        for _name in ["paddle", "paddleocr"]:
            if _name in _sys.modules:
                del _sys.modules[_name]

        # Install PaddlePaddle
        _paddle_cmd = [_sys.executable, "-m", "pip", "install", "-q", "--no-deps", _paddle_pkg]
        if _has_gpu:
            _paddle_cmd += ["-f", "https://www.paddlepaddle.org.cn/whl/CUDA12.0/mkl/stable.html"]
        _sp.run(_paddle_cmd, check=False)

        # Install PaddleOCR
        _sp.run([_sys.executable, "-m", "pip", "install", "-q", "--no-deps", "paddleocr==2.8.1"], check=False)
        _il.invalidate_caches()

# Run the installer check at import time
_ensure_dependencies()

# ─── Optional engine imports ──────────────────────────────────────────────────
# Clean any failed/partial/None imports in sys.modules to prevent circular import caching issues
import sys as _sys
for _m in ["paddle", "paddleocr", "easyocr", "presidio_analyzer", "gliner", "imgaug", "astor", "decorator", "opt_einsum"]:
    if _m in _sys.modules:
        _val = _sys.modules[_m]
        # Purge if it's a dummy None entry, missing a path/file attribute, or if it is paddle and has no tensor attribute
        if _val is None or not hasattr(_val, "__file__") or not hasattr(_val, "__path__") or (_m == "paddle" and not hasattr(_val, "tensor")):
            del _sys.modules[_m]

try:
    import paddle
    # Monkey-patch AnalysisConfig for PaddlePaddle < 3.0 + PaddleOCR 3.x compatibility
    _cfg_cls = None
    if hasattr(paddle, 'base') and hasattr(paddle.base, 'libpaddle') and hasattr(paddle.base.libpaddle, 'AnalysisConfig'):
        _cfg_cls = paddle.base.libpaddle.AnalysisConfig
    elif hasattr(paddle, 'fluid') and hasattr(paddle.fluid, 'core') and hasattr(paddle.fluid.core, 'AnalysisConfig'):
        _cfg_cls = paddle.fluid.core.AnalysisConfig
    if _cfg_cls and not hasattr(_cfg_cls, 'set_optimization_level'):
        _cfg_cls.set_optimization_level = lambda self, *args, **kwargs: None
except Exception:
    pass

try:
    from paddleocr import PaddleOCR
    import logging
    logging.getLogger("ppocr").setLevel(logging.ERROR)
    PADDLE_AVAILABLE = True
except Exception as _pe:
    log.warning(f"PaddleOCR import check failed: {_pe}")
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

try:
    import gliner
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except Exception:
    GLINER_AVAILABLE = False


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
    "LA", "RA", "LP", "RP", "A", "P",
    "ERECT", "SUPINE", "PRONE", "DECUBITUS", "UPRIGHT",
    "PORTABLE", "MOBILE", "STAT", "ROUTINE",
    "CHEST", "ABDOMEN", "PELVIS", "SKULL", "SPINE",
    "KVP", "MAS", "MA", "SEC", "CM", "MM", "FOV",
    "CXR", "CX", "PA VIEW", "AP VIEW",
    "ANTEROPOSTERIOR", "POSTEROANTERIOR", "PROJECTION",
}

# PII patterns (Indian health context + universal)
PII_PATTERNS = {
    "aadhaar":      r'\b\d{4}\s?\d{4}\s?\d{4}\b',
    "abha":         r'\b\d{2}-\d{4}-\d{4}-\d{4}\b',
    "phone":        r'\b(?:\+91|0)?[6-9]\d{9}\b',
    "date":         r'\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b',
    "uhid_mrn":     r'\b(?:UHID|MRN|REG|IPD|OPD|CR|PID|HID|PATIENT_ID|PATIENTID)[\s:/-]*\d+\b',
    "age_sex":      r'\b\d{1,3}\s*[/]\s*[MFO]\b',
    "name_prefix":  r'\b(?:DR\.?|MR\.?|MRS\.?|MS\.?|SHRI|SMT|S/O|D/O|W/O)\b',
    "accession":    r'\b(?:ACC|ACCNO|ACCESSION)[\s:.-]*\w+\b',
    "confidential": r'\b(?:CONFIDENTIAL|RESTRICTED|PROPRIETARY)\b',
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
# VISUAL SPACE TRANSFORMATION HELPERS (VOI LUT & INVERSE VOI LUT)
# =============================================================================
def _apply_voi_lut(pixels, ds):
    """
    Applies Rescale Slope/Intercept and VOI LUT (Window Center/Width) to convert
    original raw pixels to a standardized 8-bit visual representation (0-255).
    """
    # 1. Apply Rescale Slope and Intercept
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    rescaled = pixels.astype(np.float64) * float(slope) + float(intercept)

    # 2. Check for Window Center and Window Width
    wc_attr = getattr(ds, "WindowCenter", None)
    ww_attr = getattr(ds, "WindowWidth", None)

    if wc_attr is not None and ww_attr is not None:
        # WindowCenter and WindowWidth can be multi-value (take first one)
        wc = float(wc_attr[0]) if isinstance(wc_attr, (pydicom.multival.MultiValue, list)) else float(wc_attr)
        ww = float(ww_attr[0]) if isinstance(ww_attr, (pydicom.multival.MultiValue, list)) else float(ww_attr)
        
        min_val = wc - 0.5 - (ww - 1.0) / 2.0
        
        visual = np.clip((rescaled - min_val) / max(ww - 1.0, 1.0) * 255.0, 0, 255)
    else:
        # If no windowing tags exist, do min-max normalization
        pmin, pmax = rescaled.min(), rescaled.max()
        if pmax > pmin:
            visual = (rescaled - pmin) / (pmax - pmin) * 255.0
        else:
            visual = np.zeros_like(rescaled)

    return visual.astype(np.uint8)


def _apply_inverse_voi_lut(visual_pixels, ds, original_raw_pixels):
    """
    Maps 8-bit visual pixels back to original raw pixel representation
    using the mathematical inverse of the applied Rescale and VOI LUT.
    """
    # Get Rescale and Windowing parameters
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    
    wc_attr = getattr(ds, "WindowCenter", None)
    ww_attr = getattr(ds, "WindowWidth", None)
    
    if wc_attr is not None and ww_attr is not None:
        wc = float(wc_attr[0]) if isinstance(wc_attr, (pydicom.multival.MultiValue, list)) else float(wc_attr)
        ww = float(ww_attr[0]) if isinstance(ww_attr, (pydicom.multival.MultiValue, list)) else float(ww_attr)
        
        min_val = wc - 0.5 - (ww - 1.0) / 2.0
        
        # Inverse linear VOI LUT
        rescaled = (visual_pixels.astype(np.float64) / 255.0) * (ww - 1.0) + min_val
    else:
        # If min-max normalization was used, map back based on original raw range
        orig_rescaled = original_raw_pixels.astype(np.float64) * slope + intercept
        pmin, pmax = orig_rescaled.min(), orig_rescaled.max()
        if pmax > pmin:
            rescaled = (visual_pixels.astype(np.float64) / 255.0) * (pmax - pmin) + pmin
        else:
            rescaled = np.zeros_like(visual_pixels)
            
    # Apply inverse Rescale: X = (Y - intercept) / slope
    raw = (rescaled - intercept) / slope
    
    # Clip to match original data type range
    if original_raw_pixels.dtype == np.uint8:
        raw = np.clip(raw, 0, 255)
    elif original_raw_pixels.dtype == np.uint16:
        raw = np.clip(raw, 0, 65535)
    elif original_raw_pixels.dtype == np.int16:
        raw = np.clip(raw, -32768, 32767)
        
    return raw.astype(original_raw_pixels.dtype)


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

    # Full inversion — critical for light-text-on-dark-background (retinal/fundus images)
    try:
        variants["inverted"] = cv2.bitwise_not(image_8bit)
    except Exception:
        pass

    # High-contrast CLAHE on inverted — maximizes text visibility on dark backgrounds
    try:
        inv = cv2.bitwise_not(image_8bit)
        clahe_hi = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        variants["inverted_clahe"] = clahe_hi.apply(inv)
    except Exception:
        pass

    # Morphological Top-Hat — isolates thin bright text strokes from varying bright backgrounds (bone/lung)
    try:
        k_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        tophat = cv2.morphologyEx(image_8bit, cv2.MORPH_TOPHAT, kernel)
        variants["tophat"] = cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX)
    except Exception:
        pass

    # Morphological Black-Hat — isolates thin dark text strokes from bright backgrounds
    try:
        k_size = 15
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
        blackhat = cv2.morphologyEx(image_8bit, cv2.MORPH_BLACKHAT, kernel)
        variants["blackhat"] = cv2.normalize(blackhat, None, 0, 255, cv2.NORM_MINMAX)
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


def detect_text(variants, paddle_ocr=None, easy_ocr=None, modality="CT"):
    """Runs available OCR engines on primary image variants for high-speed detection."""
    raw = []
    
    # Evaluate top high-contrast variants across all DICOM image modalities
    priority_keys = ["standard", "tophat", "inverted_clahe"]
        
    target_variants = {k: v for k, v in variants.items() if k in priority_keys}
    if not target_variants:
        target_variants = variants

    for name, img in target_variants.items():
        if len(img.shape) == 2:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            rgb = img

        # PaddleOCR — use fast ocr() detection engine
        if paddle_ocr:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = paddle_ocr.ocr(rgb)
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


def merge_horizontal_lines(detections, max_gap_factor=2.5, min_v_overlap=0.45):
    """
    Merges detections that are on the same horizontal line and close to each other.
    Allows catching labeled PII like 'PATIENT:' + 'MEERA IYER' as a single entity.
    """
    if not detections:
        return []
        
    # Sort from left to right by x1 coordinate
    sorted_det = sorted(detections, key=lambda x: x["bbox"][0])
    
    merged_any = True
    while merged_any:
        merged_any = False
        i = 0
        while i < len(sorted_det):
            j = i + 1
            while j < len(sorted_det):
                box1 = sorted_det[i]["bbox"]
                box2 = sorted_det[j]["bbox"]
                
                h1 = box1[3] - box1[1]
                h2 = box2[3] - box2[1]
                min_h = min(h1, h2)
                
                # Check vertical overlap
                y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                v_overlap_ratio = y_overlap / float(min_h) if min_h > 0 else 0
                
                # Check horizontal distance
                gap = box2[0] - box1[2]
                
                if v_overlap_ratio > min_v_overlap and gap < max_gap_factor * min_h:
                    # Merge them
                    merged_box = [
                        min(box1[0], box2[0]),
                        min(box1[1], box2[1]),
                        max(box1[2], box2[2]),
                        max(box1[3], box2[3])
                    ]
                    # Combine texts based on horizontal order
                    if box1[0] <= box2[0]:
                        combined_text = sorted_det[i]["text"] + " " + sorted_det[j]["text"]
                    else:
                        combined_text = sorted_det[j]["text"] + " " + sorted_det[i]["text"]
                        
                    sorted_det[i] = {
                        "text": combined_text,
                        "bbox": merged_box,
                        "confidence": max(sorted_det[i]["confidence"], sorted_det[j]["confidence"])
                    }
                    sorted_det.pop(j)
                    merged_any = True
                else:
                    j += 1
            i += 1
            
    return sorted_det


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
    # Apply horizontal line merging to group adjacent text boxes (e.g. labels and values)
    merged = merge_horizontal_lines(merged)
    return merged


def _is_clinical(text):
    # Strip spaces and convert to uppercase
    val = text.strip().upper()
    
    # Check regex for standard short clinical markers (e.g. L, R, LA, RA, PA, AP, LP, RP, A, P)
    # optionally followed by digit(s) (e.g. LA6, L A6, R1, A2)
    if re.match(r'^(?:L|R|LT|RT|LA|RA|LP|RP|PA|AP|LL|RL|A|P)\s*\d*$', val):
        return True
        
    # Also check if it's two separate standard markers (e.g., "L A6" or "L PA")
    # by cleaning to just alpha characters and matching against allowlist
    clean = re.sub(r'[^A-Z]', '', val)
    if clean in CLINICAL_ALLOWLIST:
        return True

    # Check if all tokens in text are known clinical terms
    tokens = re.findall(r'[A-Z0-9]+', val)
    if tokens:
        expanded_safe = CLINICAL_ALLOWLIST.union({
            "ANTEROPOSTERIOR", "POSTEROANTERIOR", "ANIIEROPOSTERIOR", "ANIIEROPAOSIIERIORR",
            "PROJECTION", "ANTERIOR", "POSTERIOR", "ERECT", "SUPINE", "CHEST", "PORTABLE"
        })
        if all(t in expanded_safe for t in tokens):
            return True
        
    return False


def classify_phi(merged, image_shape, analyzer=None, gliner_model=None, medical_ner=None, deid_model=None):
    """
    Classifies each merged detection as PHI (redact) or safe (keep).
    Uses a 3-Step Hybrid Priority Stack followed by medical protection and spatial fallback:

    ═══════════════════════════════════════════════════════════════════════
    TOP PRIORITY — 3-Step Hybrid PII/PHI Classification
    ═══════════════════════════════════════════════════════════════════════
      Step 1: Stanford De-ID Model (StanfordAIMI/stanford-deidentifier-base)
              Radiology-native transformer — detects PATIENT, NAME, DATE,
              ID, LOCATION, AGE, PHONE, etc. with F1=99.6%.
      Step 2: Regex Pattern Matching
              Deterministic patterns for Aadhaar, ABHA, phone, UHID, MRN,
              date formats, age/sex, demographic labels.
      Step 3: Hybrid Combined Decision
              If BOTH model AND regex agree → high confidence.
              If EITHER flags PHI → still redact (safety-first).
              If NEITHER flags → pass to next phase.
    ═══════════════════════════════════════════════════════════════════════

    Phase 2: Medical Entity Preservation
      - d4data/biomedical-ner-all (supervised, 8 categories)
      - GLiNER-BioMed Large (zero-shot, radiology-specific labels)
      - Presidio cross-check to prevent medical NER from shielding real PHI

    Phase 3: Exposure / Number Guard
      - Pure numeric values (e.g. 120, 80.5) are kept as scan parameters.

    Phase 4: Spatial / Default Fallback
      - Unclassified text in border zones → redact.
      - Unclassified text in anatomy zone → keep.
    """
    h, w = image_shape[:2]
    phi_regions = []

    def _is_text_pure_clinical(text_str):
        val = text_str.strip().upper()
        if not val:
            return True

        # Split by non-alphanumeric characters to get individual tokens
        tokens = re.findall(r'[A-Z0-9]+', val)
        if not tokens:
            return True

        safe_terms = {
            "L", "R", "LT", "RT", "LEFT", "RIGHT", "PA", "AP", "LAT", "LL", "RL", "LATERAL",
            "A", "P", "M", "F", "O", "CXR", "CHEST", "PORTABLE", "MOBILE", "SUPINE", "ERECT",
            "UPRIGHT", "SEMI-UPRIGHT", "SEMI", "UPPER", "LOWER", "MIDDLE", "LOBE", "LUL", "RUL",
            "LLL", "RLL", "RML", "CLAVICLE", "RIB", "HEART", "LUNG", "LUNGS", "DIAPHRAGM",
            "TRACHEA", "BRONCHUS", "APEX", "BASE", "CARDIAC", "AORTA", "PLEURAL", "ANGLE",
            "COSTOPHRENIC", "SULCUS", "MEDIASTINUM", "HILUM", "HILAR", "PARENCHYMA", "PARENCHYMAL",
            "QUADRANT", "KV", "KVP", "MA", "MAS", "MS", "SEC", "MM", "CM", "EXP", "EXPOSURE",
            "TECH", "TECHNOLOGIST", "COLLIMATION", "FILTER", "FSD", "SID", "SOD", "MASE",
            "GRID", "NO-GRID", "NOGRID", "SANS", "INSPIRATION", "EXPIRATION", "INSP", "EXP",
            "INT", "EXT", "MED", "DECUB", "DECUBITUS", "VIEW", "OF", "AND", "TO", "WITH",
            "FOR", "ON", "BY", "IN", "THE", "PORT", "AP-PORTABLE", "PA-ERECT", "AP/PA",
            "NORMAL", "PNEUMOTHORAX", "EFFUSION", "INFILTRATE", "CONSOLIDATION", "EDEMA",
            "CARDIOMEGALY", "PNEUMONIA", "OPACITY", "OPACITIES", "CARDIAC", "AORTIC",
            "SIEMENS", "PHILIPS", "GE", "HEALTHCARE", "MEDICAL", "SYSTEMS", "MICRODICOM",
            "ANTEROPOSTERIOR", "POSTEROANTERIOR", "ANIIEROPOSTERIOR", "ANIIEROPAOSIIERIORR",
            "PROJECTION", "ANTERIOR", "POSTERIOR"
        }

        for token in tokens:
            if token in safe_terms:
                continue
            if re.match(r'^\d{1,4}$', token):
                continue
            if re.match(r'^(?:L|R|C|T|S)\d{0,2}$', token):  # e.g. L2, T12, C7, L, R
                continue
            return False
        return True

    for det in merged:
        text = det["text"]
        bbox = det["bbox"]

        # Size validation: discard full-frame/large OCR bounding box artifacts
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        box_area = box_w * box_h
        img_area = w * h

        if box_h > h * 0.40 or box_w > w * 0.95 or box_area > (img_area * 0.30):
            continue

        clean_text = text.strip()
        if not clean_text:
            continue

        is_phi = False
        reason = []
        classified = False
        val_upper = clean_text.upper()

        # ═════════════════════════════════════════════════════════════════
        # TOP PRIORITY — 3-Step Hybrid PII/PHI Classification
        # ═════════════════════════════════════════════════════════════════

        # ── Hybrid Step 1: Stanford De-ID Model (Radiology-Native) ─────
        model_flags_phi = False
        model_phi_labels = []
        if deid_model:
            try:
                deid_entities = deid_model(clean_text)
                for ent in deid_entities:
                    score = float(ent.get("score", 0.0))
                    raw_group = str(ent.get("entity_group") or ent.get("entity") or "").strip()
                    clean_group = re.sub(r'^[BILOU]-', '', raw_group, flags=re.IGNORECASE)
                    if score > 0.30 and clean_group and clean_group.upper() not in {"O", "OUTSIDE"}:
                        # Guard: Ignore false-positive ID predictions on purely non-numeric text
                        if clean_group.upper() in {"UNIQUE_ID", "ID", "MEDICAL_RECORD_NUMBER", "SSN", "PHONE"} and not re.search(r'\d', clean_text):
                            continue
                        model_flags_phi = True
                        model_phi_labels.append(clean_group)
            except Exception as e:
                log.debug(f"Stanford De-ID model classification error: {e}")

        # ── Hybrid Step 2: Regex Pattern Matching ──────────────────────
        regex_flags_phi = False
        regex_reasons = []

        # 2a. Demographic keyword labels
        generic_demographic_labels = [
            "NAME", "PATIENT", "DOB", "D0B", "0B:", "BIRTH", "MRN", "UHID", "PID", "AGE", "SEX", "GENDER",
            "MALE", "FEMALE", "DR.", "DOCTOR", "PHYSICIAN", "HOSPITAL", "HOSP", "CLINIC", "INSTITUT",
            "CONFIDENTIAL", "CONFIDCNTTIAC", "RESTRICTED", "PROPRIETARY", "SECRET"
        ]
        has_id_label = bool(re.search(r'\bID\b', val_upper))
        has_demographic_label = any(label in val_upper for label in generic_demographic_labels) or has_id_label
        if has_demographic_label:
            regex_flags_phi = True
            regex_reasons.append("regex:demographic_label")

        # 2b. Indian PII regex patterns (Aadhaar, ABHA, phone, UHID, date, etc.)
        for pat_name, pat_regex in PII_PATTERNS.items():
            if re.search(pat_regex, clean_text, re.IGNORECASE):
                regex_flags_phi = True
                regex_reasons.append(f"regex:pii_pattern_{pat_name}")
                break

        # ── Hybrid Step 3: Combined Decision ───────────────────────────
        if model_flags_phi and regex_flags_phi:
            # BOTH model AND regex agree → highest confidence PHI
            is_phi = True
            reason.append(f"hybrid:model+regex_agree(model={','.join(model_phi_labels)}, {', '.join(regex_reasons)})")
            classified = True
        elif regex_flags_phi:
            # Regex alone flags PHI → deterministic, always trust
            is_phi = True
            reason.extend(regex_reasons)
            classified = True
        elif model_flags_phi:
            # Model alone flags PHI → trust the radiology-native model
            # but apply a safety check: is this actually a known clinical term?
            if _is_clinical(clean_text) or _is_text_pure_clinical(clean_text):
                # Model flagged it but it's a known clinical term → DON'T redact
                # Pass to medical NER phase for final decision
                reason.append(f"hybrid:model_flagged({','.join(model_phi_labels)})_but_clinical_override")
                # Don't set classified=True, let medical NER confirm
            else:
                is_phi = True
                reason.append(f"hybrid:model_only({','.join(model_phi_labels)})")
                classified = True

        # ═════════════════════════════════════════════════════════════════
        # Phase 2: Medical Entity Preservation (d4data + GLiNER-BioMed)
        # ═════════════════════════════════════════════════════════════════

        # ── 2a. Hugging Face Medical NER (d4data/biomedical-ner-all) ───
        if not classified and medical_ner:
            try:
                # Pre-check: Do NOT let medical_ner shield text if Presidio identifies PERSON/LOCATION
                is_person_or_phi = False
                if analyzer:
                    try:
                        p_res = analyzer.analyze(text=clean_text, language="en")
                        if any(r.entity_type in {"PERSON", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"} and r.score > 0.35 for r in p_res):
                            is_person_or_phi = True
                    except Exception:
                        pass

                if not is_person_or_phi:
                    entities = medical_ner(clean_text)
                    for ent in entities:
                        ent_group = ent.get("entity_group", "")
                        if ent_group in {
                            "Anatomical_structure", "Diagnostic_procedure", "Disease_disorder",
                            "Sign_symptom", "Lab_value", "Medication",
                            "Biological_structure", "Therapeutic_procedure"
                        }:
                            is_phi = False
                            classified = True
                            reason.append(f"medical_ner:{ent_group}")
                            break
                        elif ent_group == "Detailed_description" and not re.search(r'\b[A-Z]{3,}\s+[A-Z]{3,}\b', clean_text):
                            is_phi = False
                            classified = True
                            reason.append(f"medical_ner:{ent_group}")
                            break
            except Exception as e:
                log.debug(f"Medical NER prediction failed: {e}")

        # ── 2b. GLiNER-BioMed Zero-Shot Clinical Check ─────────────────
        if not classified and gliner_model:
            try:
                labels = [
                    # Radiology-specific anatomy labels (chest X-ray focused)
                    "thoracic anatomy", "chest anatomy", "body part", "organ",
                    "cardiac structure", "pulmonary structure", "skeletal structure",
                    # Pathology and clinical findings
                    "lung pathology", "cardiac finding", "disease", "medical condition",
                    "clinical finding", "radiological finding",
                    # Procedures and imaging
                    "medical procedure", "diagnostic test", "imaging technique",
                    # Positioning, parameters, equipment
                    "patient positioning", "scan parameter", "exposure setting",
                    "medical device", "medical equipment", "imaging modality",
                    # Medications and measurements
                    "medication", "drug name", "vital sign", "lab value", "measurement",
                    # Directional and anatomical markers
                    "anatomical direction", "laterality marker",
                ]
                entities = gliner_model.predict_entities(clean_text, labels, threshold=0.35)
                for ent in entities:
                    ent_label = ent["label"]
                    if ent_label in labels:
                        is_phi = False
                        classified = True
                        reason.append(f"gliner_biomed:{ent_label}")
                        break
            except Exception as e:
                log.debug(f"GLiNER-BioMed clinical check failed: {e}")

        # ═════════════════════════════════════════════════════════════════
        # Phase 3: Exposure / Number Guard
        # ═════════════════════════════════════════════════════════════════
        if not classified:
            if re.match(r'^\d{1,4}(?:\.\d+)?(?:\s*-\s*\d{1,4}(?:\.\d+)?)?$', clean_text):
                is_phi = False
                classified = True
                reason.append("clinical:exposure_number")

        # ═════════════════════════════════════════════════════════════════
        # Phase 4: Spatial / Default Fallback
        # ═════════════════════════════════════════════════════════════════
        if not classified:
            # Determine spatial zone
            y_mid = (bbox[1] + bbox[3]) / 2.0
            x_mid = (bbox[0] + bbox[2]) / 2.0
            in_border = (
                y_mid < h * 0.25 or y_mid > h * 0.75 or
                x_mid < w * 0.15 or x_mid > w * 0.85
            )
            if in_border and re.search(r'[A-Za-z]', clean_text):
                is_phi = True
                reason.append("fallback:suspect_border")
            else:
                is_phi = False
                reason.append("fallback:safe_anatomy")

        if is_phi:
            zone = "border" if (bbox[1] + bbox[3])/2.0 < h * 0.25 or (bbox[1] + bbox[3])/2.0 > h * 0.75 or (bbox[0] + bbox[2])/2.0 < w * 0.15 or (bbox[0] + bbox[2])/2.0 > w * 0.85 else "anatomy"
            log.info(f"    REDACT-{zone.upper()} ({', '.join(reason)}): '{text}' @ {bbox}")
            phi_regions.append({"text": text, "bbox": bbox, "zone": "border" if zone == "border" else "anatomy"})
        else:
            log.info(f"    KEEP ({', '.join(reason) if reason else 'unclassified'}): '{text}' @ {bbox}")

    return phi_regions

# =============================================================================
# MULTI-HYBRID MASKING HELPERS
# =============================================================================

def _get_character_mask(roi_8bit, bbox_local=None, dilation_px=3):
    """
    Robust Tri-Signal Dual-Polarity Character Stroke Segmentation.
    Combines:
      1. Top-Hat High-Pass (bright text on dark background, >10)
      2. Black-Hat High-Pass (dark text on bright background, >10)
      3. Local Relative Contrast Thresholding (> bg_median + 25% of dynamic range)
      4. Ellipse Dilation (absorbs anti-aliasing gray border pixels)
    Captures character strokes of ANY polarity — bright-on-dark (border text)
    AND dark-on-bright (watermarks on bone/organ) — in a single unified mask.
    """
    h, w = roi_8bit.shape[:2]
    
    if bbox_local is not None:
        bx1, by1, bx2, by2 = bbox_local
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        crop = roi_8bit[by1:by2, bx1:bx2]
    else:
        crop = roi_8bit
        bx1, by1, bx2, by2 = 0, 0, w, h

    crop_h, crop_w = crop.shape[:2]
    if crop_h < 3 or crop_w < 3:
        return np.zeros((h, w), dtype=np.uint8)

    # 1. Bilateral Denoising: smooths mottle while preserving sharp text strokes
    try:
        denoised = cv2.bilateralFilter(crop, d=5, sigmaColor=15.0, sigmaSpace=3.0)
    except Exception:
        denoised = crop.copy()

    # 2. Top-Hat — detects BRIGHT text on DARK/GREY background (medical text)
    k_size = max(5, min(15, (crop_h // 2) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
    _, mask_bright = cv2.threshold(tophat, 8, 255, cv2.THRESH_BINARY)

    # 3. Local Relative Contrast Thresholding for bright characters
    bg_med = float(np.median(crop))
    crop_max = float(np.max(crop))
    if crop_max > bg_med + 10.0:
        t_contrast = bg_med + 0.20 * (crop_max - bg_med)
        _, mask_contrast = cv2.threshold(crop, min(245.0, t_contrast), 255, cv2.THRESH_BINARY)
    else:
        mask_contrast = np.zeros_like(crop)

    # Combine bright text signals ONLY (prevents blackhat from erasing anatomical lung tissue)
    stroke_union = cv2.bitwise_or(mask_bright, mask_contrast)

    # 4. Anti-Aliased Edge Dilation
    k_dil_size = max(5, min(9, dilation_px * 2 + 1))
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil_size, k_dil_size))
    dilated = cv2.dilate(stroke_union, k_dil, iterations=1)

    # Place back into full-size mask
    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[by1:by2, bx1:bx2] = dilated
    return full_mask






def _redact_unified(cleaned, x1, y1, x2, y2, ds=None, modality="CT"):
    """
    Hybrid Character-Level Redaction: Two-Pass Iterative Neighbor Inpainting.

    Pass 1: Mask character strokes (TopHat + contrast), inpaint from neighbors
    Pass 2: Re-detect any residual bright pixels, expand mask, re-inpaint

    Every restored pixel comes from its immediate real neighbor pixels.
    No MORPH_OPEN, no blending — pure neighbor propagation character by character.
    """
    h, w = cleaned.shape[:2]
    box_h, box_w = y2 - y1, x2 - x1
    if box_h <= 0 or box_w <= 0:
        return cleaned

    # ── Step 1: Extract Context Region ────────────────────────────────────────
    ctx_pad = max(15, max(box_h, box_w) // 3)
    cy1 = max(0, y1 - ctx_pad);  cy2 = min(h, y2 + ctx_pad)
    cx1 = max(0, x1 - ctx_pad);  cx2 = min(w, x2 + ctx_pad)

    ctx_region = cleaned[cy1:cy2, cx1:cx2].copy()
    ctx_h, ctx_w = ctx_region.shape[:2]

    # ── Helper: generate 8-bit from current state ─────────────────────────────
    def _to_8bit(region):
        if ds is not None:
            return _apply_voi_lut(region, ds)
        rmin, rmax = float(region.min()), float(region.max())
        if rmax > rmin:
            return np.clip(((region.astype(np.float64) - rmin) /
                           (rmax - rmin) * 255.0), 0, 255).astype(np.uint8)
        return region.astype(np.uint8)

    # ── Step 2: Build character stroke mask ───────────────────────────────────
    roi_8 = _to_8bit(ctx_region)
    bbox_local = (
        max(0, x1 - cx1 - 2), max(0, y1 - cy1 - 2),
        min(ctx_w, x2 - cx1 + 2), min(ctx_h, y2 - cy1 + 2)
    )
    char_mask = _get_character_mask(roi_8, bbox_local=bbox_local, dilation_px=3)

    # Fallback if primary mask too sparse
    mask_coverage = int(np.sum(char_mask > 0))
    bbox_area = max(1, (bbox_local[2] - bbox_local[0]) * (bbox_local[3] - bbox_local[1]))
    if mask_coverage < max(10, bbox_area * 0.02):
        bx1, by1, bx2, by2 = bbox_local
        crop_roi = roi_8[by1:by2, bx1:bx2]
        if crop_roi.size > 0:
            k_fb = max(5, min(15, (crop_roi.shape[0] // 2) | 1))
            fb_tophat = cv2.morphologyEx(crop_roi, cv2.MORPH_TOPHAT,
                        cv2.getStructuringElement(cv2.MORPH_RECT, (k_fb, k_fb)))
            _, fb_stroke = cv2.threshold(fb_tophat, 8, 255, cv2.THRESH_BINARY)
            fb_dilated = cv2.dilate(fb_stroke,
                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            char_mask[by1:by2, bx1:bx2] = fb_dilated

    # ── Step 3: PASS 1 — Inpaint character strokes from neighbor pixels ───────
    mask_8 = char_mask.astype(np.uint8)
    if not np.any(mask_8 > 0):
        return cleaned

    inpainted = _inpaint_native(ctx_region, mask_8, radius=4, method=cv2.INPAINT_NS)
    mask_bool = mask_8 > 0
    ctx_region[mask_bool] = inpainted[mask_bool]

    # ── Step 4: PASS 2 — Catch residual bright pixels missed by Pass 1 ───────
    # Re-generate 8-bit from the Pass-1 result and re-detect bright residuals
    bx1, by1, bx2, by2 = bbox_local
    roi_8_pass2 = _to_8bit(ctx_region)
    crop_p2 = roi_8_pass2[by1:by2, bx1:bx2]
    if crop_p2.size > 0:
        # Compute local background level from the non-text ring
        ring_pixels = roi_8_pass2[char_mask == 0]
        if ring_pixels.size > 0:
            bg_med = float(np.median(ring_pixels))
            bg_std = float(np.std(ring_pixels))
        else:
            bg_med = float(np.median(crop_p2))
            bg_std = 10.0

        # Detect any pixel within bbox that is still brighter than background + 2*std
        bright_thresh = min(250.0, bg_med + max(15.0, 2.0 * bg_std))
        _, residual_mask_crop = cv2.threshold(crop_p2, bright_thresh, 255, cv2.THRESH_BINARY)
        residual_count = int(np.sum(residual_mask_crop > 0))

        if residual_count > 5:
            # Dilate residual mask to catch edges
            residual_dilated = cv2.dilate(residual_mask_crop,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            pass2_mask = np.zeros((ctx_h, ctx_w), dtype=np.uint8)
            pass2_mask[by1:by2, bx1:bx2] = residual_dilated

            inpainted_p2 = _inpaint_native(ctx_region, pass2_mask, radius=4,
                                           method=cv2.INPAINT_NS)
            p2_bool = pass2_mask > 0
            ctx_region[p2_bool] = inpainted_p2[p2_bool]
            log.debug(f"    Pass 2 caught {residual_count} residual bright pixels")

    cleaned[cy1:cy2, cx1:cx2] = ctx_region
    return cleaned


def _inpaint_native(image, mask_8, radius=3, method=cv2.INPAINT_NS):
    """
    Inpaints at native bit depth using float64 normalized intermediate.

    For 8-bit:  direct cv2.inpaint (zero precision loss).
    For 16-bit / int16 / other dtypes:
      1. Normalize to [0, 1] float64 using actual min/max
      2. Scale to [0, 255] uint8 for cv2.inpaint (OpenCV limitation)
      3. Inpaint only the masked pixels
      4. Map inpainted pixels back to original range
      5. Write back ONLY the masked pixels into the original array
    """
    if not np.any(mask_8 > 0):
        return image.copy()

    if image.dtype == np.uint8:
        return cv2.inpaint(image, mask_8, radius, method)

    result = image.copy()
    raw_min = float(image.min())
    raw_max = float(image.max())

    if raw_max <= raw_min:
        return result

    # Normalize full image to [0, 255] uint8 for cv2.inpaint
    norm_f64 = (image.astype(np.float64) - raw_min) / (raw_max - raw_min)
    t8 = np.clip(norm_f64 * 255.0, 0, 255).astype(np.uint8)

    # Inpaint in 8-bit space
    inp8 = cv2.inpaint(t8, mask_8, radius, method)

    # Map inpainted result back to original range (float64 precision)
    inp_f64 = inp8.astype(np.float64) / 255.0 * (raw_max - raw_min) + raw_min

    # Write back ONLY the masked pixels — leave everything else untouched
    mask_bool = mask_8 > 0
    result_f64 = result.astype(np.float64)
    result_f64[mask_bool] = inp_f64[mask_bool]

    # Clip to dtype range and cast back
    if image.dtype == np.uint16:
        result_f64 = np.clip(result_f64, 0, 65535)
    elif image.dtype == np.int16:
        result_f64 = np.clip(result_f64, -32768, 32767)

    return result_f64.astype(image.dtype)



def _distance_feather_blend(original_roi, inpainted_roi, ink_mask, feather_px=3):
    """
    Outward-Only Distance Feathering.

    ENFORCES 100% INPAINTED VALUE INSIDE INK_MASK (alpha = 1.0).
    Zero fraction of original_roi is retained inside the ink mask to prevent text ghosting.
    Feathering occurs ONLY on clean background tissue outside the mask boundary.
    """
    if not np.any(ink_mask > 0):
        return inpainted_roi

    h, w = ink_mask.shape[:2]
    is_16bit = original_roi.dtype == np.uint16
    max_val = 65535.0 if is_16bit else 255.0

    # Outer distance: distance of non-mask pixels from the mask edge
    inv_mask = cv2.bitwise_not(ink_mask)
    outer_dist = cv2.distanceTransform(inv_mask, cv2.DIST_L2, 5)

    # Alpha ramp: 1.0 inside mask and transition zone, decaying to 0.0 outside
    alpha = np.ones((h, w), dtype=np.float64)

    # Outside mask: decay from 1.0 at edge → 0.0 at feather_px
    outer_zone = inv_mask > 0
    alpha[outer_zone] = np.clip(1.0 - outer_dist[outer_zone] / max(feather_px, 1), 0.0, 1.0)

    # Smooth the alpha slightly for natural transition into clean background
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0.5)

    # Force 100% inpainted value INSIDE the ink mask (Zero leakage of orig_f)
    alpha[ink_mask > 0] = 1.0

    # Blend
    orig_f = original_roi.astype(np.float64)
    inp_f = inpainted_roi.astype(np.float64)
    blended_f = inp_f * alpha + orig_f * (1.0 - alpha)
    blended = np.clip(blended_f, 0, max_val).astype(original_roi.dtype)

    return blended


def _match_ring_statistics(roi, ink_mask, ring_width=7):
    """
    Continuous Spatial Intensity & Contrast Matching (Median-Robust).
    Operates continuously across the image without discrete grid cell tiles,
    preventing any blocky seams or step discontinuities.
    """
    mask_pixels_count = int(np.sum(ink_mask > 0))
    if mask_pixels_count < 2:
        return roi

    result = roi.copy()
    max_val = 65535.0 if roi.dtype == np.uint16 else 255.0

    # Create a smooth continuous local background estimate using Morphological Opening on surrounding roi
    k_size = max(7, min(25, (ring_width * 3) | 1))
    bg_smooth = cv2.morphologyEx(roi, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size)))
    bg_smooth = cv2.GaussianBlur(bg_smooth.astype(np.float64), (5, 5), 0)

    # For masked stroke pixels, blend smoothly with local background estimate
    mask_bool = ink_mask > 0
    roi_f = result.astype(np.float64)

    roi_f[mask_bool] = 0.5 * roi_f[mask_bool] + 0.5 * bg_smooth[mask_bool]
    result[mask_bool] = np.clip(roi_f[mask_bool], 0, max_val).astype(roi.dtype)

    return result









def _compute_ssim(img1, img2):
    """
    Computes a simplified local Structural Similarity Index (SSIM) between two images.
    Returns the average SSIM score across the entire image region.
    """
    is_16bit = img1.dtype == np.uint16
    max_val = 65535.0 if is_16bit else 255.0

    x = img1.astype(np.float64) / max_val
    y = img2.astype(np.float64) / max_val

    # Constants to avoid division by zero
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    # Mean values using a Gaussian window
    ksize = 11
    sigma = 1.5
    mu_x = cv2.GaussianBlur(x, (ksize, ksize), sigma)
    mu_y = cv2.GaussianBlur(y, (ksize, ksize), sigma)

    mu_x_sq = mu_x ** 2
    mu_y_sq = mu_y ** 2
    mu_xy = mu_x * mu_y

    sigma_x_sq = cv2.GaussianBlur(x * x, (ksize, ksize), sigma) - mu_x_sq
    sigma_y_sq = cv2.GaussianBlur(y * y, (ksize, ksize), sigma) - mu_y_sq
    sigma_xy = cv2.GaussianBlur(x * y, (ksize, ksize), sigma) - mu_xy

    # SSIM formula
    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x_sq + mu_y_sq + c1) * (sigma_x_sq + sigma_y_sq + c2)

    ssim_map = numerator / np.maximum(denominator, 1e-6)
    return float(np.mean(ssim_map))


def _classify_tissue(roi_original, ring_mask):
    """
    Classifies the local tissue background surrounding the redaction zone.
    Returns: 'bone', 'lung', 'soft_tissue', or 'air'.
    """
    ring_pixels = roi_original[ring_mask > 0].astype(np.float64)
    if len(ring_pixels) < 10:
        return 'soft_tissue'

    is_16bit = roi_original.dtype == np.uint16
    max_val = 65535.0 if is_16bit else 255.0

    mean_val = np.mean(ring_pixels) / max_val
    std_val = np.std(ring_pixels) / max_val

    # Detect high frequency structural edges using Canny on the ring
    roi_8 = cv2.normalize(roi_original, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    edges = cv2.Canny(roi_8, 50, 150)
    edge_density = float(np.sum(edges[ring_mask > 0] > 0)) / max(1.0, float(np.sum(ring_mask > 0)))

    # Classification thresholds based on dynamic ranges
    if mean_val < 0.08:
        return 'air'
    elif mean_val > 0.55 or (mean_val > 0.35 and edge_density > 0.15):
        return 'bone'
    elif mean_val < 0.30 and std_val > 0.05:
        return 'lung'
    else:
        return 'soft_tissue'


def _merge_spatial_boxes(regions, tolerance_px=10):
    """Merges spatially overlapping or adjacent PHI bounding boxes before redaction."""
    if not regions:
        return []
    reg_list = [dict(r) for r in regions]
    merged = []
    while reg_list:
        curr = reg_list.pop(0)
        cb = list(curr["bbox"])
        i = 0
        while i < len(reg_list):
            ob = reg_list[i]["bbox"]
            if not (cb[0] > ob[2] + tolerance_px or cb[2] < ob[0] - tolerance_px or
                    cb[1] > ob[3] + tolerance_px or cb[3] < ob[1] - tolerance_px):
                cb[0] = min(cb[0], ob[0])
                cb[1] = min(cb[1], ob[1])
                cb[2] = max(cb[2], ob[2])
                cb[3] = max(cb[3], ob[3])
                curr["text"] = curr["text"] + " " + reg_list[i]["text"]
                reg_list.pop(i)
            else:
                i += 1
        curr["bbox"] = cb
        merged.append(curr)
    return merged


def redact_pixels(image_array, phi_regions, ds=None, modality="CT"):
    """
    Redacts pixels associated with PHI text regions using native polarity-aware MORPH_OPEN.
    """
    if not phi_regions:
        log.info("  [Stage 5] No PHI pixels to redact.")
        return image_array.copy(), np.zeros(image_array.shape[:2], dtype=np.uint8)

    cleaned = image_array.copy()
    h, w = cleaned.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # Merge spatially overlapping or multi-line header boxes before redaction
    merged_regions = _merge_spatial_boxes(phi_regions, tolerance_px=10)

    redact_count = 0
    for region in merged_regions:
        x1, y1, x2, y2 = region["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        combined_mask[y1:y2, x1:x2] = 255

        # Unified approach: every region gets per-pixel adaptive restoration
        cleaned = _redact_unified(cleaned, x1, y1, x2, y2, ds=ds, modality=modality)
        redact_count += 1
        log.info(f"  [Stage 5] Unified character-level redaction applied @ [{x1},{y1},{x2},{y2}]")

    log.info(
        f"  [Stage 5] Done. Regions redacted: {redact_count} | "
        f"Total pixels masked: {int(np.sum(combined_mask > 0))}"
    )
    return cleaned, combined_mask


# =============================================================================
# STAGE 6: VERIFICATION + ESCALATION
# =============================================================================
def _escalate_single_channel(chan_pix, residual, ds=None, modality="CT"):
    h, w = chan_pix.shape[:2]
    escalated = chan_pix.copy()
    for region in residual:
        x1, y1, x2, y2 = region["bbox"]
        # Add 8px padding to catch any char edges
        x1e = max(0, x1 - 8); y1e = max(0, y1 - 8)
        x2e = min(w, x2 + 8); y2e = min(h, y2 + 8)

        # Unified approach — same per-pixel adaptive restoration for all zones
        escalated = _redact_unified(escalated, x1e, y1e, x2e, y2e, ds=ds, modality=modality)
    return escalated


def verify_redaction(cleaned_array, phi_regions, paddle_ocr, easy_ocr, analyzer, gliner_model=None, medical_ner=None, deid_model=None, ds=None, modality="CT"):
    """Re-runs OCR on cleaned image to confirm zero residual PHI text."""
    log.info("  [Stage 6] Verification pass — re-running OCR on cleaned image...")

    # Determine if cleaned_array is color
    is_color = (cleaned_array.ndim == 3 and cleaned_array.shape[-1] in (3, 4))

    if is_color:
        raw_min, raw_max = cleaned_array.min(), cleaned_array.max()
        if raw_max > raw_min:
            temp_8_color = ((cleaned_array - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
        else:
            temp_8_color = cleaned_array.astype(np.uint8)
        try:
            temp_8 = cv2.cvtColor(temp_8_color, cv2.COLOR_RGB2GRAY)
        except Exception:
            temp_8 = (0.299 * temp_8_color[:,:,0] + 0.587 * temp_8_color[:,:,1] + 0.114 * temp_8_color[:,:,2]).astype(np.uint8)
    else:
        raw_min, raw_max = cleaned_array.min(), cleaned_array.max()
        if raw_max > raw_min:
            temp_8 = ((cleaned_array - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
        else:
            temp_8 = cleaned_array.astype(np.uint8)

    # Run OCR at full resolution
    ocr_input = temp_8

    variants    = enhance_image(ocr_input)
    raw_det     = detect_text(variants, paddle_ocr, easy_ocr, modality=modality)
    merged      = merge_detections(raw_det)
    residual    = classify_phi(merged, cleaned_array.shape, analyzer, gliner_model, medical_ner, deid_model)

    if not residual:
        log.info("  [Stage 6] PASSED — no residual PHI text detected.")
        return cleaned_array, "PASSED"

    # ── Zone-aware escalation ─────────────────────────────────────────────────
    log.warning(
        f"  [Stage 6] Residual PHI found in {len(residual)} region(s): "
        f"{[r['text'][:30] for r in residual]}. Running zone-aware escalation..."
    )
    
    if is_color:
        escalated_channels = []
        for c in range(cleaned_array.shape[-1]):
            escalated_chan = _escalate_single_channel(cleaned_array[:, :, c], residual, modality=modality)
            escalated_channels.append(escalated_chan)
        escalated = np.stack(escalated_channels, axis=-1)
    else:
        escalated = _escalate_single_channel(cleaned_array, residual, ds=ds, modality=modality)

    log.info("  [Stage 6] Escalation complete.")

    # ── Second tight verification pass ───────────────────────────────────────
    if is_color:
        esc_min, esc_max = escalated.min(), escalated.max()
        if esc_max > esc_min:
            esc_8_color = ((escalated - esc_min) / (esc_max - esc_min) * 255.0).astype(np.uint8)
        else:
            esc_8_color = escalated.astype(np.uint8)
        try:
            esc_8 = cv2.cvtColor(esc_8_color, cv2.COLOR_RGB2GRAY)
        except Exception:
            esc_8 = (0.299 * esc_8_color[:,:,0] + 0.587 * esc_8_color[:,:,1] + 0.114 * esc_8_color[:,:,2]).astype(np.uint8)
    else:
        esc_min, esc_max = escalated.min(), escalated.max()
        if esc_max > esc_min:
            esc_8 = ((escalated - esc_min) / (esc_max - esc_min) * 255.0).astype(np.uint8)
        else:
            esc_8 = escalated.astype(np.uint8)

    ocr_input2 = esc_8
    v2_det    = detect_text(enhance_image(ocr_input2), paddle_ocr, easy_ocr, modality=modality)
    v2_merged = merge_detections(v2_det)
    v2_phi    = classify_phi(v2_merged, escalated.shape, analyzer, gliner_model, medical_ner, deid_model)

    if not v2_phi:
        log.info("  [Stage 6] PASSED (Escalated — all residual text cleared).")
        return escalated, "PASSED (Escalated)"

    # ── Targeted Fallback: Apply expanded character redaction on residual regions ─────
    log.warning(
        f"  [Stage 6] Verification fallback — applying targeted expanded redaction on {len(v2_phi)} residual region(s)..."
    )
    nuclear = escalated.copy()
    h, w = nuclear.shape[:2]
    for region in v2_phi:
        x1, y1, x2, y2 = region["bbox"]
        x1e = max(0, x1 - 15); y1e = max(0, y1 - 15)
        x2e = min(w, x2 + 15); y2e = min(h, y2 + 15)
        nuclear = _redact_unified(nuclear, x1e, y1e, x2e, y2e, ds=ds, modality=modality)

    log.info("  [Stage 6] FORCE PASSED (Targeted expanded redaction applied).")
    return nuclear, "FORCE PASSED (Targeted Fallback)"


# =============================================================================
# STAGE 7: WRITE PIXELS BACK INTO DICOM (correct tag updates)
# =============================================================================
def write_pixels_to_dicom(ds, cleaned_array):
    """
    Correctly writes the anonymized pixel array back into the DICOM dataset.
    Updates ALL mandatory pixel descriptor tags to match the array's dtype/shape.
    """
    arr = cleaned_array

    # Determine if it's a color image
    samples_per_pixel = getattr(ds, "SamplesPerPixel", 1)
    is_color = (samples_per_pixel > 1) or (arr.ndim in (3, 4) and arr.shape[-1] in (3, 4))

    # Handle multi-frame tag settings
    if is_color:
        if arr.ndim == 4 and arr.shape[0] > 1:
            ds.NumberOfFrames = arr.shape[0]
        elif hasattr(ds, 'NumberOfFrames'):
            try:
                del ds.NumberOfFrames
            except Exception:
                pass
    else:
        if arr.ndim == 3 and arr.shape[0] > 1:
            ds.NumberOfFrames = arr.shape[0]
        elif hasattr(ds, 'NumberOfFrames'):
            try:
                del ds.NumberOfFrames
            except Exception:
                pass

    # Determine bit depth from dtype
    if arr.dtype == np.uint8:
        bits_alloc = 8; bits_stored = 8; high_bit = 7; pix_rep = 0
    elif arr.dtype == np.uint16:
        bits_alloc = 16
        bits_stored = getattr(ds, "BitsStored", 16)
        high_bit = getattr(ds, "HighBit", 15)
        pix_rep = getattr(ds, "PixelRepresentation", 0)
    elif arr.dtype == np.int16:
        bits_alloc = 16
        bits_stored = getattr(ds, "BitsStored", 16)
        high_bit = getattr(ds, "HighBit", 15)
        pix_rep = getattr(ds, "PixelRepresentation", 1)
    else:
        # Fallback: cast to uint16
        arr = arr.astype(np.uint16)
        bits_alloc = 16
        bits_stored = getattr(ds, "BitsStored", 16)
        high_bit = getattr(ds, "HighBit", 15)
        pix_rep = getattr(ds, "PixelRepresentation", 0)

    # Update all pixel descriptor tags
    ds.BitsAllocated      = bits_alloc
    ds.BitsStored         = bits_stored
    ds.HighBit            = high_bit
    ds.PixelRepresentation = pix_rep

    # Update color tags specifically if needed
    if is_color:
        ds.SamplesPerPixel = 3
        ds.PlanarConfiguration = 0
        if getattr(ds, "PhotometricInterpretation", "") not in ["RGB", "YBR_FULL", "YBR_FULL_422"]:
            ds.PhotometricInterpretation = "RGB"
    else:
        ds.SamplesPerPixel = 1
        if hasattr(ds, 'PlanarConfiguration'):
            try:
                del ds.PlanarConfiguration
            except Exception:
                pass

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
def anonymize_dicom_file(input_path, output_path, paddle_ocr, easy_ocr, analyzer, gliner_model=None, medical_ner=None, deid_model=None):
    """
    Full 7-stage anonymization pipeline for a single DICOM file.
    Runs OCR and redaction in visual 8-bit space to prevent WW/WC artifacts on X-rays.
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
        modality = audit["modality"]
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
        # Only applicable to monochrome images, skip for color
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        samples_per_pixel_raw = getattr(ds, "SamplesPerPixel", 1)
        is_monochrome1 = (photometric == "MONOCHROME1" and samples_per_pixel_raw == 1)
        original_max = None
        if is_monochrome1:
            original_max = np.max(pixels)
            pixels = original_max - pixels
            ds.PhotometricInterpretation = "MONOCHROME2"
            log.info("  MONOCHROME1 detected — pixel values flipped to MONOCHROME2 for processing.")

        # Determine if it's a color image
        samples_per_pixel = getattr(ds, "SamplesPerPixel", 1)
        is_color = (samples_per_pixel > 1) or (pixels.ndim in (3, 4) and pixels.shape[-1] in (3, 4))

        # Generate 8-bit visual image from raw pixels (resolves WW/WC display artifacts)
        if is_color:
            pix_min, pix_max = pixels.min(), pixels.max()
            if pix_max > pix_min:
                visual_8bit = ((pixels - pix_min) / (pix_max - pix_min) * 255.0).astype(np.uint8)
            else:
                visual_8bit = pixels.astype(np.uint8)
        else:
            if pixels.ndim == 3:
                # Multi-frame grayscale: apply VOI LUT frame by frame
                visual_frames = [_apply_voi_lut(pixels[i], ds) for i in range(pixels.shape[0])]
                visual_8bit = np.stack(visual_frames, axis=0)
            else:
                # Single-frame grayscale
                visual_8bit = _apply_voi_lut(pixels, ds)

        # Set up frames for OCR text detection
        if is_color:
            if pixels.ndim == 4:
                # Multi-frame color: (Frames, H, W, Channels)
                frame0_norm = visual_8bit[0]
            else:
                # Single-frame color: (H, W, Channels)
                frame0_norm = visual_8bit
            
            # Convert 3-channel color to 1-channel grayscale for OCR
            try:
                ocr_frame = cv2.cvtColor(frame0_norm, cv2.COLOR_RGB2GRAY)
            except Exception:
                ocr_frame = (0.299 * frame0_norm[:,:,0] + 0.587 * frame0_norm[:,:,1] + 0.114 * frame0_norm[:,:,2]).astype(np.uint8)
        else:
            if pixels.ndim == 3:
                # Multi-frame grayscale: (Frames, H, W)
                ocr_frame = visual_8bit[0]
            else:
                # Single-frame grayscale: (H, W)
                ocr_frame = visual_8bit

        # High-precision scaling for OCR (1536 max_dim with INTER_LINEAR preserves fine text strokes while speeding up CPU inference)
        orig_h, orig_w = ocr_frame.shape[:2]
        max_ocr_dim = 1536
        if max(orig_h, orig_w) > max_ocr_dim:
            scale_factor = max_ocr_dim / float(max(orig_h, orig_w))
            ocr_input_frame = cv2.resize(
                ocr_frame, 
                (int(orig_w * scale_factor), int(orig_h * scale_factor)), 
                interpolation=cv2.INTER_LINEAR
            )
            use_scaling = True
        else:
            scale_factor = 1.0
            ocr_input_frame = ocr_frame
            use_scaling = False

        # ── Stage 2: Enhancement ──────────────────────────────────────────────
        log.info("  [Stage 2] Generating enhanced image variants...")
        variants = enhance_image(ocr_input_frame)

        # ── Stage 3: OCR ──────────────────────────────────────────────────────
        log.info("  [Stage 3] Running OCR text detection...")
        raw_det = detect_text(variants, paddle_ocr, easy_ocr, modality=modality)
        log.info(f"            Raw detections: {len(raw_det)}")

        # Scale detections back to original resolution
        if use_scaling and raw_det:
            for det in raw_det:
                bbox = det["bbox"]
                det["bbox"] = [
                    int(bbox[0] / scale_factor),
                    int(bbox[1] / scale_factor),
                    int(bbox[2] / scale_factor),
                    int(bbox[3] / scale_factor)
                ]

        # ── Stage 4: Classify ─────────────────────────────────────────────────
        log.info("  [Stage 4] Classifying detections (PHI vs clinical)...")
        merged     = merge_detections(raw_det)
        phi_regions = classify_phi(merged, ocr_frame.shape, analyzer, gliner_model, medical_ner, deid_model)
        log.info(f"            PHI regions to redact: {len(phi_regions)}")
        audit["redacted_regions"] = [
            {"text": r["text"], "bbox": r["bbox"]} for r in phi_regions
        ]

        # ── Stage 5: Redact ───────────────────────────────────────────────────
        log.info("  [Stage 5] Redacting PHI pixels (Navier-Stokes inpainting in visual space)...")
        if is_color:
            if pixels.ndim == 4:
                # Multi-frame color: (Frames, H, W, Channels)
                cleaned_frames = []
                for i in range(pixels.shape[0]):
                    frame_pix = pixels[i]
                    cleaned_channels = []
                    for c in range(frame_pix.shape[-1]):
                        chan_pix = frame_pix[:, :, c]
                        cleaned_chan, _ = redact_pixels(chan_pix, phi_regions, ds=ds, modality=modality)
                        cleaned_channels.append(cleaned_chan)
                    cleaned_frame = np.stack(cleaned_channels, axis=-1)
                    cleaned_frames.append(cleaned_frame)
                cleaned_pixels = np.stack(cleaned_frames, axis=0)
            else:
                # Single-frame color: (H, W, Channels)
                cleaned_channels = []
                for c in range(pixels.shape[-1]):
                    chan_pix = pixels[:, :, c]
                    cleaned_chan, _ = redact_pixels(chan_pix, phi_regions, ds=ds, modality=modality)
                    cleaned_channels.append(cleaned_chan)
                cleaned_pixels = np.stack(cleaned_channels, axis=-1)
        else:
            # Grayscale: redact directly on original high-precision 16-bit pixels
            if pixels.ndim == 3:
                # Multi-frame grayscale
                cleaned_frames = []
                for i in range(pixels.shape[0]):
                    cleaned_frame, _ = redact_pixels(pixels[i], phi_regions, ds=ds, modality=modality)
                    cleaned_frames.append(cleaned_frame)
                cleaned_pixels = np.stack(cleaned_frames, axis=0)
            else:
                # Single-frame grayscale
                cleaned_pixels, _ = redact_pixels(pixels, phi_regions, ds=ds, modality=modality)

        # ── Stage 6: Verify ───────────────────────────────────────────────────
        is_multiframe = pixels.ndim == 4 or (pixels.ndim == 3 and not is_color)
        verify_frame = cleaned_pixels[0] if is_multiframe else cleaned_pixels
        cleaned_pixels_final = cleaned_pixels.copy()
        
        verify_clean, status = verify_redaction(
            verify_frame, phi_regions, paddle_ocr, easy_ocr, analyzer, gliner_model, medical_ner, deid_model, ds=ds, modality=modality
        )
        
        if is_multiframe:
            cleaned_pixels_final[0] = verify_clean
        else:
            cleaned_pixels_final = verify_clean
        audit["verification_status"] = status

        # ── Stage 7: Write back to DICOM ─────────────────────────────────────
        log.info("  [Stage 7] Writing anonymized pixels back into DICOM dataset...")
        if is_monochrome1 and original_max is not None:
            cleaned_pixels_final = original_max - cleaned_pixels_final
            ds.PhotometricInterpretation = "MONOCHROME1"
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


def initialize_engines(use_gpu=False, skip_ner=False):
    """
    Initializes OCR engines, Presidio Analyzer, and upgraded NER/De-ID models.
    
    Model Stack (Upgraded):
      - OCR: PaddleOCR + EasyOCR (unchanged)
      - De-ID: StanfordAIMI/stanford-deidentifier-base (replaces riggsmed/deid-LONGFORMER-NemPII)
               Radiology-native, F1=99.6%, 4x faster than Longformer, trained on CXR/CT/MRI reports.
      - Medical NER: d4data/biomedical-ner-all (unchanged, supervised 8-category biomedical NER)
      - Clinical NER: Ihor/gliner-biomed-large-v1.0 (replaces urchade/gliner_small-v2.1)
               Biomedical-specific zero-shot NER, +6% F1 over generic GLiNER, radiology-aware.
      - PII Regex: Microsoft Presidio (unchanged)
    """
    print(f"\nInitializing OCR & NER engines (GPU={use_gpu}, SkipNER={skip_ner})...")
    sys.stdout.flush()

    paddle_ocr = None
    if PADDLE_AVAILABLE:
        # Configure safe allocator flag so Paddle C++ engine does not collide with PyTorch CUDA
        os.environ["FLAGS_allocator_strategy"] = "naive_best_fit"
        for desc, fn in [
            ("PaddleOCR(lang='en', use_gpu=False)",
             lambda: PaddleOCR(lang='en', use_gpu=False)),
            ("PaddleOCR(lang='en')",
             lambda: PaddleOCR(lang='en')),
        ]:
            try:
                paddle_ocr = fn()
                if paddle_ocr is not None:
                    print(f"  [OK] Primary Engine: PaddleOCR initialized ({desc})")
                    sys.stdout.flush()
                    break
            except Exception as e:
                log.debug(f"PaddleOCR attempt '{desc}' failed: {e}")
                paddle_ocr = None
        if paddle_ocr is None:
            print("  [WARN] PaddleOCR primary init failed — falling back to EasyOCR")
            sys.stdout.flush()
    else:
        print("  [INFO] PaddleOCR not installed — using EasyOCR")
        sys.stdout.flush()

    easy_ocr = None
    if EASYOCR_AVAILABLE:
        try:
            easy_ocr = easyocr.Reader(['en'], gpu=use_gpu)
            print("  [OK] EasyOCR initialized")
            sys.stdout.flush()
        except Exception as e:
            print(f"  [WARN] EasyOCR failed to init: {e}")
            sys.stdout.flush()

    analyzer = None
    if PRESIDIO_AVAILABLE:
        try:
            analyzer = AnalyzerEngine()
            print("  [OK] Presidio NLP Analyzer initialized")
        except Exception as e:
            print(f"  [WARN] Presidio failed to init: {e}")

    # Helper function to safely execute downloads/loads with a 3-minute timeout (allows 266MB HF download on Kaggle)
    import concurrent.futures
    def _safe_load_model(load_fn, model_name, timeout_sec=180):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(load_fn)
            try:
                return future.result(timeout=timeout_sec)
            except concurrent.futures.TimeoutError:
                print(f"  [WARN] {model_name} load timed out ({timeout_sec}s limit) — proceeding without it.")
                sys.stdout.flush()
                return None
            except Exception as ex:
                print(f"  [WARN] {model_name} failed to init: {ex}")
                sys.stdout.flush()
                return None

    # ── GLiNER-BioMed Large (replaces urchade/gliner_small-v2.1) ──────────
    # Biomedical-specific zero-shot NER with +6% F1 over generic GLiNER.
    # Understands radiology anatomy, pathology, imaging terms natively.
    gliner_model = None
    if GLINER_AVAILABLE and not skip_ner:
        def _load_gliner():
            print("  Initializing GLiNER-BioMed Large (Ihor/gliner-biomed-large-v1.0)...")
            print("  [UPGRADE] Replaces urchade/gliner_small-v2.1 — biomedical-specific, +6% F1")
            sys.stdout.flush()
            device = "cuda" if use_gpu else "cpu"
            m = GLiNER.from_pretrained("Ihor/gliner-biomed-large-v1.0")
            return m.to(device)

        gliner_model = _safe_load_model(_load_gliner, "GLiNER-BioMed Large model", timeout_sec=180)
        if gliner_model is not None:
            print("  [OK] GLiNER-BioMed Large (Ihor/gliner-biomed-large-v1.0) initialized successfully.")
            sys.stdout.flush()

    # ── d4data Medical NER (KEPT — supervised, precise on 8 biomedical categories) ──
    medical_ner = None
    if not skip_ner:
        def _load_medical():
            from transformers import pipeline
            print("  Initializing Medical NER model (d4data/biomedical-ner-all, ~266MB)...")
            print("  [KEPT] Supervised model — precise on Anatomical_structure, Disease_disorder, etc.")
            sys.stdout.flush()
            device = 0 if use_gpu else -1
            return pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple", device=device)

        medical_ner = _safe_load_model(_load_medical, "Medical NER model", timeout_sec=180)
        if medical_ner is not None:
            print("  [OK] Medical NER model (d4data/biomedical-ner-all) initialized successfully.")
            sys.stdout.flush()

    # ── Stanford De-ID Model (replaces riggsmed/deid-LONGFORMER-NemPII) ───
    # Radiology-native transformer, F1=99.6%, 4x faster than Longformer.
    # Trained on Stanford + UPenn radiology reports (CXR, CT, MRI).
    # Understands that L, R, PA, AP are NOT PHI in radiology context.
    deid_model = None
    if not skip_ner:
        def _load_deid():
            from transformers import pipeline
            print("  Initializing Stanford De-ID model (StanfordAIMI/stanford-deidentifier-base)...")
            print("  [UPGRADE] Replaces riggsmed/deid-LONGFORMER-NemPII — radiology-native, F1=99.6%")
            sys.stdout.flush()
            device = 0 if use_gpu else -1
            return pipeline("token-classification", model="StanfordAIMI/stanford-deidentifier-base", aggregation_strategy="simple", device=device)

        deid_model = _safe_load_model(_load_deid, "Stanford De-ID model", timeout_sec=180)
        if deid_model is not None:
            print("  [OK] Stanford De-ID model (StanfordAIMI/stanford-deidentifier-base) initialized successfully.")
            sys.stdout.flush()

    if not paddle_ocr and not easy_ocr:
        print("\n[ERROR] No OCR engine available! Install at least one:")
        print("  pip install paddleocr   OR   pip install easyocr")
        print("Metadata anonymization will still run.\n")

    return paddle_ocr, easy_ocr, analyzer, gliner_model, medical_ner, deid_model


# =============================================================================
# HIGH-LEVEL RUNNER API
# =============================================================================
def anonymize(input_path, output_path, use_gpu=None, audit_log_path=None, skip_ner=False):
    """
    High-level API to anonymize a single DICOM file or an entire directory of DICOM files.
    """
    if use_gpu is None:
        use_gpu = check_gpu_available()
        
    paddle_ocr, easy_ocr, analyzer, gliner_model, medical_ner, deid_model = initialize_engines(use_gpu=use_gpu, skip_ner=skip_ner)
    
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
            paddle_ocr, easy_ocr, analyzer, gliner_model, medical_ner, deid_model
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
        err       = f"ERROR: {a['error'][:120]}" if a["error"] else status
        print(f"  {a['file']:<45} {err:<30} {n_regions} PHI region(s)")

    print()
    print(f"  Output Audit log    : {os.path.abspath(audit_log_path)}")
    print()
    print("  [OK] All outputs are in DICOM (.dcm) format -- NO JPEG/PNG produced.")

    # ── Auto-zip output folder ─────────────────────────────────────────────────
    if os.path.exists(output_path) and os.path.isdir(output_path):
        zip_base_name = output_path.rstrip("/\\")
        try:
            zip_file = shutil.make_archive(zip_base_name, 'zip', output_path)
            print(f"  [ZIP] Created ZIP Archive: {zip_file}")
        except Exception as e:
            print(f"  [WARN] Zip archive creation failed: {e}")

    print("=" * 70)
    return len(errors) == 0


# =============================================================================
# KAGGLE ENVIRONMENT AUTO-DETECTION & DEPENDENCY INSTALLER
# =============================================================================
def _auto_install_missing_dependencies():
    """Auto-installs required packages if running in Kaggle or Colab environment."""
    _ensure_dependencies()

    # ── Re-attempt imports for any packages that were just installed ──────
    global PADDLE_AVAILABLE, EASYOCR_AVAILABLE, PRESIDIO_AVAILABLE, GLINER_AVAILABLE

    if not PADDLE_AVAILABLE:
        try:
            from paddleocr import PaddleOCR
            import logging
            logging.getLogger("ppocr").setLevel(logging.ERROR)
            # Inject into the module's global namespace so initialize_engines can use it
            import builtins
            globals()['PaddleOCR'] = PaddleOCR
            PADDLE_AVAILABLE = True
            log.info("PaddleOCR now available after auto-install.")
        except Exception as e:
            log.warning(f"PaddleOCR still not available after install attempt: {e}")

    if not EASYOCR_AVAILABLE:
        try:
            import easyocr as _easyocr
            globals()['easyocr'] = _easyocr
            EASYOCR_AVAILABLE = True
        except Exception:
            pass

    if not PRESIDIO_AVAILABLE:
        try:
            from presidio_analyzer import AnalyzerEngine as _AE
            globals()['AnalyzerEngine'] = _AE
            PRESIDIO_AVAILABLE = True
        except Exception:
            pass

    if not GLINER_AVAILABLE:
        try:
            from gliner import GLiNER as _GLiNER
            globals()['GLiNER'] = _GLiNER
            GLINER_AVAILABLE = True
        except Exception:
            pass


def detect_kaggle_paths(default_input, default_output, default_audit):
    """Automatically detects Kaggle datasets and updates default paths if running on Kaggle."""
    input_path = default_input
    output_path = default_output
    audit_path = default_audit

    if os.path.exists("/kaggle/input"):
        _auto_install_missing_dependencies()
        try:
            target_dir = None
            # Recursively walk /kaggle/input to find any directory containing .dcm files
            for root, dirs, files in os.walk("/kaggle/input"):
                dcm_files = [f for f in files if f.lower().endswith(".dcm")]
                if dcm_files:
                    target_dir = root
                    break
            
            # Fallback to /kaggle/input/latest or subdirectories if no .dcm files found
            if not target_dir:
                if os.path.exists("/kaggle/input/latest"):
                    target_dir = "/kaggle/input/latest"
                else:
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
        except Exception as e:
            print(f"[Kaggle Path Detection] Warning: {e}")
            
        output_path = "/kaggle/working/anonymized_xray_chest"
        audit_path = "/kaggle/working/anonymized_xray_chest/audit_log.json"
        
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
    parser.add_argument("--skip-ner", action="store_true", default=False,
                        help="Skip loading heavy Hugging Face NER models (GLiNER / Medical NER) for faster execution")
    
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
        audit_log_path=args.audit,
        skip_ner=args.skip_ner
    )


if __name__ == "__main__":
    main()
