# -*- coding: utf-8 -*-
"""
=============================================================================
UNIFIED DICOM BURNED-IN TEXT ANONYMIZATION PIPELINE
=============================================================================
INPUT  : Any .dcm file (chest X-ray, 8-bit or 16-bit, CR/DX/XR modality)
OUTPUT : Anonymized .dcm file — DICOM FORMAT ONLY (no JPEG, no PNG)

Pipeline Stages:
  Stage 1 — Metadata Sanitization  (DICOM tags: PatientName, ID, DOB, etc.)
  Stage 2 — Image Enhancement       (1 Standard 8-bit variant only)
  Stage 3 — OCR Text Detection      (PaddleOCR + EasyOCR dual engine)
  Stage 4 — PHI Classification      (Allowlist + Regex + NLP + Spatial)
  Stage 5 — Pixel Redaction         (Navier-Stokes inpainting, 16-bit safe)
  Stage 6 — Verification + Retry    (Re-OCR to confirm zero residual text)
  Stage 7 — DICOM Write-Back        (correct BitsAllocated/BitsStored tags)

Usage:
  python dicom_anonymizer_pipeline.py --input chest.dcm --output anonymized/
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
import glob

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
            if _mod in _sys.modules:
                del _sys.modules[_mod]
            _to_install.append(_pip)

    if _to_install:
        print(f"[Setup] Installing/Re-installing packages: {', '.join(_to_install)}...")
        _sp.run([_sys.executable, "-m", "pip", "install", "-q", "--no-deps"] + _to_install, check=False)
        _il.invalidate_caches()

    _has_paddle = _is_installed("paddle")
    _has_paddleocr = _is_installed("paddleocr")

    if not _has_paddle or not _has_paddleocr:
        _paddle_pkg = "paddlepaddle-gpu==2.6.2" if _has_gpu else "paddlepaddle==2.6.2"
        print(f"[Setup] Installing Paddle engine ({_paddle_pkg}) & PaddleOCR...")
        for _name in ["paddle", "paddleocr"]:
            if _name in _sys.modules:
                del _sys.modules[_name]
        _paddle_cmd = [_sys.executable, "-m", "pip", "install", "-q", "--no-deps", _paddle_pkg]
        if _has_gpu:
            _paddle_cmd += ["-f", "https://www.paddlepaddle.org.cn/whl/CUDA12.0/mkl/stable.html"]
        _sp.run(_paddle_cmd, check=False)
        _sp.run([_sys.executable, "-m", "pip", "install", "-q", "--no-deps", "paddleocr==2.8.1"], check=False)
        _il.invalidate_caches()

_ensure_dependencies()

# ─── Optional engine imports ──────────────────────────────────────────────────
import sys as _sys
for _m in ["paddle", "paddleocr", "easyocr", "presidio_analyzer", "gliner", "imgaug", "astor", "decorator", "opt_einsum"]:
    if _m in _sys.modules:
        _val = _sys.modules[_m]
        if _val is None or not hasattr(_val, "__file__") or not hasattr(_val, "__path__") or (_m == "paddle" and not hasattr(_val, "tensor")):
            del _sys.modules[_m]

try:
    import paddle
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
CLINICAL_ALLOWLIST = {
    "L", "R", "LT", "RT", "LEFT", "RIGHT",
    "PA", "AP", "LAT", "LL", "RL", "LATERAL",
    "LA", "RA", "LP", "RP", "A", "P",
    "ERECT", "SUPINE", "PRONE", "DECUBITUS", "UPRIGHT",
    "PORTABLE", "MOBILE", "STAT", "ROUTINE",
    "CHEST", "ABDOMEN", "PELVIS", "SKULL", "SPINE",
    "KVP", "MAS", "MA", "SEC", "CM", "MM", "FOV",
    "CXR", "CX", "PA VIEW", "AP VIEW",
}

PII_PATTERNS = {
    "aadhaar":     r'\b\d{4}\s?\d{4}\s?\d{4}\b',
    "abha":        r'\b\d{2}-\d{4}-\d{4}-\d{4}\b',
    "phone":       r'\b(?:\+91|0)?[6-9]\d{9}\b',
    "date":        r'\b(?:\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|\d{4}[-/.]\d{1,2}[-/.]\d{1,2})\b',
    "uhid_mrn":    r'\b(?:UHID|MRN|REG|IPD|OPD|CR|PID|HID)[\s:/-]?\d+\b',
    "age_sex":     r'\b\d{1,3}\s*[/]\s*[MFO]\b',
    "name_prefix": r'\b(?:DR\.?|MR\.?|MRS\.?|MS\.?|SHRI|SMT|S/O|D/O|W/O)\b',
    "accession":   r'\b(?:ACC|ACCNO|ACCESSION)[\s:.-]?\w+\b',
}

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
    slope = getattr(ds, "RescaleSlope", 1.0)
    intercept = getattr(ds, "RescaleIntercept", 0.0)
    rescaled = pixels.astype(np.float64) * float(slope) + float(intercept)

    wc_attr = getattr(ds, "WindowCenter", None)
    ww_attr = getattr(ds, "WindowWidth", None)

    if wc_attr is not None and ww_attr is not None:
        wc = float(wc_attr[0]) if isinstance(wc_attr, (pydicom.multival.MultiValue, list)) else float(wc_attr)
        ww = float(ww_attr[0]) if isinstance(ww_attr, (pydicom.multival.MultiValue, list)) else float(ww_attr)
        min_val = wc - 0.5 - (ww - 1.0) / 2.0
        visual = np.clip((rescaled - min_val) / max(ww - 1.0, 1.0) * 255.0, 0, 255)
    else:
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
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    
    wc_attr = getattr(ds, "WindowCenter", None)
    ww_attr = getattr(ds, "WindowWidth", None)
    
    if wc_attr is not None and ww_attr is not None:
        wc = float(wc_attr[0]) if isinstance(wc_attr, (pydicom.multival.MultiValue, list)) else float(wc_attr)
        ww = float(ww_attr[0]) if isinstance(ww_attr, (pydicom.multival.MultiValue, list)) else float(ww_attr)
        min_val = wc - 0.5 - (ww - 1.0) / 2.0
        rescaled = (visual_pixels.astype(np.float64) / 255.0) * (ww - 1.0) + min_val
    else:
        orig_rescaled = original_raw_pixels.astype(np.float64) * slope + intercept
        pmin, pmax = orig_rescaled.min(), orig_rescaled.max()
        if pmax > pmin:
            rescaled = (visual_pixels.astype(np.float64) / 255.0) * (pmax - pmin) + pmin
        else:
            rescaled = np.zeros_like(visual_pixels)
            
    raw = (rescaled - intercept) / slope
    
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

    for uid_attr in ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']:
        if hasattr(ds, uid_attr):
            setattr(ds, uid_attr, generate_uid())

    ds.remove_private_tags()
    log.info("  [Stage 1] Done. All PHI metadata cleared.")
    return ds


# =============================================================================
# STAGE 2: IMAGE ENHANCEMENT (STANDARD 1-MODE ONLY)
# =============================================================================
def enhance_image(image_8bit):
    """
    Returns image variants for text detection.
    By default, it uses only standard mode for speed.
    Uncomment lines below if you need advanced variants for low-contrast text.
    """
    variants = {
        "standard": image_8bit,
        
        # --- UNCOMMENT BELOW TO ENABLE 6 ADVANCED VARIANTS ---
        # "clahe": cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image_8bit),
        # "sharpen": cv2.addWeighted(image_8bit, 2.5, cv2.GaussianBlur(image_8bit, (7, 7), 0), -1.5, 0),
        # "gamma_bright": cv2.LUT(image_8bit, np.array([np.clip(((i/255.0)**(1.0/0.5))*255, 0, 255) for i in range(256)]).astype(np.uint8)),
        # "gamma_dark": cv2.LUT(image_8bit, np.array([np.clip(((i/255.0)**(1.0/1.8))*255, 0, 255) for i in range(256)]).astype(np.uint8)),
        # "denoise": cv2.bilateralFilter(image_8bit, 9, 75, 75),
        # "inverted": 255 - image_8bit,
    }
    return variants


# =============================================================================
# STAGE 3: OCR TEXT DETECTION
# =============================================================================
def _parse_paddle_result(result):
    """
    Parses PaddleOCR result into a flat list of {text, bbox, confidence} dicts.
    """
    detections = []
    if not result or len(result) == 0:
        return detections

    for item in result:
        rec_texts  = None
        rec_polys  = None
        rec_scores = None

        if hasattr(item, 'rec_texts'):
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
            continue

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
    
    # By default, only run on "standard" variant. Uncomment other keys to enable them in OCR:
    priority_keys = [
        "standard",
        # "clahe",
        # "sharpen",
        # "gamma_bright",
        # "gamma_dark",
        # "denoise",
        # "inverted",
    ]
    target_variants = {k: v for k, v in variants.items() if k in priority_keys}
    if not target_variants:
        target_variants = variants

    for name, img in target_variants.items():
        h, w = img.shape[:2]
        max_dim = 1500
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            img_for_ocr = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            log.info(f"            Resized variant '{name}' from {w}x{h} to {new_w}x{new_h} (scale={scale:.3f}) for OCR stability.")
        else:
            scale = 1.0
            img_for_ocr = img

        if len(img_for_ocr.shape) == 2:
            rgb = cv2.cvtColor(img_for_ocr, cv2.COLOR_GRAY2RGB)
        else:
            rgb = img_for_ocr

        variant_raw = []

        if paddle_ocr:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = paddle_ocr.ocr(rgb)
                for det in _parse_paddle_result(result):
                    det["variant"] = name
                    variant_raw.append(det)
            except Exception as e:
                log.debug(f"PaddleOCR failed on variant '{name}': {e}")

        if easy_ocr:
            try:
                results = easy_ocr.readtext(rgb)
                for (bbox_pts, text, conf) in results:
                    pts = np.array(bbox_pts, dtype=np.int32)
                    xs = pts[:, 0]; ys = pts[:, 1]
                    variant_raw.append({
                        "text": str(text).strip(),
                        "bbox": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                        "confidence": float(conf),
                        "variant": name,
                        "engine": "easyocr"
                    })
            except Exception as e:
                log.debug(f"EasyOCR failed on variant '{name}': {e}")

        # Scale detections back to the original size
        if scale != 1.0:
            for det in variant_raw:
                x1, y1, x2, y2 = det["bbox"]
                det["bbox"] = [
                    int(round(x1 / scale)),
                    int(round(y1 / scale)),
                    int(round(x2 / scale)),
                    int(round(y2 / scale))
                ]

        raw.extend(variant_raw)

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
    """
    if not detections:
        return []
        
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
                
                y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                v_overlap_ratio = y_overlap / float(min_h) if min_h > 0 else 0
                gap = box2[0] - box1[2]
                
                if v_overlap_ratio > min_v_overlap and gap < max_gap_factor * min_h:
                    merged_box = [
                        min(box1[0], box2[0]),
                        min(box1[1], box2[1]),
                        max(box1[2], box2[2]),
                        max(box1[3], box2[3])
                    ]
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
    merged = merge_horizontal_lines(merged)
    return merged


def _is_clinical(text):
    val = text.strip().upper()
    if re.match(r'^(?:L|R|LT|RT|LA|RA|LP|RP|PA|AP|LL|RL|A|P)\s*\d*$', val):
        return True
    clean = re.sub(r'[^A-Z]', '', val)
    if clean in CLINICAL_ALLOWLIST:
        return True
    return False


def classify_phi(merged, image_shape, analyzer=None, gliner_model=None, medical_ner=None):
    """
    Classifies each merged detection as PHI (redact) or safe (keep).
    """
    h, w = image_shape[:2]
    phi_regions = []

    def _is_text_pure_clinical(text_str):
        val = text_str.strip().upper()
        if not val:
            return True
            
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
            "SIEMENS", "PHILIPS", "GE", "HEALTHCARE", "MEDICAL", "SYSTEMS", "MICRODICOM"
        }
        
        for token in tokens:
            if token in safe_terms:
                continue
            if re.match(r'^\d{1,4}$', token):
                continue
            if re.match(r'^(?:L|R|C|T|S)\d{0,2}$', token):
                continue
            return False
        return True

    for det in merged:
        text = det["text"]
        bbox = det["bbox"]

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

        # Phase 1: PHI/PII Identification (Must Redact FIRST)
        val_upper = clean_text.upper()
        generic_demographic_labels = [
            "NAME", "PATIENT", "DOB", "D0B", "0B:", "BIRTH", "MRN", "UHID", "PID", "AGE", "SEX", "GENDER",
            "MALE", "FEMALE", "DR.", "DOCTOR", "PHYSICIAN", "HOSPITAL", "HOSP", "CLINIC", "INSTITUT"
        ]
        has_id_label = bool(re.search(r'\bID\b', val_upper))
        has_demographic_label = any(label in val_upper for label in generic_demographic_labels) or has_id_label
        if has_demographic_label:
            is_phi = True
            reason.append("strong:demographic_label")
            classified = True

        # GLiNER Zero-Shot AI PHI Check
        if not classified and gliner_model:
            try:
                phi_ai_labels = [
                    "patient name", "person name", "doctor name",
                    "hospital name", "healthcare facility", "medical center",
                    "location", "city", "address",
                    "date of birth", "patient id", "medical record number", "id number"
                ]
                phi_entities = gliner_model.predict_entities(clean_text, phi_ai_labels, threshold=0.30)
                if phi_entities:
                    top_entity = phi_entities[0]
                    is_phi = True
                    reason.append(f"gliner_ai_phi:{top_entity['label']}")
                    classified = True
            except Exception as e:
                log.debug(f"GLiNER AI PHI detection error: {e}")

        # Microsoft Presidio NLP PII Check
        if not classified and analyzer:
            try:
                results = analyzer.analyze(text=clean_text, language="en")
                for r in results:
                    if r.entity_type in {"PERSON", "DATE_TIME", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"} and r.score > 0.35:
                        is_phi = True
                        reason.append(f"presidio:{r.entity_type}")
                        classified = True
                        break
            except Exception as e:
                log.debug(f"Presidio PII analysis failed: {e}")

        # Regex PII Pattern Check
        if not classified:
            for pat_name, pat_regex in PII_PATTERNS.items():
                if re.search(pat_regex, clean_text, re.IGNORECASE):
                    is_phi = True
                    reason.append(f"strong:pii_pattern_{pat_name}")
                    classified = True
                    break

        # Phase 2: Weak / Zero-shot PII Identification
        if not classified and analyzer:
            try:
                results = analyzer.analyze(text=clean_text, language="en")
                for r in results:
                    if r.entity_type in {"PERSON", "DATE_TIME", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"} and r.score > 0.45:
                        is_phi = True
                        reason.append(f"presidio:{r.entity_type}")
                        classified = True
                        break
            except Exception as e:
                log.debug(f"Presidio analyze failed: {e}")

        if not classified and gliner_model:
            try:
                labels = ["patient name", "doctor name", "hospital name", "location", "date of birth", "id number"]
                entities = gliner_model.predict_entities(clean_text, labels, threshold=0.40)
                for ent in entities:
                    ent_label = ent["label"]
                    if ent_label in labels:
                        is_phi = True
                        reason.append(f"gliner_pii:{ent_label}")
                        classified = True
                        break
            except Exception as e:
                log.debug(f"GLiNER PII check failed: {e}")

        # Phase 3: Clinical/Medical Entity Preservation (Must Keep)
        if not classified and medical_ner:
            try:
                entities = medical_ner(clean_text)
                for ent in entities:
                    ent_group = ent.get("entity_group", "")
                    if ent_group in {
                        "Anatomical_structure", "Diagnostic_procedure", "Disease_disorder",
                        "Sign_symptom", "Lab_value", "Medication", "Detailed_description",
                        "Biological_structure", "Therapeutic_procedure"
                    }:
                        if not re.search(r'(?:PID|MRN|UHID|DOB|ID|\d{4}-\w+-\d+)', clean_text, re.IGNORECASE):
                            is_phi = False
                            classified = True
                            reason.append(f"medical_ner:{ent_group}")
                            break
            except Exception as e:
                log.debug(f"Medical NER prediction failed: {e}")

        if not classified and gliner_model:
            try:
                labels = [
                    "anatomy", "body part", "organ",
                    "disease", "medical condition", "diagnosis", "clinical finding",
                    "medical procedure", "diagnostic test",
                    "medication", "drug name",
                    "positioning term", "scan parameter", "imaging technique",
                    "medical device", "medical equipment",
                    "vital sign", "lab value", "measurement"
                ]
                entities = gliner_model.predict_entities(clean_text, labels, threshold=0.40)
                for ent in entities:
                    ent_label = ent["label"]
                    if ent_label in labels:
                        is_phi = False
                        classified = True
                        reason.append(f"gliner_clinical:{ent_label}")
                        break
            except Exception as e:
                log.debug(f"GLiNER clinical check failed: {e}")

        if not classified:
            if re.match(r'^\d{1,4}(?:\.\d+)?(?:\s*-\s*\d{1,4}(?:\.\d+)?)?$', clean_text):
                is_phi = False
                classified = True
                reason.append("clinical:exposure_number")

        if not classified:
            if _is_text_pure_clinical(clean_text):
                is_phi = False
                classified = True
                reason.append("clinical:allowlist_fallback")

        # Phase 4: Spatial / Default Fallback
        if not classified:
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
# STAGE 5: PIXEL REDACTION (Navier-Stokes inpainting, 16-bit safe)
# =============================================================================
def _get_character_mask(roi_8bit, bbox_local=None, dilation_px=3):
    """
    Robust Tri-Signal Dual-Polarity Character Stroke Segmentation.
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

    try:
        denoised = cv2.bilateralFilter(crop, d=5, sigmaColor=15.0, sigmaSpace=3.0)
    except Exception:
        denoised = crop.copy()

    k_size = max(5, min(15, (crop_h // 2) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
    _, mask_bright = cv2.threshold(tophat, 8, 255, cv2.THRESH_BINARY)

    bg_med = float(np.median(crop))
    crop_max = float(np.max(crop))
    if crop_max > bg_med + 10.0:
        t_contrast = bg_med + 0.20 * (crop_max - bg_med)
        _, mask_contrast = cv2.threshold(crop, min(245.0, t_contrast), 255, cv2.THRESH_BINARY)
    else:
        mask_contrast = np.zeros_like(crop)

    stroke_union = cv2.bitwise_or(mask_bright, mask_contrast)

    k_dil_size = max(5, min(9, dilation_px * 2 + 1))
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil_size, k_dil_size))
    dilated = cv2.dilate(stroke_union, k_dil, iterations=1)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[by1:by2, bx1:bx2] = dilated
    return full_mask


def _inpaint_native(image, mask_8, radius=3, method=cv2.INPAINT_NS):
    """
    Inpaints at native bit depth using float64 normalized intermediate.
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

    norm_f64 = (image.astype(np.float64) - raw_min) / (raw_max - raw_min)
    t8 = np.clip(norm_f64 * 255.0, 0, 255).astype(np.uint8)

    inp8 = cv2.inpaint(t8, mask_8, radius, method)

    inp_f64 = inp8.astype(np.float64) / 255.0
    inp_raw = inp_f64 * (raw_max - raw_min) + raw_min

    mask_bool = mask_8 > 0
    result[mask_bool] = inp_raw[mask_bool].astype(image.dtype)
    return result


def _redact_unified(cleaned, x1, y1, x2, y2, ds=None, modality="CT"):
    """
    Hybrid Character-Level Redaction: Two-Pass Iterative Neighbor Inpainting.
    """
    h, w = cleaned.shape[:2]
    box_h, box_w = y2 - y1, x2 - x1
    if box_h <= 0 or box_w <= 0:
        return cleaned

    ctx_pad = max(15, max(box_h, box_w) // 3)
    cy1 = max(0, y1 - ctx_pad);  cy2 = min(h, y2 + ctx_pad)
    cx1 = max(0, x1 - ctx_pad);  cx2 = min(w, x2 + ctx_pad)

    ctx_region = cleaned[cy1:cy2, cx1:cx2].copy()
    ctx_h, ctx_w = ctx_region.shape[:2]

    def _to_8bit(region):
        if ds is not None:
            return _apply_voi_lut(region, ds)
        rmin, rmax = float(region.min()), float(region.max())
        if rmax > rmin:
            return np.clip(((region.astype(np.float64) - rmin) /
                           (rmax - rmin) * 255.0), 0, 255).astype(np.uint8)
        return region.astype(np.uint8)

    roi_8 = _to_8bit(ctx_region)
    bbox_local = (
        max(0, x1 - cx1 - 2), max(0, y1 - cy1 - 2),
        min(ctx_w, x2 - cx1 + 2), min(ctx_h, y2 - cy1 + 2)
    )
    char_mask = _get_character_mask(roi_8, bbox_local=bbox_local, dilation_px=3)

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

    mask_8 = char_mask.astype(np.uint8)
    if not np.any(mask_8 > 0):
        return cleaned

    inpainted = _inpaint_native(ctx_region, mask_8, radius=4, method=cv2.INPAINT_NS)
    mask_bool = mask_8 > 0
    ctx_region[mask_bool] = inpainted[mask_bool]

    bx1, by1, bx2, by2 = bbox_local
    roi_8_pass2 = _to_8bit(ctx_region)
    crop_p2 = roi_8_pass2[by1:by2, bx1:bx2]
    if crop_p2.size > 0:
        ring_pixels = roi_8_pass2[char_mask == 0]
        if ring_pixels.size > 0:
            bg_med = float(np.median(ring_pixels))
            bg_std = float(np.std(ring_pixels))
        else:
            bg_med = float(np.median(crop_p2))
            bg_std = 10.0

        bright_thresh = min(250.0, bg_med + max(15.0, 2.0 * bg_std))
        _, residual_mask_crop = cv2.threshold(crop_p2, bright_thresh, 255, cv2.THRESH_BINARY)
        residual_count = int(np.sum(residual_mask_crop > 0))

        if residual_count > 5:
            residual_dilated = cv2.dilate(residual_mask_crop,
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            pass2_mask = np.zeros((ctx_h, ctx_w), dtype=np.uint8)
            pass2_mask[by1:by2, bx1:bx2] = residual_dilated

            inpainted_p2 = _inpaint_native(ctx_region, pass2_mask, radius=4,
                                           method=cv2.INPAINT_NS)
            p2_bool = pass2_mask > 0
            ctx_region[p2_bool] = inpainted_p2[p2_bool]

    cleaned[cy1:cy2, cx1:cx2] = ctx_region
    return cleaned


# =============================================================================
# STAGE 6: VERIFICATION + RETRY
# =============================================================================
def verify_redaction(
    pixel_array_16bit,
    redacted_regions,
    ds,
    paddle_ocr=None,
    easy_ocr=None,
    confidence_threshold=0.3,
    max_iterations=3,
):
    """
    Verification loop: re-run OCR on redacted regions to ensure no text remains.
    If text is found, apply aggressive uniform fill on that box and retry.
    """
    current_array = pixel_array_16bit.copy()
    messages = []
    h, w = current_array.shape[:2]

    for iteration in range(max_iterations):
        img_8bit = _apply_voi_lut(current_array, ds)
        remaining_text = []

        for region in redacted_regions:
            box = region["bbox"]
            x1, y1, x2, y2 = box
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            crop = img_8bit[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
                continue

            variants = {"standard": crop}
            dets = detect_text(variants, paddle_ocr=paddle_ocr, easy_ocr=easy_ocr)

            for det in dets:
                if det["confidence"] >= confidence_threshold and len(det["text"].strip()) > 0:
                    local_bbox = det["bbox"]
                    global_bbox = [
                        x1 + local_bbox[0],
                        y1 + local_bbox[1],
                        x1 + local_bbox[2],
                        y1 + local_bbox[3]
                    ]
                    remaining_text.append({
                        "text": det["text"],
                        "bbox": global_bbox,
                        "confidence": det["confidence"]
                    })

        if not remaining_text:
            messages.append(
                f"Verification PASSED on iteration {iteration+1}: "
                "No residual text detected in redacted regions."
            )
            return current_array, True, messages

        messages.append(
            f"Iteration {iteration+1}: Found {len(remaining_text)} residual text region(s). "
            "Re-redacting with aggressive uniform fill..."
        )

        for rt in remaining_text:
            bx1, by1, bx2, by2 = rt["bbox"]
            bx1, by1 = max(0, bx1), max(0, by1)
            bx2, by2 = min(w, bx2), min(h, by2)
            if bx2 > bx1 and by2 > by1:
                roi = current_array[by1:by2, bx1:bx2]
                bg_val = np.median(roi)
                current_array[by1:by2, bx1:bx2] = bg_val

    messages.append(
        f"Max verification iterations ({max_iterations}) reached. "
        "Manual review recommended."
    )
    return current_array, False, messages


# =============================================================================
# STAGE 7: DICOM WRITE-BACK
# =============================================================================
def save_dicom(
    pixel_array_16bit,
    original_ds,
    output_path,
    add_anonymization_tags=True,
    redaction_summary=None,
):
    """
    Write the anonymized pixel data back to a DICOM file.
    Corrects BitsAllocated/BitsStored tags.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    if not output_path.lower().endswith(".dcm"):
        output_path += ".dcm"

    ds = original_ds.copy()
    ds.PixelData = pixel_array_16bit.tobytes()
    ds.Rows = pixel_array_16bit.shape[0]
    ds.Columns = pixel_array_16bit.shape[1]

    if pixel_array_16bit.dtype == np.uint8:
        ds.BitsAllocated = 8
        ds.BitsStored = 8
        ds.HighBit = 7
    else:
        ds.BitsAllocated = 16
        orig_bits_stored = getattr(ds, "BitsStored", 16)
        if orig_bits_stored not in [12, 14, 16]:
            ds.BitsStored = 16
        else:
            ds.BitsStored = orig_bits_stored
        ds.HighBit = ds.BitsStored - 1

    if add_anonymization_tags:
        ds.PatientIdentityRemoved = "YES"
        ds.DeidentificationMethod = (
            "Burned-in Text Pixel Anonymization (OCR + AI + Adaptive Redaction)"
        )
        ds.BurnedInAnnotation = "NO"

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        comment = f"Pixel anonymization performed on {timestamp}"
        if redaction_summary:
            comment += (
                f" | Regions detected: {redaction_summary.get('total_detected', 0)}"
                f" | Redacted: {redaction_summary.get('total_redacted', 0)}"
                f" | Kept: {redaction_summary.get('total_kept', 0)}"
                f" | Verification: {redaction_summary.get('verification_status', 'N/A')}"
            )
        existing = str(getattr(ds, "ImageComments", ""))
        ds.ImageComments = f"{existing} | {comment}" if existing else comment

        ds.SOPInstanceUID = generate_uid()
        now = datetime.datetime.now()
        ds.InstanceCreationDate = now.strftime("%Y%m%d")
        ds.InstanceCreationTime = now.strftime("%H%M%S.%f")

    ds.save_as(output_path, write_like_original=True)
    return output_path


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================
def run_anonymizer_pipeline(
    input_path,
    output_path,
    paddle_ocr=None,
    easy_ocr=None,
    analyzer=None,
    gliner_model=None,
    medical_ner=None,
    enable_verification=True,
    max_verification_loops=3,
):
    """Runs all stages of the unified DICOM pixel anonymization pipeline."""
    log.info(f"Processing DICOM file: {input_path}")
    
    ds = pydicom.dcmread(input_path)
    if not hasattr(ds, "pixel_array"):
        raise ValueError(f"DICOM file has no pixel data: {input_path}")
        
    pixel_array = ds.pixel_array.copy()
    
    ds_sanitized = sanitize_metadata(ds)
    img_8bit = _apply_voi_lut(pixel_array, ds_sanitized)
    variants = enhance_image(img_8bit)
    
    log.info("  [Stage 3] Detecting text using OCR dual engines...")
    raw_detections = detect_text(variants, paddle_ocr=paddle_ocr, easy_ocr=easy_ocr, modality=getattr(ds_sanitized, "Modality", "CT"))
    log.info(f"  [Stage 3] Raw OCR Detections found: {len(raw_detections)}")
    
    log.info("  [Stage 4] Merging bounding boxes and classifying PHI...")
    merged_detections = merge_detections(raw_detections)
    log.info(f"  [Stage 4] Merged text blocks: {len(merged_detections)}")
    
    phi_regions = classify_phi(
        merged_detections, 
        pixel_array.shape, 
        analyzer=analyzer, 
        gliner_model=gliner_model, 
        medical_ner=medical_ner
    )
    log.info(f"  [Stage 4] Classified as PHI (to redact): {len(phi_regions)}")
    
    log.info("  [Stage 5] Redacting pixels at native bit depth...")
    cleaned_pixel_array = pixel_array.copy()
    for region in phi_regions:
        box = region["bbox"]
        bx1, by1, bx2, by2 = box
        cleaned_pixel_array = _redact_unified(
            cleaned_pixel_array, 
            bx1, by1, bx2, by2, 
            ds=ds_sanitized
        )
        
    verification_passed = True
    verification_msgs = []
    if enable_verification and len(phi_regions) > 0:
        log.info("  [Stage 6] Running Verification and Retry loop...")
        cleaned_pixel_array, verification_passed, verification_msgs = verify_redaction(
            cleaned_pixel_array,
            phi_regions,
            ds_sanitized,
            paddle_ocr=paddle_ocr,
            easy_ocr=easy_ocr,
            confidence_threshold=0.3,
            max_iterations=max_verification_loops
        )
        for msg in verification_msgs:
            log.info(f"    {msg}")
            
    log.info(f"  [Stage 7] Writing anonymized DICOM file...")
    redaction_summary = {
        "total_detected": len(merged_detections),
        "total_redacted": len(phi_regions),
        "total_kept": len(merged_detections) - len(phi_regions),
        "verification_status": "PASSED" if verification_passed else "NEEDS_REVIEW"
    }
    
    saved_path = save_dicom(
        cleaned_pixel_array,
        ds_sanitized,
        output_path,
        add_anonymization_tags=True,
        redaction_summary=redaction_summary
    )
    log.info(f"Anonymization complete. Output saved to: {saved_path}")
    return saved_path


# =============================================================================
# MAIN CLI ENTRY POINT
# =============================================================================
def main(args_list=None):
    parser = argparse.ArgumentParser(
        description="Unified Standalone DICOM Burned-In Text Anonymizer Pipeline (Single 8-Bit Variant mode)"
    )
    parser.add_argument("--input", "-i", required=True, help="Input DICOM file or folder of .dcm files")
    parser.add_argument("--output", "-o", required=True, help="Output directory for anonymized files")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR engines")
    parser.add_argument("--no-verify", action="store_true", help="Disable re-OCR verification loop")
    parser.add_argument("--max-loops", type=int, default=3, help="Max verification loop iterations")
    parser.add_argument("--fallback", choices=["REDACT", "KEEP"], default="REDACT", help="Action for unclassified text")
    
    args = parser.parse_args(args_list)
    
    log.info("Loading OCR engines & NLP models...")
    paddle_ocr = None
    easy_ocr = None
    analyzer = None
    gliner_model = None
    
    if PADDLE_AVAILABLE:
        try:
            paddle_ocr = PaddleOCR(use_angle_cls=True, lang="en", use_gpu=args.gpu)
            log.info("  PaddleOCR engine loaded.")
        except Exception as e:
            log.warning(f"  Failed to load PaddleOCR: {e}")
            
    easy_ocr = None
    # Disabled EasyOCR to run in PaddleOCR-only mode (saves RAM & execution time)
    # if EASYOCR_AVAILABLE:
    #     try:
    #         easy_ocr = easyocr.Reader(['en'], gpu=args.gpu)
    #         log.info("  EasyOCR engine loaded.")
    #     except Exception as e:
    #         log.warning(f"  Failed to load EasyOCR: {e}")

            
    if PRESIDIO_AVAILABLE:
        try:
            analyzer = AnalyzerEngine()
            log.info("  Presidio Analyzer loaded.")
        except Exception as e:
            log.warning(f"  Failed to load Presidio: {e}")
            
    if GLINER_AVAILABLE:
        try:
            gliner_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
            log.info("  GLiNER Zero-Shot NLP model loaded.")
        except Exception as e:
            log.warning(f"  Failed to load GLiNER: {e}")

    input_files = []
    if os.path.isfile(args.input):
        input_files.append(args.input)
    elif os.path.isdir(args.input):
        input_files = glob.glob(os.path.join(args.input, "*.dcm")) + glob.glob(os.path.join(args.input, "*.DCM"))
    else:
        log.error(f"Input path not found: {args.input}")
        sys.exit(1)
        
    if not input_files:
        log.error(f"No .dcm files found at: {args.input}")
        sys.exit(1)
        
    os.makedirs(args.output, exist_ok=True)
    log.info(f"Processing {len(input_files)} file(s)...")
    
    for i, file_path in enumerate(input_files):
        filename = os.path.basename(file_path)
        out_path = os.path.join(args.output, f"anonymized_{filename}")
        try:
            run_anonymizer_pipeline(
                file_path,
                out_path,
                paddle_ocr=paddle_ocr,
                easy_ocr=easy_ocr,
                analyzer=analyzer,
                gliner_model=gliner_model,
                enable_verification=not args.no_verify,
                max_verification_loops=args.max_loops
            )
        except Exception as e:
            log.error(f"Error processing {filename}: {e}", exc_info=True)


if __name__ == "__main__":
    is_notebook = False
    try:
        from IPython import get_ipython
        shell = get_ipython().__class__.__name__
        if 'InteractiveShell' in shell:
            is_notebook = True
    except Exception:
        pass
        
    if not is_notebook and not any('ipykernel' in arg for arg in sys.argv):
        main()
    else:
        log.info("[Notebook Mode] Pipeline defined successfully. You can run the pipeline by calling:")
        log.info("  run_anonymizer_pipeline(input_path, output_path, ...)")
        log.info("or by calling main() with arguments, e.g.:")
        log.info("  main(['--input', 'path/to/dicom', '--output', 'path/to/output'])")
