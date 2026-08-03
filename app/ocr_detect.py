"""
ocr_detect.py — Stage 3: OCR text detection across all image variants and
available engines (PaddleOCR / EasyOCR).
"""

import numpy as np
import cv2

from config import log


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
                    variant_raw.append(det)
            except Exception as e:
                log.debug(f"PaddleOCR failed on variant '{name}': {e}")

        # EasyOCR
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

