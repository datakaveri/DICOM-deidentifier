"""
ocr_detect.py — Stage 3: OCR text detection, restricted to the candidate
regions found by Stage 2 (text_region_detect.py), across available engines
(PaddleOCR / EasyOCR).
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


_MIN_CROP_HEIGHT = 48
_MAX_UPSCALE = 4.0
_PAD_FRAC = 0.6


def detect_text_in_regions(image_8bit, bboxes, paddle_ocr=None, easy_ocr=None, padding=12):
    """
    Runs all available OCR engines only inside each candidate bbox from
    Stage 2 (text_region_detect.detect_text_regions), instead of across the
    whole image. Each bbox is padded (generously, scaled to the box's own
    height, since Stage 2 boxes are tight single text lines and OCR engines
    need surrounding context to recognize characters reliably) and, if still
    small, upscaled so the recognizer has enough resolution to work with.
    Results are translated back into full-image coordinates.
    """
    raw = []
    h, w = image_8bit.shape[:2]

    if len(image_8bit.shape) == 2:
        rgb_full = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    else:
        rgb_full = image_8bit

    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        pad = max(padding, int((y2 - y1) * _PAD_FRAC))
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        if x2 <= x1 or y2 <= y1:
            continue

        crop = rgb_full[y1:y2, x1:x2]

        scale = 1.0
        crop_h = y2 - y1
        if crop_h < _MIN_CROP_HEIGHT:
            scale = min(_MAX_UPSCALE, _MIN_CROP_HEIGHT / crop_h)
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        def _to_full(local_bbox):
            return [
                x1 + local_bbox[0] / scale, y1 + local_bbox[1] / scale,
                x1 + local_bbox[2] / scale, y1 + local_bbox[3] / scale,
            ]

        # PaddleOCR — use new predict() API (ocr() is deprecated in PaddleX)
        if paddle_ocr:
            try:
                # Try new predict() first (PaddleX >= 3.x)
                if hasattr(paddle_ocr, 'predict'):
                    result = paddle_ocr.predict(crop)
                else:
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", DeprecationWarning)
                        result = paddle_ocr.ocr(crop, cls=False)
                for det in _parse_paddle_result(result):
                    det["bbox"] = [int(v) for v in _to_full(det["bbox"])]
                    det["variant"] = "region"
                    raw.append(det)
            except Exception as e:
                log.debug(f"PaddleOCR failed on region {bbox}: {e}")

        # EasyOCR
        if easy_ocr:
            try:
                results = easy_ocr.readtext(crop)
                for (bbox_pts, text, conf) in results:
                    pts = np.array(bbox_pts, dtype=np.int32)
                    xs = pts[:, 0]; ys = pts[:, 1]
                    local_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    raw.append({
                        "text": str(text).strip(),
                        "bbox": [int(v) for v in _to_full(local_bbox)],
                        "confidence": float(conf),
                        "variant": "region",
                        "engine": "easyocr"
                    })
            except Exception as e:
                log.debug(f"EasyOCR failed on region {bbox}: {e}")

    return raw
