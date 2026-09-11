"""
ocr_detect.py — Stage 3: OCR text detection, powered exclusively by PaddleOCR.
"""

import numpy as np
import cv2

from config import log

_MIN_CROP_HEIGHT = 48
_MAX_UPSCALE = 4.0
_PAD_FRAC = 0.6


def _parse_paddle_output(res):
    """
    Normalizes outputs from native PaddleOCR and RapidOCR into standard (pts, text, conf) tuples.
    """
    parsed = []
    if not res:
        return parsed

    # Check if native PaddleOCR format: [[ [box_pts, (text, conf)], ... ]]
    if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
        # Native Paddle returns list of lines per image
        lines = res[0] if (len(res) == 1 and isinstance(res[0], list) and len(res[0]) > 0 and isinstance(res[0][0], list)) else res
        for line in lines:
            if isinstance(line, (list, tuple)) and len(line) >= 2:
                box_pts = line[0]
                text_info = line[1]
                if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                    text, conf = str(text_info[0]), float(text_info[1])
                elif isinstance(text_info, str):
                    text = text_info
                    conf = float(line[2]) if len(line) > 2 else 0.9
                else:
                    continue
                parsed.append((box_pts, text, conf))
    elif isinstance(res, list):
        # RapidOCR format: [ [box_pts, text, str_conf], ... ]
        for line in res:
            if isinstance(line, (list, tuple)) and len(line) >= 3:
                box_pts = line[0]
                text = str(line[1])
                try:
                    conf = float(line[2])
                except Exception:
                    conf = 0.9
                parsed.append((box_pts, text, conf))

    return parsed


def detect_text_in_regions(image_8bit, bboxes, paddle_ocr=None, padding=12, run_on_regions=False):
    """
    Detects text from the image using the primary PaddleOCR engine.
    If run_on_regions is False, runs PaddleOCR on the full image directly.
    If run_on_regions is True, runs PaddleOCR strictly on the individual cropped candidate regions.
    """
    raw = []

    if len(image_8bit.shape) == 2:
        rgb_full = cv2.cvtColor(image_8bit, cv2.COLOR_GRAY2RGB)
    else:
        rgb_full = image_8bit

    # ── 1. Primary: PaddleOCR (Full Image) ──────────────────────────────────
    if paddle_ocr and not run_on_regions:
        try:
            img_h, img_w = image_8bit.shape[:2]
            if hasattr(paddle_ocr, 'ocr'):
                # Native PaddleOCR
                res = paddle_ocr.ocr(rgb_full, cls=False)
            elif callable(paddle_ocr):
                # RapidOCR / callable
                res, _ = paddle_ocr(rgb_full)
            else:
                res = None

            parsed = _parse_paddle_output(res)
            _BBOX_PAD = 5  # px padding to prevent half-cut text
            for (box_pts, text, conf) in parsed:
                pts = np.array(box_pts, dtype=np.int32)
                xs = pts[:, 0]
                ys = pts[:, 1]
                bbox = [
                    max(0, int(xs.min()) - _BBOX_PAD),
                    max(0, int(ys.min()) - _BBOX_PAD),
                    min(img_w, int(xs.max()) + _BBOX_PAD),
                    min(img_h, int(ys.max()) + _BBOX_PAD),
                ]
                raw.append({
                    "text": str(text).strip(),
                    "bbox": bbox,
                    "confidence": float(conf),
                    "variant": "full",
                    "engine": "paddleocr"
                })
            return raw
        except Exception as e:
            log.error(f"PaddleOCR full-image mode failed: {e}")

    # ── 2. Targeted: PaddleOCR (Region-based / Crops) ───────────────────────
    if paddle_ocr and run_on_regions and bboxes:
        h, w = image_8bit.shape[:2]
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            pad = padding
            x1 = max(0, x1 - pad)
            y1 = max(0, y1 - pad)
            x2 = min(w, x2 + pad)
            y2 = min(h, y2 + pad)
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

            try:
                if hasattr(paddle_ocr, 'ocr'):
                    res = paddle_ocr.ocr(crop, cls=False)
                elif callable(paddle_ocr):
                    res, _ = paddle_ocr(crop)
                else:
                    res = None

                parsed = _parse_paddle_output(res)
                for (box_pts, text, conf) in parsed:
                    pts = np.array(box_pts, dtype=np.int32)
                    xs = pts[:, 0]
                    ys = pts[:, 1]
                    local_bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
                    raw.append({
                        "text": str(text).strip(),
                        "bbox": [int(v) for v in _to_full(local_bbox)],
                        "confidence": float(conf),
                        "variant": "region",
                        "engine": "paddleocr_region"
                    })
            except Exception as e:
                log.debug(f"PaddleOCR region-based failed on {bbox}: {e}")

    return raw
