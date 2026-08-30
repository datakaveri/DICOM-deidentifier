"""
text_region_detect.py — Stage 2: shape-based burned-in text region detection.

Detects candidate text regions by connected-component/shape analysis and
horizontal phrase clustering — no heavy OCR is needed here. Stage 3 (ocr_detect.py)
then runs PaddleOCR strictly inside these compact, localized bounding boxes.
"""

import numpy as np
import cv2

_SCALE = 2

_MIN_CHAR_H, _MAX_CHAR_H = 4, 60
_MIN_CHAR_W, _MAX_CHAR_W = 2, 120
_MIN_CHAR_AREA = 8

# Maximum horizontal gap between adjacent characters/words to be grouped in the same box
_MAX_WORD_GAP_PX = 45
_LINE_Y_TOLERANCE = 10
_MIN_BLOBS_PER_REGION = 2

_PAD_X, _PAD_Y = 10, 6
_MIN_PEAK = 70


def _to_grayscale(image):
    if image.ndim == 2:
        return image

    c = image.shape[2]
    if c == 1:
        return image[:, :, 0]
    if c in (3, 4):
        code = cv2.COLOR_BGR2GRAY if c == 3 else cv2.COLOR_BGRA2GRAY
        return cv2.cvtColor(image, code)
    return image.mean(axis=2).astype(np.uint8)


def _binarize(image):
    """
    Produces dual foreground masks (Otsu global + Adaptive local contrast)
    to capture both high-contrast margin text and text overlaid on anatomy.
    """
    gray = _to_grayscale(image)
    gray2x = cv2.resize(gray, None, fx=_SCALE, fy=_SCALE, interpolation=cv2.INTER_CUBIC)

    _, otsu = cv2.threshold(gray2x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(otsu) < 127:
        otsu = cv2.bitwise_not(otsu)
    fg_otsu = cv2.bitwise_not(otsu)

    fg_adaptive = cv2.adaptiveThreshold(
        gray2x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        31, -15
    )

    return fg_otsu, fg_adaptive


def detect_text_regions(image_8bit):
    """
    Detects candidate burned-in text regions by shape and horizontal phrase clustering.
    Produces tight, localized bounding boxes for each text line/phrase without merging
    across the entire width of the image.

    Returns a list of [x1, y1, x2, y2] bounding boxes in image pixel coordinates.
    """
    img_h, img_w = image_8bit.shape[:2]

    gray = _to_grayscale(image_8bit)
    if int(gray.max()) < _MIN_PEAK:
        return []

    fg_otsu, fg_adaptive = _binarize(image_8bit)

    char_boxes = []
    seen = set()

    for fg in (fg_otsu, fg_adaptive):
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

        for i in range(1, n):
            x = stats[i, cv2.CC_STAT_LEFT] // _SCALE
            y = stats[i, cv2.CC_STAT_TOP] // _SCALE
            bw = stats[i, cv2.CC_STAT_WIDTH] // _SCALE
            bh = stats[i, cv2.CC_STAT_HEIGHT] // _SCALE
            ba = stats[i, cv2.CC_STAT_AREA] // (_SCALE * _SCALE)

            if (_MIN_CHAR_H <= bh <= _MAX_CHAR_H and
                    _MIN_CHAR_W <= bw <= _MAX_CHAR_W and
                    ba >= _MIN_CHAR_AREA):
                box = (int(x), int(y), int(x + bw), int(y + bh))
                if box not in seen:
                    seen.add(box)
                    char_boxes.append(box)

    if not char_boxes:
        return []

    # Sort characters top-to-bottom, left-to-right
    char_boxes.sort(key=lambda b: (b[1], b[0]))

    # ── Horizontal Phrase Clustering ──────────────────────────────────────────
    # Group characters into distinct phrases based on Y-alignment AND horizontal proximity.
    clusters = []
    for box in char_boxes:
        bx1, by1, bx2, by2 = box
        matched = False
        for cluster in clusters:
            cy_box = (by1 + by2) / 2
            cy_cluster = sum((b[1] + b[3]) / 2 for b in cluster) / len(cluster)

            # Check if vertically on same line
            if abs(cy_box - cy_cluster) <= _LINE_Y_TOLERANCE:
                # Check horizontal distance to nearest character in cluster
                min_dx = min(
                    abs(bx1 - b[2]) if bx1 >= b[2] else (abs(b[0] - bx2) if b[0] >= bx2 else 0)
                    for b in cluster
                )
                if min_dx <= _MAX_WORD_GAP_PX:
                    cluster.append(box)
                    matched = True
                    break

        if not matched:
            clusters.append([box])

    # Convert clusters into tight bounding boxes
    results = []
    for cluster in clusters:
        if len(cluster) < _MIN_BLOBS_PER_REGION:
            continue

        cx1 = max(0, min(b[0] for b in cluster) - _PAD_X)
        cy1 = max(0, min(b[1] for b in cluster) - _PAD_Y)
        cx2 = min(img_w, max(b[2] for b in cluster) + _PAD_X)
        cy2 = min(img_h, max(b[3] for b in cluster) + _PAD_Y)

        box_w = cx2 - cx1
        box_h = cy2 - cy1

        # Reject enormous whole-image noise boxes
        if box_w > img_w * 0.95 and box_h > img_h * 0.5:
            continue

        results.append([int(cx1), int(cy1), int(cx2), int(cy2)])

    # Merge overlapping/adjacent candidate boxes (iterative until stable)
    def _merge_pass(boxes):
        merged = []
        for box in sorted(boxes, key=lambda b: (b[1], b[0])):
            matched = False
            bx1, by1, bx2, by2 = box
            for m in merged:
                mx1, my1, mx2, my2 = m
                # If boxes overlap or are within 15px of each other
                if not (bx2 < mx1 - 15 or bx1 > mx2 + 15 or by2 < my1 - 8 or by1 > my2 + 8):
                    m[0] = min(mx1, bx1)
                    m[1] = min(my1, by1)
                    m[2] = max(mx2, bx2)
                    m[3] = max(my2, by2)
                    matched = True
                    break
            if not matched:
                merged.append(box)
        return merged

    merged_results = results
    for _ in range(5):  # max 5 iterations to convergence
        new_merged = _merge_pass(merged_results)
        if len(new_merged) == len(merged_results):
            break
        merged_results = new_merged

    return merged_results
