"""
text_region_detect.py — Stage 2: shape-based burned-in text region detection.

Detects candidate text regions by connected-component/shape analysis only —
no OCR, no text is read here. Stage 3 (ocr_detect.py) then runs OCR strictly
inside the bounding boxes this stage finds, instead of across the whole
image. Ported from bbox/detect_text_regions_standalone.py's TextRegionDetector.
"""

import numpy as np
import cv2

_SCALE = 2

_MIN_CHAR_H, _MAX_CHAR_H = 4, 60
_MIN_CHAR_W, _MAX_CHAR_W = 2, 120
_MIN_CHAR_AREA = 8

_LINE_Y_TOLERANCE = 8
_MIN_BLOBS_PER_LINE = 2

_PAD_X, _PAD_Y = 8, 3

_MIN_PEAK = 120
_MIN_LINE_PEAK = 120
_MAX_BG_MEAN = 80
_BRIGHT_THRESHOLD = 130


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
    gray = _to_grayscale(image)
    gray2x = cv2.resize(gray, None, fx=_SCALE, fy=_SCALE, interpolation=cv2.INTER_CUBIC)

    _, binary = cv2.threshold(gray2x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    return binary


def detect_text_regions(image_8bit):
    """
    Detects candidate burned-in text regions by shape (blob/line geometry),
    not OCR. `image_8bit` is an 8-bit grayscale (or BGR/BGRA) image.

    Returns a list of [x1, y1, x2, y2] bounding boxes, in the same pixel
    coordinates as `image_8bit`.
    """
    crop_h, crop_w = image_8bit.shape[:2]

    gray_check = _to_grayscale(image_8bit)
    if int(gray_check.max()) < _MIN_PEAK:
        return []

    binary = _binarize(image_8bit)
    fg = cv2.bitwise_not(binary)

    n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

    char_boxes = []

    for i in range(1, n):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]
        bh = stats[i, cv2.CC_STAT_HEIGHT]
        ba = stats[i, cv2.CC_STAT_AREA]

        x1_c = x // _SCALE
        y1_c = y // _SCALE
        x2_c = (x + bw) // _SCALE
        y2_c = (y + bh) // _SCALE
        h_c = y2_c - y1_c
        w_c = x2_c - x1_c
        area_c = ba // (_SCALE * _SCALE)

        if (_MIN_CHAR_H <= h_c <= _MAX_CHAR_H and
                _MIN_CHAR_W <= w_c <= _MAX_CHAR_W and
                area_c >= _MIN_CHAR_AREA):
            char_boxes.append((x1_c, y1_c, x2_c, y2_c))

    if not char_boxes:
        return []

    char_boxes.sort(key=lambda b: ((b[1] + b[3]) / 2, b[0]))

    lines = []
    current = [char_boxes[0]]

    for box in char_boxes[1:]:
        prev_cy = (current[-1][1] + current[-1][3]) / 2
        curr_cy = (box[1] + box[3]) / 2
        if abs(curr_cy - prev_cy) <= _LINE_Y_TOLERANCE:
            current.append(box)
        else:
            lines.append(current)
            current = [box]
    lines.append(current)

    gray_orig = _to_grayscale(image_8bit)
    results = []

    for line in lines:
        if len(line) < _MIN_BLOBS_PER_LINE:
            continue

        y1 = max(0, min(b[1] for b in line) - _PAD_Y)
        y2 = min(crop_h, max(b[3] for b in line) + _PAD_Y)

        row_band = gray_orig[y1:y2, :]

        line_peak = int(row_band.max()) if row_band.size > 0 else 0
        if line_peak < _MIN_LINE_PEAK:
            continue

        bg_mean = float(np.mean(row_band)) if row_band.size > 0 else 0.0
        if bg_mean > _MAX_BG_MEAN:
            continue

        bright_cols = np.where(row_band.max(axis=0) > _BRIGHT_THRESHOLD)[0]

        if len(bright_cols) >= 2:
            x1 = max(0, int(bright_cols[0]) - _PAD_X)
            x2 = min(crop_w, int(bright_cols[-1]) + _PAD_X)
        else:
            x1 = max(0, min(b[0] for b in line) - _PAD_X)
            x2 = min(crop_w, max(b[2] for b in line) + _PAD_X)

        results.append([int(x1), int(y1), int(x2), int(y2)])

    return results
