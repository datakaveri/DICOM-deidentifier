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
_BRIGHT_THRESHOLD = 130

# Local-contrast gate (replaces a flat "background must be dark" cutoff, which
# rejected any line sitting over bright anatomy instead of just the margin).
# Background is now sampled only in a margin around the text blobs themselves,
# and the line is kept if it's brighter than THAT local surrounding by at
# least _MIN_CONTRAST -- so text over bright tissue still passes as long as
# it stands out from the tissue immediately around it.
_LOCAL_BG_MARGIN = 40
_MIN_CONTRAST = 50

# Local adaptive-threshold pass (paired with the global Otsu pass in
# _binarize): Otsu alone assumes text is the minority class against a
# uniform background, which holds at the dark margin but not over anatomy,
# where local contrast is what marks text, not absolute brightness.
_ADAPTIVE_BLOCK = 31
_ADAPTIVE_C = -15


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
    Returns the two foreground masks (255 = candidate text pixel) separately
    instead of OR-ing them into one mask. Merging pixels before connected-
    component labeling lets a bridge of adaptive-threshold pixels (e.g. along
    a background/anatomy boundary) fuse an otherwise-isolated line of
    characters into one oversized blob, which then fails the char-size filter
    and silently disappears. Keeping the masks separate and running
    connected components on each independently avoids that.
    """
    gray = _to_grayscale(image)
    gray2x = cv2.resize(gray, None, fx=_SCALE, fy=_SCALE, interpolation=cv2.INTER_CUBIC)

    _, otsu = cv2.threshold(gray2x, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.mean(otsu) < 127:
        otsu = cv2.bitwise_not(otsu)
    fg_otsu = cv2.bitwise_not(otsu)

    # Bright-relative-to-its-own-neighborhood pass: catches text sitting over
    # anatomy, where Otsu's single whole-image threshold sees the text as part
    # of the same bright class as the surrounding tissue.
    fg_adaptive = cv2.adaptiveThreshold(
        gray2x, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
        _ADAPTIVE_BLOCK, _ADAPTIVE_C
    )

    return fg_otsu, fg_adaptive


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

    fg_otsu, fg_adaptive = _binarize(image_8bit)

    char_boxes = []
    seen = set()

    for fg in (fg_otsu, fg_adaptive):
        n, _, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)

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
                box = (x1_c, y1_c, x2_c, y2_c)
                if box not in seen:
                    seen.add(box)
                    char_boxes.append(box)

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

        # Background sampled only in a margin around this line's blobs, not
        # the whole row width, so bright anatomy elsewhere in the image can't
        # sink a line that's genuinely brighter than what's immediately
        # around it.
        line_x1 = min(b[0] for b in line)
        line_x2 = max(b[2] for b in line)
        bg_x1 = max(0, line_x1 - _LOCAL_BG_MARGIN)
        bg_x2 = min(crop_w, line_x2 + _LOCAL_BG_MARGIN)
        local_band = gray_orig[y1:y2, bg_x1:bg_x2]
        bg_mean = float(np.mean(local_band)) if local_band.size > 0 else 0.0
        if (line_peak - bg_mean) < _MIN_CONTRAST:
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
