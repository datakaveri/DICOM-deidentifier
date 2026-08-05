"""
bbox_visualize.py — saves a PNG of the DICOM frame with the detected PHI
regions drawn as bounding boxes, for visual audit alongside the DICOM output.
"""

import os

import cv2

from config import log


def save_bbox_image(gray_frame, phi_regions, out_path):
    """
    Draws each phi_regions[i]["bbox"] (x1, y1, x2, y2) as a red rectangle over
    `gray_frame` (8-bit grayscale) and writes the result to `out_path` as PNG.
    """
    display = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

    for region in phi_regions:
        x1, y1, x2, y2 = [int(v) for v in region["bbox"]]
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 0, 255), 2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, display)
    log.info(f"  Bounding-box visualization saved -> {out_path}")
