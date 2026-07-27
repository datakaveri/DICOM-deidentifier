"""
image_enhance.py — Stage 2: image enhancement. Generates contrast variants
of the 8-bit grayscale image to improve OCR detection rates.
"""

import numpy as np
import cv2


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

    # Full inversion — critical for light-text-on-dark-background
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

    # Morphological Top-Hat — isolates thin bright text strokes from varying bright backgrounds
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
