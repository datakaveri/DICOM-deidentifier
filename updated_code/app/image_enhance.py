"""
image_enhance.py — Stage 2: image enhancement. Generates contrast variants
of the 8-bit grayscale image to improve OCR detection rates.
"""

import numpy as np
import cv2


def enhance_image(image_8bit):
    """
    Takes an 8-bit grayscale image (already normalized from 16-bit).
    Returns a dict of enhanced variants: standard (with others disabled for speed).
    Uncomment lines below if you need advanced variants for low-contrast text.
    """
    variants = {
        "standard": image_8bit,

        # --- UNCOMMENT BELOW TO ENABLE ADVANCED VARIANTS ---
        # "clahe": cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(image_8bit),
        # "adaptive_thresh": cv2.adaptiveThreshold(image_8bit, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 5),
        # "inverse_thresh": cv2.threshold(image_8bit, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        # "gamma": cv2.LUT(image_8bit, np.array([((i / 255.0) ** (1.0 / 0.5)) * 255 for i in range(256)], dtype=np.uint8)),
        # "sharpened": cv2.addWeighted(image_8bit, 1.5, cv2.GaussianBlur(image_8bit, (3, 3), 0), -0.5, 0),
        # "inverted": cv2.bitwise_not(image_8bit),
        # "inverted_clahe": cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(cv2.bitwise_not(image_8bit)),
        # "tophat": cv2.normalize(cv2.morphologyEx(image_8bit, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))), None, 0, 255, cv2.NORM_MINMAX),
        # "blackhat": cv2.normalize(cv2.morphologyEx(image_8bit, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))), None, 0, 255, cv2.NORM_MINMAX),
    }

    return variants

