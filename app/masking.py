"""
masking.py — Stage 5: Character-level stroke isolation & Navier-Stokes pixel redaction.

Features:
  - VOI LUT & Inverse VOI LUT mapping for exact 16-bit / 8-bit visual reconstruction
  - Adaptive Character Stroke Segmentation with quality gate (auto-reduces aggressiveness
    when too much background texture is captured)
  - Gaussian-feathered edge blending to eliminate hard rectangular boundaries
  - Unified Character Stroke Inpainting across all image regions (border and anatomy alike)
  - Native 8-bit and 16-bit support with Navier-Stokes structure-preserving propagation
"""

import numpy as np
import cv2
import pydicom

from config import log


def _apply_voi_lut(pixels, ds):
    """
    Applies Rescale Slope/Intercept and VOI LUT (Window Center/Width) to convert
    original raw pixels to a standardized 8-bit visual representation (0-255).
    """
    slope = float(getattr(ds, "RescaleSlope", 1.0)) if ds else 1.0
    intercept = float(getattr(ds, "RescaleIntercept", 0.0)) if ds else 0.0
    rescaled = pixels.astype(np.float64) * slope + intercept

    wc_attr = getattr(ds, "WindowCenter", None) if ds else None
    ww_attr = getattr(ds, "WindowWidth", None) if ds else None

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
    slope = float(getattr(ds, "RescaleSlope", 1.0)) if ds else 1.0
    intercept = float(getattr(ds, "RescaleIntercept", 0.0)) if ds else 0.0

    wc_attr = getattr(ds, "WindowCenter", None) if ds else None
    ww_attr = getattr(ds, "WindowWidth", None) if ds else None

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


def _get_character_mask(roi_8bit, bbox_local=None, dilation_px=3):
    """
    Adaptive Character Stroke Segmentation with Quality Gate.

    Uses a tiered approach:
      Level 1 (Conservative): Top-Hat + Contrast only — works for clear text on
              uniform backgrounds (dark borders, light backgrounds)
      Level 2 (Aggressive): Adds Black-Hat + Adaptive threshold — used only when
              Level 1 doesn't find enough strokes, and only if the result doesn't
              capture too much background texture

    Quality Gate: If the mask covers >45% of the bbox area, the detection is
    capturing background texture (bone, muscle fibers, etc.) instead of just
    character strokes. In that case, fall back to the conservative mask.
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

    bbox_area = max(1, crop_h * crop_w)

    # ── Level 1: Conservative (Top-Hat + Contrast) ──────────────────────────
    k_size = max(5, min(15, (crop_h // 2) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))

    # Top-Hat: bright text on dark background
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
    _, mask_bright = cv2.threshold(tophat, 8, 255, cv2.THRESH_BINARY)

    # Relative contrast threshold (bright text only — conservative)
    bg_med = float(np.median(crop))
    crop_max = float(np.max(crop))
    mask_contrast = np.zeros_like(crop)
    if crop_max > bg_med + 10.0:
        t_contrast = bg_med + 0.22 * (crop_max - bg_med)
        _, mask_contrast = cv2.threshold(crop, min(245.0, t_contrast), 255, cv2.THRESH_BINARY)

    conservative_mask = cv2.bitwise_or(mask_bright, mask_contrast)
    conservative_ratio = np.sum(conservative_mask > 0) / float(bbox_area)

    # If conservative mask captures a reasonable amount of strokes (5-45%), use it
    if 0.02 < conservative_ratio <= 0.45:
        stroke_mask = conservative_mask
    elif conservative_ratio > 0.45:
        # Too much captured even at conservative level — use only Top-Hat
        # (the contrast threshold is too sensitive for this ROI)
        tophat_ratio = np.sum(mask_bright > 0) / float(bbox_area)
        if tophat_ratio <= 0.45:
            stroke_mask = mask_bright
        else:
            # Even Top-Hat alone is too aggressive (highly textured anatomy)
            # Use a higher threshold to only catch the strongest strokes
            _, stroke_mask = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
    else:
        # ── Level 2: Aggressive (add Black-Hat + Adaptive) ──────────────────
        # Conservative didn't find enough — try adding more signals

        # Black-Hat: dark text on light background
        blackhat = cv2.morphologyEx(denoised, cv2.MORPH_BLACKHAT, kernel)
        _, mask_dark = cv2.threshold(blackhat, 8, 255, cv2.THRESH_BINARY)

        # Adaptive threshold
        block_size = max(11, min(31, (crop_h // 3) | 1))
        mask_adaptive = cv2.adaptiveThreshold(
            denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, block_size, -10
        )

        # Dark contrast (inverted)
        crop_min = float(np.min(crop))
        if bg_med > crop_min + 10.0:
            t_dark = bg_med - 0.22 * (bg_med - crop_min)
            _, mc_dark = cv2.threshold(crop, max(10.0, t_dark), 255, cv2.THRESH_BINARY_INV)
            mask_contrast = cv2.bitwise_or(mask_contrast, mc_dark)

        aggressive_mask = cv2.bitwise_or(conservative_mask, mask_dark)
        aggressive_mask = cv2.bitwise_or(aggressive_mask, mask_adaptive)
        aggressive_mask = cv2.bitwise_or(aggressive_mask, mask_contrast)

        aggressive_ratio = np.sum(aggressive_mask > 0) / float(bbox_area)

        # Quality gate on aggressive mask
        if aggressive_ratio <= 0.45:
            stroke_mask = aggressive_mask
        else:
            # Aggressive captured too much texture — fall back to conservative
            stroke_mask = conservative_mask if conservative_ratio > 0.01 else mask_bright

    # Morphological close to fill tiny gaps in character strokes
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    stroke_closed = cv2.morphologyEx(stroke_mask, cv2.MORPH_CLOSE, k_close)

    # Final dilation to cover anti-aliased stroke edges
    k_dil_size = max(3, min(7, dilation_px * 2 + 1))
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil_size, k_dil_size))
    dilated = cv2.dilate(stroke_closed, k_dil, iterations=1)

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[by1:by2, bx1:bx2] = dilated
    return full_mask


def _inpaint_16bit(image_16, mask_8, radius=9, method=cv2.INPAINT_NS):
    """
    Runs cv2.inpaint on 16-bit or signed pixel arrays by scaling to 8-bit,
    inpainting, and scaling back smoothly.
    """
    raw_min = float(image_16.min())
    raw_max = float(image_16.max())
    if raw_max > raw_min:
        temp_8 = ((image_16.astype(np.float32) - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
        inpainted_8 = cv2.inpaint(temp_8, mask_8, radius, method)
        result = (
            inpainted_8.astype(np.float32) / 255.0 * (raw_max - raw_min) + raw_min
        ).astype(image_16.dtype)
    else:
        result = image_16.copy()
    return result


def _feather_blend(original_roi, inpainted_roi, mask, feather_px=4):
    """
    Gaussian-feathered edge blending between original and inpainted ROI.
    Creates a smooth, invisible transition instead of a hard rectangular boundary.
    """
    if feather_px < 1:
        # No feathering — just direct replacement
        result = original_roi.copy()
        result[mask > 0] = inpainted_roi[mask > 0]
        return result

    # Create a soft alpha mask by blurring the binary mask edges
    blur_size = feather_px * 2 + 1
    alpha = cv2.GaussianBlur(mask.astype(np.float32) / 255.0, (blur_size, blur_size), feather_px)
    alpha = np.clip(alpha, 0.0, 1.0)

    # Blend: result = alpha * inpainted + (1 - alpha) * original
    if original_roi.dtype != np.float32:
        orig_f = original_roi.astype(np.float32)
        inp_f = inpainted_roi.astype(np.float32)
        blended = alpha * inp_f + (1.0 - alpha) * orig_f
        return blended.astype(original_roi.dtype)
    else:
        return alpha * inpainted_roi + (1.0 - alpha) * original_roi


def redact_roi(cleaned, x1, y1, x2, y2, pad=10):
    """
    Unified character-level stroke inpainting for any ROI.
    Extracts character strokes and inpaints with Navier-Stokes neighbor reconstruction.

    Uses adaptive stroke detection with a quality gate to avoid over-masking
    textured anatomy, and Gaussian-feathered blending to eliminate hard edges.
    """
    h, w = cleaned.shape[:2]
    rx1, rx2 = max(0, x1 - pad), min(w, x2 + pad)
    ry1, ry2 = max(0, y1 - pad), min(h, y2 + pad)

    roi = cleaned[ry1:ry2, rx1:rx2].copy()
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 3 or roi_w < 3:
        return cleaned

    roi_min, roi_max = float(roi.min()), float(roi.max())
    if roi_max > roi_min:
        roi_8 = ((roi.astype(np.float32) - roi_min) / (roi_max - roi_min) * 255.0).astype(np.uint8)
    else:
        roi_8 = roi.astype(np.uint8)

    iy1 = y1 - ry1
    iy2 = y2 - ry1
    ix1 = x1 - rx1
    ix2 = x2 - rx1

    bbox_local = (ix1, iy1, ix2, iy2)
    char_mask = _get_character_mask(roi_8, bbox_local=bbox_local, dilation_px=3)
    crop_mask = char_mask[0:roi_h, 0:roi_w]

    bbox_area = max(1, (iy2 - iy1) * (ix2 - ix1))
    mask_ratio = np.sum(crop_mask > 0) / float(bbox_area)

    if not np.any(crop_mask > 0):
        # Fallback: no strokes found at all — use a gentle Telea inpainting
        # on the bbox area (Telea handles large rectangles better than NS)
        crop_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        crop_mask[iy1:iy2, ix1:ix2] = 255
        inpaint_method = cv2.INPAINT_TELEA
        inpaint_radius = 8
    elif mask_ratio > 0.50:
        # Quality gate triggered — too much was captured, likely background texture
        # Use Telea (which handles large masked areas more gracefully)
        inpaint_method = cv2.INPAINT_TELEA
        inpaint_radius = 8
    else:
        # Normal case: clean stroke mask — use Navier-Stokes for best quality
        inpaint_method = cv2.INPAINT_NS
        inpaint_radius = 10

    if roi.dtype != np.uint8:
        roi_inpainted = _inpaint_16bit(roi, crop_mask, radius=inpaint_radius, method=inpaint_method)
    else:
        roi_inpainted = cv2.inpaint(roi, crop_mask, inpaint_radius, inpaint_method)

    # Gaussian-feathered blend to eliminate hard rectangular edges
    blended = _feather_blend(roi, roi_inpainted, crop_mask, feather_px=3)

    cleaned[ry1:ry2, rx1:rx2] = blended
    return cleaned


# Backward-compatible alias
_redact_border_zone = redact_roi
_redact_anatomy_zone = redact_roi


def redact_pixels(image_array, phi_regions, ds=None):
    """
    Unified character-stroke level pixel redaction with Navier-Stokes neighbor propagation.
    Works identically across all regions without arbitrary border/anatomy distinction.

    Works on native 8-bit OR 16-bit pixel arrays without losing dynamic range.
    Returns (cleaned_array, combined_mask).
    """
    if not phi_regions:
        log.info("  [Stage 5] No PHI pixels to redact.")
        return image_array.copy(), np.zeros(image_array.shape[:2], dtype=np.uint8)

    cleaned = image_array.copy()
    h, w = cleaned.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    redact_count = 0

    for region in phi_regions:
        x1, y1, x2, y2 = region["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        combined_mask[y1:y2, x1:x2] = 255
        cleaned = redact_roi(cleaned, x1, y1, x2, y2)
        redact_count += 1
        log.info(f"  [Stage 5] Character stroke inpainting @ [{x1},{y1},{x2},{y2}]")

    log.info(
        f"  [Stage 5] Done. Total regions redacted: {redact_count} | "
        f"Total pixels masked: {int(np.sum(combined_mask > 0))}"
    )
    return cleaned, combined_mask
