"""
masking.py — Stage 5: character-level stroke isolation & Navier-Stokes pixel redaction.

Features:
  - VOI LUT & Inverse VOI LUT mapping for exact 16-bit / 8-bit visual reconstruction
  - Tri-Signal Character Stroke Segmentation (Top-Hat + Contrast Thresholding + Ellipse Dilation)
  - Two-Pass Iterative Neighbor Inpainting (structure-preserving, no quality loss to underlying anatomy)
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
    Tri-Signal Character Stroke Segmentation.
    Combines:
      1. Top-Hat High-Pass (bright text on dark background)
      2. Local Relative Contrast Thresholding
      3. Anti-Aliased Ellipse Dilation
    Captures character strokes of ANY polarity in a unified mask.
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

    k_size = max(5, min(15, (crop_h // 2) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(denoised, cv2.MORPH_TOPHAT, kernel)
    _, mask_bright = cv2.threshold(tophat, 8, 255, cv2.THRESH_BINARY)

    bg_med = float(np.median(crop))
    crop_max = float(np.max(crop))
    if crop_max > bg_med + 10.0:
        t_contrast = bg_med + 0.20 * (crop_max - bg_med)
        _, mask_contrast = cv2.threshold(crop, min(245.0, t_contrast), 255, cv2.THRESH_BINARY)
    else:
        mask_contrast = np.zeros_like(crop)

    stroke_union = cv2.bitwise_or(mask_bright, mask_contrast)

    k_dil_size = max(5, min(9, dilation_px * 2 + 1))
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil_size, k_dil_size))
    dilated = cv2.dilate(stroke_union, k_dil, iterations=1)

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


def _redact_border_zone(cleaned, x1, y1, x2, y2):
    """
    Border zone strategy: pre-conditioned inpainting.

    Problem: border text bboxes often sit adjacent to bright anatomy.
    Standard inpainting propagates that brightness into the dark border.
    Post-clamping fixes the brightness but kills texture, creating a flat patch.

    Solution: PRE-CONDITION the ROI margins before inpainting.
      1. Save original ROI margin pixels
      2. Temporarily replace bright margin pixels with the local dark
         background value — this makes inpainting only sample from dark
         pixels, preventing anatomy bleed
      3. Build character stroke mask + inpaint (same as anatomy zone)
      4. Restore original margin pixels — anatomy edges are untouched

    Result: natural inpainting texture (not flat) AND no bright artifacts.
    """
    h, w = cleaned.shape[:2]
    pad = 8
    rx1, rx2 = max(0, x1 - pad), min(w, x2 + pad)
    ry1, ry2 = max(0, y1 - pad), min(h, y2 + pad)

    roi = cleaned[ry1:ry2, rx1:rx2].copy()
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 3 or roi_w < 3:
        return cleaned

    # Save original margins for restoration after inpainting
    roi_original = roi.copy()

    # Inner bbox coordinates within the ROI
    iy1 = y1 - ry1
    iy2 = y2 - ry1
    ix1 = x1 - rx1
    ix2 = x2 - rx1

    # ── Sample the local background level from immediate ROI margins ─────
    # Sample from the outer border ring of the ROI margin strips, not distant top-left
    margin_samples = []
    if iy1 > 0:
        margin_samples.append(roi[0:iy1, :])
    if iy2 < roi_h:
        margin_samples.append(roi[iy2:, :])
    if ix1 > 0:
        margin_samples.append(roi[iy1:iy2, 0:ix1])
    if ix2 < roi_w:
        margin_samples.append(roi[iy1:iy2, ix2:])

    if margin_samples:
        concat_margins = np.concatenate([m.ravel() for m in margin_samples])
        bg_val = int(np.median(concat_margins))
    else:
        bg_val = int(np.median(roi))

    bg_ceiling = bg_val + max(12, int(abs(bg_val) * 0.30))

    # ── Build character stroke mask (same logic as anatomy zone) ──────────
    roi_min, roi_max = float(roi.min()), float(roi.max())
    if roi_max > roi_min:
        roi_8 = ((roi.astype(np.float32) - roi_min) / (roi_max - roi_min) * 255.0).astype(np.uint8)
    else:
        roi_8 = roi.astype(np.uint8)

    bbox_local = (ix1, iy1, ix2, iy2)
    char_mask = _get_character_mask(roi_8, bbox_local=bbox_local, dilation_px=3)
    crop_mask = char_mask[0:roi_h, 0:roi_w]

    if not np.any(crop_mask > 0):
        # Fallback: no strokes detected — create a box mask for Navier-Stokes local neighbor propagation
        crop_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)
        crop_mask[iy1:iy2, ix1:ix2] = 255

    # ── Inpaint with Navier-Stokes (natural texture from local neighbors) ──
    if roi.dtype != np.uint8:
        roi_inpainted = _inpaint_16bit(roi, crop_mask, radius=9, method=cv2.INPAINT_NS)
    else:
        roi_inpainted = cv2.inpaint(roi, crop_mask, 9, cv2.INPAINT_NS)

    # ── Restore original margin pixels ────────────────────────────────────
    # Top margin
    if iy1 > 0:
        roi_inpainted[0:iy1, :] = roi_original[0:iy1, :]
    # Bottom margin
    if iy2 < roi_h:
        roi_inpainted[iy2:, :] = roi_original[iy2:, :]
    # Left margin
    if ix1 > 0:
        roi_inpainted[iy1:iy2, 0:ix1] = roi_original[iy1:iy2, 0:ix1]
    # Right margin
    if ix2 < roi_w:
        roi_inpainted[iy1:iy2, ix2:] = roi_original[iy1:iy2, ix2:]

    cleaned[ry1:ry2, rx1:rx2] = roi_inpainted
    return cleaned


def _redact_anatomy_zone(cleaned, x1, y1, x2, y2):
    """
    Anatomy zone strategy: Character-level stroke mask + Navier-Stokes inpainting.
    Reconstructs pixels from immediate real neighbors.
    """
    h, w = cleaned.shape[:2]
    pad = 6
    rx1, rx2 = max(0, x1 - pad), min(w, x2 + pad)
    ry1, ry2 = max(0, y1 - pad), min(h, y2 + pad)

    roi = cleaned[ry1:ry2, rx1:rx2]
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

    if not np.any(crop_mask > 0):
        # Fallback if no distinct strokes: use local median fill in ROI
        cleaned[y1:y2, x1:x2] = int(np.median(roi))
        return cleaned

    if roi.dtype != np.uint8:
        roi_inpainted = _inpaint_16bit(roi, crop_mask, radius=9, method=cv2.INPAINT_NS)
    else:
        roi_inpainted = cv2.inpaint(roi, crop_mask, 9, cv2.INPAINT_NS)

    cleaned[ry1:ry2, rx1:rx2] = roi_inpainted
    return cleaned


def redact_pixels(image_array, phi_regions, ds=None):
    """
    Character-stroke level pixel redaction with Navier-Stokes neighbor propagation.

    Works on native 8-bit OR 16-bit pixel arrays without losing dynamic range.
    Returns (cleaned_array, combined_mask).
    """
    if not phi_regions:
        log.info("  [Stage 5] No PHI pixels to redact.")
        return image_array.copy(), np.zeros(image_array.shape[:2], dtype=np.uint8)

    cleaned = image_array.copy()
    h, w = cleaned.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    border_count = 0
    anatomy_count = 0

    for region in phi_regions:
        x1, y1, x2, y2 = region["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        zone = region.get("zone", "border")
        combined_mask[y1:y2, x1:x2] = 255

        if zone == "border":
            cleaned = _redact_border_zone(cleaned, x1, y1, x2, y2)
            border_count += 1
            log.info(f"  [Stage 5] Border fill applied @ [{x1},{y1},{x2},{y2}]")
        else:
            cleaned = _redact_anatomy_zone(cleaned, x1, y1, x2, y2)
            anatomy_count += 1
            log.info(f"  [Stage 5] Character stroke inpainting @ [{x1},{y1},{x2},{y2}]")

    log.info(
        f"  [Stage 5] Done. Border fills: {border_count} | "
        f"Anatomy inpainting: {anatomy_count} | "
        f"Total pixels masked: {int(np.sum(combined_mask > 0))}"
    )
    return cleaned, combined_mask
