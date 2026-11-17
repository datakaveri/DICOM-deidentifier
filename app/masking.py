"""
masking.py — Stage 5: character-level stroke isolation & Navier-Stokes pixel redaction.

Features:
  - VOI LUT & Inverse VOI LUT mapping for exact 16-bit / 8-bit visual reconstruction
  - Tri-Signal Character Stroke Segmentation (Top-Hat + Contrast Thresholding + Ellipse Dilation)
  - Unified Neighbor Inpainting (structure-preserving, no quality loss to underlying anatomy)
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


def _get_character_mask(roi_8bit, bbox_local=None, dilation_px=2):
    """
    Tri-Signal Character Stroke Segmentation.
    Combines:
      1. Top-Hat High-Pass (bright text cores)
      2. Local Relative Contrast Thresholding
      3. Drop-Shadow / Dark Outline Capture (for stroked clinical fonts)
      4. Anti-Aliased Ellipse Dilation
    Captures complete glyphs without capturing background tissue or anatomy.
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
    _, mask_bright = cv2.threshold(tophat, 10, 255, cv2.THRESH_BINARY)

    bg_med = float(np.median(crop))
    crop_max = float(np.max(crop))
    if crop_max > bg_med + 12.0:
        t_contrast = bg_med + 0.25 * (crop_max - bg_med)
        _, mask_contrast = cv2.threshold(crop, min(245.0, t_contrast), 255, cv2.THRESH_BINARY)
    else:
        mask_contrast = np.zeros_like(crop)

    stroke_union = cv2.bitwise_or(mask_bright, mask_contrast)

    # ── Shadow / Dark Outline Capture ────────────────────────────────────────
    # Medical burned-in text commonly features an anti-aliased black border/outline
    # (near 0) around white characters. Including it in the mask prevents Navier-Stokes
    # from propagating dark border pixels into the character center.
    k_near = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    stroke_dil_near = cv2.dilate(stroke_union, k_near)
    shadow_thresh = min(30, max(5, int(bg_med * 0.4)))
    mask_shadow = (crop < shadow_thresh) & (stroke_dil_near > 0)
    stroke_union = cv2.bitwise_or(stroke_union, (mask_shadow.astype(np.uint8) * 255))

    # Morphological closing to seal pinhole gaps between strokes & shadow
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    stroke_union = cv2.morphologyEx(stroke_union, cv2.MORPH_CLOSE, k_close)

    # Elliptical dilation covers anti-aliasing without blurring adjacent letters
    k_dil_size = max(3, min(5, dilation_px * 2 + 1))
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


def redact_roi(cleaned, x1, y1, x2, y2, pad=10):
    """
    Unified character-stroke level inpainting with Navier-Stokes neighbor reconstruction.
    Applied uniformly to all regions (anatomy and border alike).
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
    char_mask = _get_character_mask(roi_8, bbox_local=bbox_local, dilation_px=2)
    crop_mask = char_mask[0:roi_h, 0:roi_w]

    if not np.any(crop_mask > 0):
        # Fallback if no distinct strokes: use local median fill in ROI
        cleaned[y1:y2, x1:x2] = int(np.median(roi))
        return cleaned

    if roi.dtype != np.uint8:
        roi_inpainted = _inpaint_16bit(roi, crop_mask, radius=7, method=cv2.INPAINT_NS)
    else:
        roi_inpainted = cv2.inpaint(roi, crop_mask, 7, cv2.INPAINT_NS)

    cleaned[ry1:ry2, rx1:rx2] = roi_inpainted
    return cleaned


# Unified aliases
_redact_border_zone = redact_roi
_redact_anatomy_zone = redact_roi


def _cluster_bboxes(regions, margin=8):
    """
    Clusters overlapping or closely adjacent bounding boxes into unified blocks.
    Prevents neighboring lines of text from leaking into each other's padding.
    """
    if not regions:
        return []

    boxes = [list(r["bbox"]) for r in regions]

    merged = True
    while merged:
        merged = False
        new_boxes = []
        visited = [False] * len(boxes)
        for i in range(len(boxes)):
            if visited[i]:
                continue
            b1 = list(boxes[i])
            visited[i] = True
            for j in range(i + 1, len(boxes)):
                if visited[j]:
                    continue
                b2 = boxes[j]
                e1 = [b1[0] - margin, b1[1] - margin, b1[2] + margin, b1[3] + margin]
                if (max(e1[0], b2[0]) < min(e1[2], b2[2]) and
                        max(e1[1], b2[1]) < min(e1[3], b2[3])):
                    b1 = [
                        min(b1[0], b2[0]),
                        min(b1[1], b2[1]),
                        max(b1[2], b2[2]),
                        max(b1[3], b2[3])
                    ]
                    visited[j] = True
                    merged = True
            new_boxes.append(b1)
        boxes = new_boxes

    return [{"bbox": b} for b in boxes]


def redact_pixels(image_array, phi_regions, ds=None):
    """
    Unified character-stroke level pixel redaction with Navier-Stokes neighbor propagation.
    Applies structure-preserving reconstruction across ALL regions uniformly.

    Works on native 8-bit OR 16-bit pixel arrays without losing dynamic range.
    Returns (cleaned_array, combined_mask).
    """
    if not phi_regions:
        log.info("  [Stage 5] No PHI pixels to redact.")
        return image_array.copy(), np.zeros(image_array.shape[:2], dtype=np.uint8)

    cleaned = image_array.copy()
    h, w = cleaned.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # Mark all individual region bboxes in the audit mask
    for region in phi_regions:
        x1, y1, x2, y2 = region["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) > 0 and (y2 - y1) > 0:
            combined_mask[y1:y2, x1:x2] = 255

    # Cluster overlapping/adjacent bboxes into unified blocks
    clusters = _cluster_bboxes(phi_regions, margin=8)
    log.info(f"  [Stage 5] Unified inpainting for {len(clusters)} region block(s)...")

    redact_count = 0
    for cluster in clusters:
        x1, y1, x2, y2 = cluster["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        cleaned = redact_roi(cleaned, x1, y1, x2, y2)
        redact_count += 1
        log.info(f"  [Stage 5] Unified character stroke inpainting @ [{x1},{y1},{x2},{y2}]")

    log.info(
        f"  [Stage 5] Done. Total blocks redacted: {redact_count} | "
        f"Total pixels masked: {int(np.sum(combined_mask > 0))}"
    )
    return cleaned, combined_mask
