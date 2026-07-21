"""
masking.py — Stage 5: zone-aware multi-hybrid pixel redaction.

BORDER zone  -> Uniform median fill (fast, complete, safe in flat margins)
ANATOMY zone -> 4-pass hybrid:
                 1. Precise ink isolation (Otsu+Adaptive fusion)
                 2. Navier-Stokes inpainting (structure-aware, r=9)
                 3. Telea inpainting on residual pixels (r=5)
                 4. Bilateral texture blend (edge-preserving smooth)
"""

import numpy as np
import cv2

from config import log


def _get_ink_mask(roi_8bit, dilation_px=2):
    """
    Extracts a precise binary mask of text ink strokes from an 8-bit ROI.
    Uses Otsu binarization + morphological cleanup + optional MSER fallback.
    dilation_px: how many pixels to expand the ink mask outward (captures edges).
    """
    # Primary: Otsu global threshold
    _, mask_otsu = cv2.threshold(roi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If text is bright on dark background, otsu gives us the bright blobs
    # If text is dark on bright background, invert
    mean_val = np.mean(roi_8bit)
    if mean_val > 127:
        mask_otsu = cv2.bitwise_not(mask_otsu)   # dark ink on bright bg

    # Secondary: Adaptive threshold (catches faint ink missed by Otsu)
    mask_adapt = cv2.adaptiveThreshold(
        roi_8bit, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )
    if mean_val > 127:
        mask_adapt = cv2.bitwise_not(mask_adapt)

    # Combine: any pixel flagged by either method
    combined = cv2.bitwise_or(mask_otsu, mask_adapt)

    # Morphological cleanup: remove noise specks (open), then dilate to cover edges
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    k_dilat = cv2.getStructuringElement(cv2.MORPH_RECT, (dilation_px + 1, dilation_px + 1))
    cleaned = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  k_open)
    dilated = cv2.dilate(cleaned, k_dilat, iterations=1)

    return dilated


def _inpaint_16bit(image_16, mask_8, radius, method):
    """
    Runs cv2.inpaint on any non-8-bit image (uint16, signed int16, etc. --
    formats OpenCV's inpaint doesn't accept directly) by temporarily scaling
    to 8-bit using the image's OWN value range (not an assumed unsigned
    range, which would be wrong for signed data), inpainting, then scaling
    back to that same original range and dtype.
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
        # Flat ROI -- nothing for inpainting to do.
        result = image_16.copy()
    return result


def _blend_boundary(image, roi_slice, feather_px=4):
    """
    Applies a Gaussian-feathered blend at the boundary of an inpainted ROI
    so the fill merges smoothly into surrounding anatomy.
    roi_slice: (y1, y2, x1, x2) of the inpainted region.
    """
    y1, y2, x1, x2 = roi_slice
    h, w = image.shape[:2]
    pad = feather_px

    # Slightly expanded context region
    ey1 = max(0, y1 - pad); ey2 = min(h, y2 + pad)
    ex1 = max(0, x1 - pad); ex2 = min(w, x2 + pad)

    region = image[ey1:ey2, ex1:ex2].copy()
    if region.dtype == np.uint16:
        region_f = region.astype(np.float32) / 65535.0
        blurred  = cv2.GaussianBlur(region_f, (2 * feather_px + 1, 2 * feather_px + 1), 0)
        image[ey1:ey2, ex1:ex2] = (blurred * 65535.0).astype(np.uint16)
    else:
        blurred = cv2.GaussianBlur(region, (2 * feather_px + 1, 2 * feather_px + 1), 0)
        image[ey1:ey2, ex1:ex2] = blurred

    return image


def _redact_border_zone(cleaned, x1, y1, x2, y2):
    """
    BORDER ZONE STRATEGY: Background uniform fill.
    Safe to use in plain black/white border margins where there is no anatomy.
    Fills with the local surrounding median to match the margin tone.
    """
    h, w = cleaned.shape[:2]
    pad  = 30
    ctx  = cleaned[
        max(0, y1 - pad):min(h, y2 + pad),
        max(0, x1 - pad):min(w, x2 + pad)
    ]
    fill_val = int(np.median(ctx))
    cleaned[y1:y2, x1:x2] = fill_val
    return cleaned


def _redact_anatomy_zone(cleaned, x1, y1, x2, y2):
    """
    ANATOMY ZONE — MULTI-HYBRID 4-PASS STRATEGY:

    Pass 1 (Ink Isolation):
        Otsu + Adaptive threshold fusion to get a precise binary ink mask.
        Only the ink strokes are marked for inpainting, NOT the full box.
        This preserves underlying anatomy pixels not covered by ink.

    Pass 2 (Navier-Stokes Inpainting, radius=9):
        Structure-aware fill that propagates texture from the boundary inward.
        Best for anatomical gradients (ribs, lung parenchyma).

    Pass 3 (Telea Inpainting on Residual, radius=5):
        Fast marching method on any ink pixels missed by NS pass.
        Handles isolated thin strokes that NS may leave.

    Pass 4 (Bilateral Texture Blend):
        Bilateral filter on the inpainted region to smooth the fill
        and match the local frequency/texture of surrounding anatomy.
        Edge-preserving — does not blur bone boundaries.
    """
    h, w = cleaned.shape[:2]
    pad = 6  # extra context around box for better inpaint propagation

    # Safe bounds
    rx1 = max(0, x1 - pad); rx2 = min(w, x2 + pad)
    ry1 = max(0, y1 - pad); ry2 = min(h, y2 + pad)

    roi = cleaned[ry1:ry2, rx1:rx2]
    roi_h, roi_w = roi.shape[:2]

    # ── Pass 1: Precise ink mask ──────────────────────────────────────────
    roi_min, roi_max = roi.min(), roi.max()
    if roi_max > roi_min:
        roi_8 = ((roi - roi_min) / (roi_max - roi_min) * 255.0).astype(np.uint8)
    else:
        roi_8 = roi.astype(np.uint8)

    ink_mask = _get_ink_mask(roi_8, dilation_px=2)

    if not np.any(ink_mask > 0):
        # Fallback: no distinct ink found — use full box fill with median
        return _redact_border_zone(cleaned, x1, y1, x2, y2)

    # ── Pass 2: NS inpainting (structure-aware, radius=9) ─────────────────
    if roi.dtype != np.uint8:
        roi_ns = _inpaint_16bit(roi, ink_mask, radius=9, method=cv2.INPAINT_NS)
    else:
        roi_ns = cv2.inpaint(roi, ink_mask, 9, cv2.INPAINT_NS)

    # ── Pass 3: Telea inpainting on residual pixels ───────────────────────
    # Re-check residual mask on the NS result
    ns_min, ns_max = roi_ns.min(), roi_ns.max()
    if ns_max > ns_min:
        ns_8 = ((roi_ns - ns_min) / (ns_max - ns_min) * 255.0).astype(np.uint8)
    else:
        ns_8 = roi_ns.astype(np.uint8)

    residual_mask = _get_ink_mask(ns_8, dilation_px=1)
    # Only re-inpaint pixels that are still ink-like AND were in original mask
    residual_mask = cv2.bitwise_and(residual_mask, ink_mask)

    if np.any(residual_mask > 0):
        if roi_ns.dtype != np.uint8:
            roi_telea = _inpaint_16bit(roi_ns, residual_mask, radius=5, method=cv2.INPAINT_TELEA)
        else:
            roi_telea = cv2.inpaint(roi_ns, residual_mask, 5, cv2.INPAINT_TELEA)
    else:
        roi_telea = roi_ns

    # ── Pass 4: Bilateral texture blend ───────────────────────────────────
    if roi_telea.dtype != np.uint8:
        rt_min = float(roi_telea.min())
        rt_max = float(roi_telea.max())
        if rt_max > rt_min:
            roi_f = (roi_telea.astype(np.float32) - rt_min) / (rt_max - rt_min)
        else:
            roi_f = np.zeros(roi_telea.shape, dtype=np.float32)
        blended_f = cv2.bilateralFilter(roi_f, d=7, sigmaColor=0.05, sigmaSpace=5)
        # Only apply bilateral blend on the ink region, keep rest original
        blend_zone = (ink_mask > 0).astype(np.float32)
        blend_zone_3d = blend_zone  # single channel
        roi_final_f = blended_f * blend_zone_3d + roi_f * (1.0 - blend_zone_3d)
        if rt_max > rt_min:
            roi_final = (roi_final_f * (rt_max - rt_min) + rt_min).astype(roi_telea.dtype)
        else:
            roi_final = roi_telea.copy()
    else:
        roi_f = roi_telea.astype(np.float32)
        blended_f = cv2.bilateralFilter(roi_f, d=7, sigmaColor=12.0, sigmaSpace=5)
        blend_zone = (ink_mask > 0).astype(np.float32)
        roi_final_f = blended_f * blend_zone + roi_f * (1.0 - blend_zone)
        roi_final = np.clip(roi_final_f, 0, 255).astype(np.uint8)

    # Write processed ROI back into the full image
    cleaned[ry1:ry2, rx1:rx2] = roi_final

    return cleaned


def redact_pixels(image_array, phi_regions):
    """
    Zone-aware multi-hybrid pixel redaction.

    BORDER zone  -> Uniform median fill (fast, complete, safe in flat margins)
    ANATOMY zone -> 4-pass hybrid:
                     1. Precise ink isolation (Otsu+Adaptive fusion)
                     2. Navier-Stokes inpainting (structure-aware, r=9)
                     3. Telea inpainting on residual pixels (r=5)
                     4. Bilateral texture blend (edge-preserving smooth)

    Works on native 8-bit OR 16-bit pixel arrays without losing dynamic range.
    Returns (cleaned_array, combined_mask).
    """
    if not phi_regions:
        log.info("  [Stage 5] No PHI pixels to redact.")
        return image_array.copy(), np.zeros(image_array.shape[:2], dtype=np.uint8)

    cleaned = image_array.copy()
    h, w = cleaned.shape[:2]
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    border_count  = 0
    anatomy_count = 0

    for region in phi_regions:
        x1, y1, x2, y2 = region["bbox"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) <= 0 or (y2 - y1) <= 0:
            continue

        zone = region.get("zone", "border")  # default to border if unset
        combined_mask[y1:y2, x1:x2] = 255

        if zone == "border":
            # ── BORDER: simple uniform fill ───────────────────────────────
            cleaned = _redact_border_zone(cleaned, x1, y1, x2, y2)
            border_count += 1
            log.info(f"  [Stage 5] Border fill applied @ [{x1},{y1},{x2},{y2}]")
        else:
            # ── ANATOMY: 4-pass hybrid masking ────────────────────────────
            cleaned = _redact_anatomy_zone(cleaned, x1, y1, x2, y2)
            anatomy_count += 1
            log.info(f"  [Stage 5] Anatomy hybrid masking @ [{x1},{y1},{x2},{y2}]")

    log.info(
        f"  [Stage 5] Done. Border fills: {border_count} | "
        f"Anatomy hybrid: {anatomy_count} | "
        f"Total pixels masked: {int(np.sum(combined_mask > 0))}"
    )
    return cleaned, combined_mask
