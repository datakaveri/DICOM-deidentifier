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


def _fill_mask_holes(mask):
    """
    Fills interior holes inside character loops (e.g. inside O, D, R, A, 0, B)
    so OpenCV inpainting never samples dark interior stroke outlines.
    """
    cnts, hier = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    filled = mask.copy()
    if hier is not None:
        for i, h in enumerate(hier[0]):
            if h[3] != -1:  # Enclosed hole contour
                cv2.drawContours(filled, cnts, i, 255, -1)
    return filled


def _get_character_mask(roi_8bit, bbox_local=None, dilation_px=2):
    """
    Dual-Polarity Character Stroke Segmentation with Hole Sealing.
    Accurately isolates characters across all contrast environments:
      1. Bright text on darker background (Top-Hat high pass)
      2. Dark text on bright background (Black-Hat high pass)
      3. Drop-shadows and dark outlines
      4. Interior hole sealing preventing dark pinhole artifacts.
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

    # 7x7 Top-hat (bright strokes) & Black-hat (dark strokes / outlines)
    k_hat = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    th = cv2.morphologyEx(crop, cv2.MORPH_TOPHAT, k_hat)
    bh = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, k_hat)

    # Sensitive threshold range (4 to 10) ensures faint / ghosted text is also captured
    th_t = max(4, min(10, int(np.percentile(th, 90) * 0.22)))
    bh_t = max(4, min(10, int(np.percentile(bh, 90) * 0.22)))

    box_mask = ((th > th_t) | (bh > bh_t)).astype(np.uint8) * 255

    # Seal interior pinholes inside character loops
    box_mask = _fill_mask_holes(box_mask)

    # Anti-aliased elliptical dilation covers stroke boundaries and anti-aliasing halo
    k_dil_size = max(5, min(9, dilation_px * 2 + 1))
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil_size, k_dil_size))
    dilated = cv2.dilate(box_mask, k_dil, iterations=1)

    # Safety: if very faint text, supplement with morphological gradient
    if np.sum(dilated > 0) < int(crop_h * crop_w * 0.04):
        grad = cv2.morphologyEx(crop, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        g_t = max(6, int(np.percentile(grad, 90) * 0.30))
        grad_mask = ((grad > g_t).astype(np.uint8) * 255)
        grad_mask = cv2.dilate(grad_mask, k_dil, iterations=1)
        dilated = cv2.bitwise_or(dilated, grad_mask)

    # Snap character strokes touching outer image boundaries so no tick residue remains
    if by1 == 0:
        top_active = np.any(dilated[0:min(8, crop_h), :] > 0, axis=0)
        dilated[0:min(6, crop_h), top_active] = 255
    if bx1 == 0:
        left_active = np.any(dilated[:, 0:min(8, crop_w)] > 0, axis=1)
        dilated[left_active, 0:min(6, crop_w)] = 255
    if by2 >= h:
        bot_active = np.any(dilated[max(0, crop_h - 8):crop_h, :] > 0, axis=0)
        dilated[max(0, crop_h - 6):crop_h, bot_active] = 255
    if bx2 >= w:
        right_active = np.any(dilated[:, max(0, crop_w - 8):crop_w] > 0, axis=1)
        dilated[right_active, max(0, crop_w - 6):crop_w] = 255

    full_mask = np.zeros((h, w), dtype=np.uint8)
    full_mask[by1:by2, bx1:bx2] = dilated
    return full_mask


def _inpaint_16bit(image_16, mask_8, radius=5, method=cv2.INPAINT_TELEA):
    """
    High-fidelity inpainting for 16-bit and integer pixel arrays.
    Scales to 8-bit using the dynamic range of the UNMASKED BACKGROUND
    (mask_8 == 0) rather than the bright text spikes. Only the masked stroke
    pixels are replaced, ensuring original unmasked tissue is preserved
    at full 16-bit precision with zero quantization smudging.
    """
    bg_mask = (mask_8 == 0)
    if not np.any(bg_mask):
        return image_16

    bg_pixels = image_16[bg_mask]
    bg_min = float(np.percentile(bg_pixels, 0.5))
    bg_max = float(np.percentile(bg_pixels, 99.5))

    if bg_max <= bg_min:
        bg_min = float(bg_pixels.min())
        bg_max = float(bg_pixels.max())

    if bg_max <= bg_min:
        result = image_16.copy()
        result[mask_8 > 0] = int(bg_min)
        return result

    # Normalize image based on background dynamic range
    temp_8 = np.clip((image_16.astype(np.float32) - bg_min) / (bg_max - bg_min) * 255.0, 0, 255).astype(np.uint8)

    inpainted_8 = cv2.inpaint(temp_8, mask_8, radius, method)

    result = image_16.copy()
    update_mask = (mask_8 > 0)
    result[update_mask] = (
        inpainted_8[update_mask].astype(np.float32) / 255.0 * (bg_max - bg_min) + bg_min
    ).astype(image_16.dtype)
    return result


def redact_roi(cleaned, x1, y1, x2, y2, pad=14):
    """
    Unified character-stroke level inpainting with neighbor gradient reconstruction.
    Applied uniformly to all regions (soft tissue, anatomy, and margins alike).
    """
    h, w = cleaned.shape[:2]
    # Edge boundary snap: if box is within 14px of image border, snap cleanly to border
    if y1 <= 14:
        y1 = 0
    if x1 <= 14:
        x1 = 0
    if y2 >= h - 14:
        y2 = h
    if x2 >= w - 14:
        x2 = w

    rx1, rx2 = max(0, x1 - pad), min(w, x2 + pad)
    ry1, ry2 = max(0, y1 - pad), min(h, y2 + pad)

    roi = cleaned[ry1:ry2, rx1:rx2].copy()
    roi_h, roi_w = roi.shape[:2]
    if roi_h < 3 or roi_w < 3:
        return cleaned

    roi_min, roi_max = float(roi.min()), float(roi.max())

    # ── High-Fidelity Character-Stroke Inpainting ──────────────────────────
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
        # Fallback: Otsu threshold inside the target box to isolate text strokes
        box_crop = roi_8[iy1:iy2, ix1:ix2]
        if box_crop.size > 0:
            _, otsu_t = cv2.threshold(box_crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            crop_mask[iy1:iy2, ix1:ix2] = cv2.dilate(otsu_t, k_dil, iterations=1)
        if not np.any(crop_mask > 0):
            crop_mask[iy1:iy2, ix1:ix2] = 255

    # Guarantee zero boundary residue if region touches image edge
    if ry1 == 0:
        crop_mask[0:min(8, roi_h), ix1:ix2] = 255
    if rx1 == 0:
        crop_mask[iy1:iy2, 0:min(8, roi_w)] = 255
    if ry2 == h:
        crop_mask[max(0, roi_h - 8):roi_h, ix1:ix2] = 255
    if rx2 == w:
        crop_mask[iy1:iy2, max(0, roi_w - 8):roi_w] = 255

    inpaint_radius = 5
    if roi.dtype != np.uint8:
        roi_inpainted = _inpaint_16bit(roi, crop_mask, radius=inpaint_radius, method=cv2.INPAINT_TELEA)
    else:
        temp_inp = cv2.inpaint(roi, crop_mask, inpaint_radius, cv2.INPAINT_TELEA)
        roi_inpainted = roi.copy()
        roi_inpainted[crop_mask > 0] = temp_inp[crop_mask > 0]

    cleaned[ry1:ry2, rx1:rx2] = roi_inpainted
    return cleaned


# Unified aliases
_redact_border_zone = redact_roi
_redact_anatomy_zone = redact_roi


def _cluster_bboxes(regions, margin_x=8, margin_y=2):
    """
    Clusters horizontally overlapping or closely adjacent bounding boxes on the same line.
    Preserves clean separation between vertically stacked lines so inpainting interpolates
    from real tissue immediately above and below each line.
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
                e1 = [b1[0] - margin_x, b1[1] - margin_y, b1[2] + margin_x, b1[3] + margin_y]
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
    Unified character-stroke level pixel redaction with neighbor propagation.
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

    # Cluster horizontally adjacent bboxes on same lines (preserving line separation)
    clusters = _cluster_bboxes(phi_regions, margin_x=8, margin_y=4)
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

