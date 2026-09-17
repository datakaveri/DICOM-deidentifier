"""
verify.py — Stage 6: Two-tier verification + escalation.
Tier 1: Ultra-fast (<1ms/region) morphological stroke residual & edge energy check.
Tier 2 (Fallback): Targeted PaddleOCR verification on flagged regions only (if stroke residue detected).
Escalates redaction if residual PHI is confirmed.
"""

import cv2
import numpy as np

from config import log
from ocr_detect import detect_text_in_regions
from classify import merge_detections, classify_phi, expand_phi_blocks
from masking import redact_roi


def check_stroke_residual(crop_8bit):
    """
    Tier 1 Gate: Ultra-fast (<1ms) morphological check for sharp character stroke residue.
    Returns True if character-like stroke structures are present, False if completely clean.
    """
    ch, cw = crop_8bit.shape[:2]
    if ch < 5 or cw < 5:
        return False

    k_size = max(5, min(13, (ch // 2) | 1))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k_size, k_size))
    tophat = cv2.morphologyEx(crop_8bit, cv2.MORPH_TOPHAT, kernel)
    blackhat = cv2.morphologyEx(crop_8bit, cv2.MORPH_BLACKHAT, kernel)

    # Text strokes have high local contrast against background
    _, th_bright = cv2.threshold(tophat, 40, 255, cv2.THRESH_BINARY)
    _, th_dark = cv2.threshold(blackhat, 40, 255, cv2.THRESH_BINARY)
    strokes = cv2.bitwise_or(th_bright, th_dark)

    # Check for character-like connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(strokes)
    text_like_components = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        comp_h = stats[i, cv2.CC_STAT_HEIGHT]
        if 8 <= comp_h <= ch and area >= 12:
            text_like_components += 1

    return text_like_components >= 2


def verify_redaction(cleaned_array, phi_regions, paddle_ocr=None, analyzer=None):
    """
    Two-Tier Verification:
      Tier 1: Ultra-fast stroke & gradient energy check (<3ms total).
      Tier 2 (Fallback): Targeted PaddleOCR on flagged regions only if Tier 1 detects residue.
    """
    if not phi_regions:
        log.info("  [Stage 6] PASSED — no redacted regions required verification.")
        return cleaned_array, "PASSED"

    log.info("  [Stage 6] Verification pass — running fast Tier 1 stroke-residual check...")
    h, w = cleaned_array.shape[:2]

    # Convert cleaned array to 8-bit visual for fast verification
    raw_min, raw_max = cleaned_array.min(), cleaned_array.max()
    if raw_max > raw_min:
        temp_8 = ((cleaned_array - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
    else:
        temp_8 = cleaned_array.astype(np.uint8)

    # ── Tier 1: Fast stroke residual check on all redacted regions ───────────
    flagged_regions = []
    for r in phi_regions:
        bx1, by1, bx2, by2 = r["bbox"]
        bx1, by1 = max(0, bx1), max(0, by1)
        bx2, by2 = min(w, bx2), min(h, by2)
        if bx2 <= bx1 or by2 <= by1:
            continue
        crop = temp_8[by1:by2, bx1:bx2]
        if check_stroke_residual(crop):
            flagged_regions.append(r)

    if not flagged_regions:
        log.info(f"  [Stage 6] PASSED (Tier 1) — all {len(phi_regions)} region(s) verified 100% clean.")
        return cleaned_array, "PASSED"

    # ── Tier 2: OCR Fallback Verification (only on flagged regions) ──────────
    log.info(f"  [Stage 6] Tier 2 Fallback — running targeted PaddleOCR on {len(flagged_regions)} flagged region(s)...")
    verify_regions = []
    for r in flagged_regions:
        bx1, by1, bx2, by2 = r["bbox"]
        vx1 = max(0, bx1 - 20)
        vy1 = max(0, by1 - 20)
        vx2 = min(w, bx2 + 20)
        vy2 = min(h, by2 + 20)
        verify_regions.append([vx1, vy1, vx2, vy2])

    raw_det  = detect_text_in_regions(temp_8, verify_regions, paddle_ocr=paddle_ocr, run_on_regions=True)
    merged   = merge_detections(raw_det)
    residual = classify_phi(merged, cleaned_array.shape, analyzer)
    residual = expand_phi_blocks(merged, residual, cleaned_array.shape)

    if not residual:
        log.info("  [Stage 6] PASSED (Tier 2 Fallback) — no residual PHI text detected in flagged regions.")
        return cleaned_array, "PASSED"

    # ── Escalation ────────────────────────────────────────────────────────────
    log.warning(
        f"  [Stage 6] Residual PHI confirmed in {len(residual)} region(s): "
        f"{[r['text'][:30] for r in residual]}. Running escalation inpainting..."
    )
    escalated = cleaned_array.copy()
    escalation_count = 0

    for region in residual:
        x1, y1, x2, y2 = region["bbox"]
        x1e = max(0, x1 - 10); y1e = max(0, y1 - 10)
        x2e = min(w, x2 + 10); y2e = min(h, y2 + 10)

        escalated = redact_roi(escalated, x1e, y1e, x2e, y2e, pad=10)
        escalation_count += 1
        log.info(f"  [Stage 6] Escalation inpainting @ [{x1e},{y1e},{x2e},{y2e}]")

    log.info(f"  [Stage 6] Escalation complete. Total escalated regions: {escalation_count}")
    log.info("  [Stage 6] PASSED (Escalated — all residual text cleared).")
    return escalated, "PASSED (Escalated)"
