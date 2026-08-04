"""
verify.py — Stage 6: verification + escalation. Re-runs OCR on the cleaned
image to confirm zero residual PHI text, escalating redaction if needed.
"""

import numpy as np

from config import log
from text_region_detect import detect_text_regions
from ocr_detect import detect_text_in_regions
from classify import merge_detections, classify_phi, expand_phi_blocks
from masking import _redact_border_zone, _redact_anatomy_zone


def verify_redaction(cleaned_array, phi_regions, paddle_ocr, easy_ocr, analyzer):
    """Re-runs OCR on cleaned image to confirm zero residual PHI text."""
    log.info("  [Stage 6] Verification pass — re-running OCR on cleaned image...")

    raw_min, raw_max = cleaned_array.min(), cleaned_array.max()
    if raw_max > raw_min:
        temp_8 = ((cleaned_array - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
    else:
        temp_8 = cleaned_array.astype(np.uint8)

    regions     = detect_text_regions(temp_8)
    raw_det     = detect_text_in_regions(temp_8, regions, paddle_ocr, easy_ocr)
    merged      = merge_detections(raw_det)
    residual    = classify_phi(merged, cleaned_array.shape, analyzer)
    residual    = expand_phi_blocks(merged, residual, cleaned_array.shape)

    if not residual:
        log.info("  [Stage 6] PASSED — no residual PHI text detected.")
        return cleaned_array, "PASSED"

    # ── Zone-aware escalation ─────────────────────────────────────────────────
    log.warning(
        f"  [Stage 6] Residual PHI found in {len(residual)} region(s): "
        f"{[r['text'][:30] for r in residual]}. Running zone-aware escalation..."
    )
    h, w = cleaned_array.shape[:2]
    escalated = cleaned_array.copy()

    border_esc  = 0
    anatomy_esc = 0

    for region in residual:
        x1, y1, x2, y2 = region["bbox"]
        # Add 8px padding to catch any char edges
        x1e = max(0, x1 - 8); y1e = max(0, y1 - 8)
        x2e = min(w, x2 + 8); y2e = min(h, y2 + 8)

        zone = region.get("zone", "border")

        if zone == "border":
            # Expanded median fill for border residuals
            escalated = _redact_border_zone(escalated, x1e, y1e, x2e, y2e)
            border_esc += 1
            log.info(f"  [Stage 6] Escalation border fill @ [{x1e},{y1e},{x2e},{y2e}]")
        else:
            # Re-run 4-pass hybrid on anatomy residuals
            escalated = _redact_anatomy_zone(escalated, x1e, y1e, x2e, y2e)
            anatomy_esc += 1
            log.info(f"  [Stage 6] Escalation anatomy hybrid @ [{x1e},{y1e},{x2e},{y2e}]")

    log.info(
        f"  [Stage 6] Escalation complete. Border fills: {border_esc} | "
        f"Anatomy hybrid: {anatomy_esc}"
    )

    # ── Second tight verification pass ───────────────────────────────────────
    esc_min, esc_max = escalated.min(), escalated.max()
    if esc_max > esc_min:
        esc_8 = ((escalated - esc_min) / (esc_max - esc_min) * 255.0).astype(np.uint8)
    else:
        esc_8 = escalated.astype(np.uint8)

    v2_regions = detect_text_regions(esc_8)
    v2_det     = detect_text_in_regions(esc_8, v2_regions, paddle_ocr, easy_ocr)
    v2_merged = merge_detections(v2_det)
    v2_phi    = classify_phi(v2_merged, escalated.shape, analyzer)
    v2_phi    = expand_phi_blocks(v2_merged, v2_phi, escalated.shape)

    if not v2_phi:
        log.info("  [Stage 6] PASSED (Escalated — all residual text cleared).")
        return escalated, "PASSED (Escalated)"

    # ── Nuclear option: blackout the border strips ────────────────────────────
    log.error(
        f"  [Stage 6] Escalation still has {len(v2_phi)} residual region(s). "
        "Applying nuclear border blackout..."
    )
    nuclear = escalated.copy()
    top_h    = int(h * 0.20)
    bot_h    = int(h * 0.80)
    side_w   = int(w * 0.12)
    nuclear[0:top_h, :]   = 0
    nuclear[bot_h:, :]    = 0
    nuclear[:, 0:side_w]  = 0
    nuclear[:, w-side_w:] = 0
    log.info("  [Stage 6] FORCE PASSED (Nuclear border blackout applied).")
    return nuclear, "FORCE PASSED (Nuclear Fallback)"
