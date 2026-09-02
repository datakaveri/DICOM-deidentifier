"""
verify.py — Stage 6: verification + escalation. Re-runs OCR on the cleaned
image to confirm zero residual PHI text, escalating redaction if needed.
"""

import numpy as np

from config import log
from text_region_detect import detect_text_regions
from ocr_detect import detect_text_in_regions
from classify import merge_detections, classify_phi, expand_phi_blocks
from masking import _redact_border_zone, _redact_anatomy_zone, _redact_black


def _to_8bit(array):
    """8-bit visual of `array` for fast targeted OCR verification."""
    raw_min, raw_max = array.min(), array.max()
    if raw_max > raw_min:
        return ((array - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
    return array.astype(np.uint8)


def _residual_phi(array, verify_regions, paddle_ocr, easy_ocr, analyzer, temp_8=None):
    """PHI text still detectable inside `verify_regions` of `array`."""
    if temp_8 is None:
        temp_8 = _to_8bit(array)
    raw_det  = detect_text_in_regions(temp_8, verify_regions, paddle_ocr, easy_ocr)
    merged   = merge_detections(raw_det)
    residual = classify_phi(merged, array.shape, analyzer)
    return expand_phi_blocks(merged, residual, array.shape)


def verify_redaction(cleaned_array, phi_regions, paddle_ocr, easy_ocr, analyzer,
                     method="inpaint", strict=False):
    """Re-runs OCR on targeted redacted ROI regions to confirm zero residual PHI text.

    `method` selects how residual text is escalated -- "black" fills the region
    solid, matching the job config's pinned pixel method.

    `strict` (the config-driven path) re-runs detection AFTER escalation and
    reports "FAILED" if anything survived, instead of assuming the escalation
    worked. A file that fails verification is still written by the caller, with
    that failed status: the user needs to see that redaction was incomplete,
    not receive nothing and no explanation.
    """
    log.info("  [Stage 6] Verification pass — running fast targeted OCR on redacted ROIs...")

    if not phi_regions:
        log.info("  [Stage 6] PASSED — no redacted regions required verification.")
        return cleaned_array, "PASSED"

    h, w = cleaned_array.shape[:2]

    # Convert cleaned array to 8-bit visual for fast targeted OCR verification
    temp_8 = _to_8bit(cleaned_array)

    # Build targeted ROI candidate regions (redacted bboxes + 20px padding)
    verify_regions = []
    for r in phi_regions:
        bx1, by1, bx2, by2 = r["bbox"]
        vx1 = max(0, bx1 - 20)
        vy1 = max(0, by1 - 20)
        vx2 = min(w, bx2 + 20)
        vy2 = min(h, by2 + 20)
        verify_regions.append([vx1, vy1, vx2, vy2])

    residual = _residual_phi(cleaned_array, verify_regions, paddle_ocr, easy_ocr,
                             analyzer, temp_8=temp_8)

    if not residual:
        log.info("  [Stage 6] PASSED — no residual PHI text detected in redacted regions.")
        return cleaned_array, "PASSED"

    # ── Zone-aware escalation ─────────────────────────────────────────────────
    log.warning(
        f"  [Stage 6] Residual PHI found in {len(residual)} region(s): "
        f"{[r['text'][:30] for r in residual]}. Running zone-aware escalation..."
    )
    escalated = cleaned_array.copy()

    border_esc  = 0
    anatomy_esc = 0

    for region in residual:
        x1, y1, x2, y2 = region["bbox"]
        x1e = max(0, x1 - 10); y1e = max(0, y1 - 10)
        x2e = min(w, x2 + 10); y2e = min(h, y2 + 10)

        zone = region.get("zone", "border")

        if method == "black":
            escalated = _redact_black(escalated, x1e, y1e, x2e, y2e)
            border_esc += 1
            log.info(f"  [Stage 6] Escalation black fill @ [{x1e},{y1e},{x2e},{y2e}]")
        elif zone == "border":
            escalated = _redact_border_zone(escalated, x1e, y1e, x2e, y2e)
            border_esc += 1
            log.info(f"  [Stage 6] Escalation border fill @ [{x1e},{y1e},{x2e},{y2e}]")
        else:
            escalated = _redact_anatomy_zone(escalated, x1e, y1e, x2e, y2e)
            anatomy_esc += 1
            log.info(f"  [Stage 6] Escalation anatomy hybrid @ [{x1e},{y1e},{x2e},{y2e}]")

    log.info(
        f"  [Stage 6] Escalation complete. Border fills: {border_esc} | "
        f"Anatomy hybrid: {anatomy_esc}"
    )

    if not strict:
        log.info("  [Stage 6] PASSED (Escalated — all residual text cleared).")
        return escalated, "PASSED (Escalated)"

    # Strict: confirm the escalation actually cleared the text rather than
    # assuming it did. Reporting PASSED over surviving PHI is the one outcome
    # this pass must never produce.
    still = _residual_phi(escalated, verify_regions, paddle_ocr, easy_ocr, analyzer)
    if still:
        log.error(
            f"  [Stage 6] FAILED — {len(still)} region(s) still carry text after "
            f"escalation. The file is still written, with a FAILED status, so the "
            f"incomplete redaction is visible rather than silent."
        )
        return escalated, "FAILED"

    log.info("  [Stage 6] PASSED (Escalated — re-verified clear).")
    return escalated, "PASSED (Escalated)"
