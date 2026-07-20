"""
classify.py — Stage 4: merge overlapping OCR detections and classify each
one as PHI (redact) or safe (keep).
"""

import re

import numpy as np

from config import log, CLINICAL_ALLOWLIST, PII_PATTERNS


def _iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0.0


def merge_detections(raw, iou_thresh=0.3):
    """NMS-style merge of overlapping boxes from all OCR engine+variant passes."""
    if not raw:
        return []
    sorted_det = sorted(raw, key=lambda x: x["confidence"], reverse=True)
    merged = []
    while sorted_det:
        cur = sorted_det.pop(0)
        group = [cur]
        remaining = []
        for d in sorted_det:
            if _iou(cur["bbox"], d["bbox"]) > iou_thresh:
                group.append(d)
            else:
                remaining.append(d)
        sorted_det = remaining
        bboxes = np.array([g["bbox"] for g in group])
        best_text = max(group, key=lambda x: len(x["text"]))["text"]
        max_conf = max(g["confidence"] for g in group)
        merged.append({
            "text": best_text,
            "bbox": [int(bboxes[:,0].min()), int(bboxes[:,1].min()),
                     int(bboxes[:,2].max()), int(bboxes[:,3].max())],
            "confidence": min(1.0, max_conf),
        })
    return merged


def _is_clinical(text):
    clean = re.sub(r'[^A-Z]', '', text.upper())
    return clean in CLINICAL_ALLOWLIST


def classify_phi(merged, image_shape, analyzer=None):
    """
    Classifies each merged detection as PHI (redact) or safe (keep).
    Returns list of dicts: {text, bbox, zone}
      zone = 'border'  -> flat border region (aggressive box fill safe)
      zone = 'anatomy' -> overlaps anatomy (must use careful hybrid inpainting)
    """
    h, w = image_shape[:2]
    phi_regions = []

    for det in merged:
        text = det["text"]
        bbox = det["bbox"]

        # ── Size validation: discard full-frame/large OCR bounding box artifacts ──
        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        box_area = box_w * box_h
        img_area = w * h

        if box_h > h * 0.15 or box_w > w * 0.85 or box_area > (img_area * 0.10):
            log.warning(
                f"    SKIP OCR ARTIFACT (box too large): '{text}' @ {bbox} "
                f"(w={box_w}, h={box_h}, area={box_area}/{img_area})"
            )
            continue

        if _is_clinical(text):

            log.info(f"    KEEP (clinical marker): '{text}'")
            continue

        # ── Regex PII check ──────────────────────────────────────────────────
        regex_hit = any(
            re.search(p, text, re.IGNORECASE)
            for p in PII_PATTERNS.values()
        )

        # ── Presidio NLP check ───────────────────────────────────────────────
        nlp_hit = False
        if analyzer:
            try:
                results = analyzer.analyze(text=text, language="en")
                nlp_hit = any(
                    r.entity_type in {"PERSON", "DATE_TIME", "LOCATION",
                                      "EMAIL_ADDRESS", "PHONE_NUMBER"}
                    and r.score > 0.45
                    for r in results
                )
            except Exception:
                pass

        # ── Spatial zone detection ────────────────────────────────────────────
        y_mid = (bbox[1] + bbox[3]) / 2.0
        x_mid = (bbox[0] + bbox[2]) / 2.0

        in_border = (
            y_mid < h * 0.18 or y_mid > h * 0.82 or
            x_mid < w * 0.10 or x_mid > w * 0.90
        )
        in_anatomy = not in_border  # Anything NOT in border is overlaid on anatomy

        # ── Short text validation ─────────────────────────────────────────────
        clean_text = text.strip()
        if not clean_text:
            log.warning(f"    SKIP OCR ARTIFACT (empty text): @ {bbox}")
            continue

        if in_anatomy and len(clean_text) < 3:
            # Discard short 1-2 char noise in anatomy zone unless it matched regex/nlp
            if not (regex_hit or nlp_hit):
                log.info(f"    KEEP (short unknown text in anatomy): '{text}' @ {bbox}")
                continue

        # ── Decision ─────────────────────────────────────────────────────────
        # Border zone: redact if border (spatial), regex, or NLP hit
        # Anatomy zone: redact if regex, NLP, OR any unknown text (safe default)
        #   (Clinical allowlist already filters out L/R/PA etc.)
        if in_border and (regex_hit or nlp_hit or in_border):
            zone = "border"

            reason = ["border"]
            if regex_hit: reason.append("regex")
            if nlp_hit:   reason.append("nlp")
            log.info(f"    REDACT-BORDER ({', '.join(reason)}): '{text}' @ {bbox}")
            phi_regions.append({"text": text, "bbox": bbox, "zone": zone})

        elif in_anatomy and (regex_hit or nlp_hit):
            # Confirmed PHI inside anatomy (matched pattern/NLP) — must redact carefully
            zone = "anatomy"
            reason = []
            if regex_hit: reason.append("regex")
            if nlp_hit:   reason.append("nlp")
            log.info(f"    REDACT-ANATOMY ({', '.join(reason)}): '{text}' @ {bbox}")
            phi_regions.append({"text": text, "bbox": bbox, "zone": zone})

        elif in_anatomy and not regex_hit and not nlp_hit:
            # Unknown non-clinical text inside anatomy — treat as suspect PHI
            # Use anatomy zone masking (careful inpainting)
            log.info(f"    REDACT-ANATOMY (suspect unknown text): '{text}' @ {bbox}")
            phi_regions.append({"text": text, "bbox": bbox, "zone": "anatomy"})

        else:
            log.info(f"    KEEP (unmatched): '{text}'")

    return phi_regions
