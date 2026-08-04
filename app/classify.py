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


def merge_horizontal_lines(detections, max_gap_factor=2.5, min_v_overlap=0.45):
    """
    Merges detections that are on the same horizontal line and close to each other.
    Allows catching labeled PII like 'PATIENT:' + 'MEERA IYER' as a single entity.
    """
    if not detections:
        return []
        
    sorted_det = sorted(detections, key=lambda x: x["bbox"][0])
    merged_any = True
    while merged_any:
        merged_any = False
        i = 0
        while i < len(sorted_det):
            j = i + 1
            while j < len(sorted_det):
                box1 = sorted_det[i]["bbox"]
                box2 = sorted_det[j]["bbox"]
                
                h1 = box1[3] - box1[1]
                h2 = box2[3] - box2[1]
                min_h = min(h1, h2)
                
                y_overlap = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
                v_overlap_ratio = y_overlap / float(min_h) if min_h > 0 else 0
                gap = box2[0] - box1[2]
                
                if v_overlap_ratio > min_v_overlap and gap < max_gap_factor * min_h:
                    merged_box = [
                        min(box1[0], box2[0]),
                        min(box1[1], box2[1]),
                        max(box1[2], box2[2]),
                        max(box1[3], box2[3])
                    ]
                    if box1[0] <= box2[0]:
                        combined_text = sorted_det[i]["text"] + " " + sorted_det[j]["text"]
                    else:
                        combined_text = sorted_det[j]["text"] + " " + sorted_det[i]["text"]
                        
                    sorted_det[i] = {
                        "text": combined_text,
                        "bbox": merged_box,
                        "confidence": max(sorted_det[i]["confidence"], sorted_det[j]["confidence"])
                    }
                    sorted_det.pop(j)
                    merged_any = True
                else:
                    j += 1
            i += 1
            
    return sorted_det


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
    merged = merge_horizontal_lines(merged)
    return merged


def _zone_for_bbox(bbox, w, h):
    y_mid = (bbox[1] + bbox[3]) / 2.0
    x_mid = (bbox[0] + bbox[2]) / 2.0
    in_border = (
        y_mid < h * 0.25 or y_mid > h * 0.75 or
        x_mid < w * 0.15 or x_mid > w * 0.85
    )
    return "border" if in_border else "anatomy"


def expand_phi_blocks(merged, phi_regions, image_shape):
    """
    Groups `merged` detections into vertically-stacked text blocks (adjacent
    lines, overlapping horizontally -- e.g. a burned-in "Name:" / "ID:" /
    "Date:" header) and, if any line in a block was classified as PHI,
    redacts the rest of that block too.

    OCR misreads individual words often enough (garbling "Name:" while
    reading "ID:"/"Date:" on the same block correctly) that per-line
    classification alone can leave a sibling line of an otherwise-redacted
    demographic block exposed.
    """
    if not merged:
        return phi_regions

    h, w = image_shape[:2]
    boxes = [d["bbox"] for d in merged]
    order = sorted(range(len(boxes)), key=lambda i: boxes[i][1])

    clusters = []
    current = [order[0]]
    for idx in order[1:]:
        prev_box, box = boxes[current[-1]], boxes[idx]
        prev_h = prev_box[3] - prev_box[1]
        gap = box[1] - prev_box[3]
        x_overlap = max(0, min(prev_box[2], box[2]) - max(prev_box[0], box[0]))
        min_w = min(prev_box[2] - prev_box[0], box[2] - box[0])
        overlap_ratio = x_overlap / min_w if min_w > 0 else 0
        if gap <= 1.8 * max(prev_h, 1) and overlap_ratio >= 0.3:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    phi_bboxes = {tuple(r["bbox"]) for r in phi_regions}
    added = 0
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        cluster_bboxes = [tuple(boxes[i]) for i in cluster]
        has_phi = any(b in phi_bboxes for b in cluster_bboxes)
        all_phi = all(b in phi_bboxes for b in cluster_bboxes)
        if has_phi and not all_phi:
            for i in cluster:
                b = tuple(boxes[i])
                if b not in phi_bboxes:
                    zone = _zone_for_bbox(list(b), w, h)
                    log.info(f"    REDACT-{zone.upper()} (block_expand): '{merged[i]['text']}' @ {list(b)}")
                    phi_regions.append({"text": merged[i]["text"], "bbox": list(b), "zone": zone})
                    phi_bboxes.add(b)
                    added += 1
    if added:
        log.info(f"    Block-expand: redacting {added} additional sibling line(s) in flagged text blocks")
    return phi_regions


def _is_clinical(text):
    val = text.strip().upper()
    if re.match(r'^(?:L|R|LT|RT|LA|RA|LP|RP|PA|AP|LL|RL|A|P)\s*\d*$', val):
        return True
        
    clean = re.sub(r'[^A-Z]', '', val)
    if clean in CLINICAL_ALLOWLIST:
        return True

    tokens = re.findall(r'[A-Z0-9]+', val)
    if tokens:
        expanded_safe = CLINICAL_ALLOWLIST.union({
            "ANTEROPOSTERIOR", "POSTEROANTERIOR", "ANIIEROPOSTERIOR", "ANIIEROPAOSIIERIORR",
            "PROJECTION", "ANTERIOR", "POSTERIOR", "ERECT", "SUPINE", "CHEST", "PORTABLE"
        })
        if all(t in expanded_safe for t in tokens):
            return True
        
    return False


def classify_phi(merged, image_shape, analyzer=None, gliner_model=None, medical_ner=None, deid_model=None):
    """
    Classifies each merged detection as PHI (redact) or safe (keep).
    Uses a 3-Step Hybrid Priority Stack followed by medical protection and spatial fallback:

    Top Priority — 3-Step Hybrid PII/PHI Classification:
      Step 1: Stanford De-ID Model (StanfordAIMI/stanford-deidentifier-base)
              Radiology-native transformer — detects PATIENT, NAME, DATE, ID, LOCATION, AGE, PHONE, etc.
      Step 2: Regex Pattern Matching
              Deterministic patterns for Aadhaar, ABHA, phone, UHID, MRN, date formats, age/sex, demographic labels.
      Step 3: Hybrid Combined Decision
              If BOTH model AND regex agree → high confidence.
              If EITHER flags PHI → redact (safety-first).

    Phase 2: Medical Entity Preservation
      - d4data/biomedical-ner-all (supervised)
      - GLiNER-BioMed Large (zero-shot, radiology-specific labels)
      - Presidio cross-check to prevent medical NER from shielding real PHI

    Phase 3: Exposure / Number Guard
      - Pure numeric values (e.g. 120, 80.5) kept as scan parameters.

    Phase 4: Spatial / Default Fallback
      - Unclassified text in border zones → redact.
      - Unclassified text in anatomy zone → keep.
    """
    h, w = image_shape[:2]
    phi_regions = []

    def _is_text_pure_clinical(text_str):
        val = text_str.strip().upper()
        if not val:
            return True

        tokens = re.findall(r'[A-Z0-9]+', val)
        if not tokens:
            return True

        safe_terms = {
            "L", "R", "LT", "RT", "LEFT", "RIGHT", "PA", "AP", "LAT", "LL", "RL", "LATERAL",
            "A", "P", "M", "F", "O", "CXR", "CHEST", "PORTABLE", "MOBILE", "SUPINE", "ERECT",
            "UPRIGHT", "SEMI-UPRIGHT", "SEMI", "UPPER", "LOWER", "MIDDLE", "LOBE", "LUL", "RUL",
            "LLL", "RLL", "RML", "CLAVICLE", "RIB", "HEART", "LUNG", "LUNGS", "DIAPHRAGM",
            "TRACHEA", "BRONCHUS", "APEX", "BASE", "CARDIAC", "AORTA", "PLEURAL", "ANGLE",
            "COSTOPHRENIC", "SULCUS", "MEDIASTINUM", "HILUM", "HILAR", "PARENCHYMA", "PARENCHYMAL",
            "QUADRANT", "KV", "KVP", "MA", "MAS", "MS", "SEC", "MM", "CM", "EXP", "EXPOSURE",
            "TECH", "TECHNOLOGIST", "COLLIMATION", "FILTER", "FSD", "SID", "SOD", "MASE",
            "GRID", "NO-GRID", "NOGRID", "SANS", "INSPIRATION", "EXPIRATION", "INSP", "EXP",
            "INT", "EXT", "MED", "DECUB", "DECUBITUS", "VIEW", "OF", "AND", "TO", "WITH",
            "FOR", "ON", "BY", "IN", "THE", "PORT", "AP-PORTABLE", "PA-ERECT", "AP/PA",
            "NORMAL", "PNEUMOTHORAX", "EFFUSION", "INFILTRATE", "CONSOLIDATION", "EDEMA",
            "CARDIOMEGALY", "PNEUMONIA", "OPACITY", "OPACITIES", "CARDIAC", "AORTIC",
            "SIEMENS", "PHILIPS", "GE", "HEALTHCARE", "MEDICAL", "SYSTEMS", "MICRODICOM",
            "ANTEROPOSTERIOR", "POSTEROANTERIOR", "ANIIEROPOSTERIOR", "ANIIEROPAOSIIERIORR",
            "PROJECTION", "ANTERIOR", "POSTERIOR"
        }

        for token in tokens:
            if token in safe_terms:
                continue
            if re.match(r'^\d{1,4}$', token):
                continue
            if re.match(r'^(?:L|R|C|T|S)\d{0,2}$', token):
                continue
            return False
        return True

    for det in merged:
        text = det["text"]
        bbox = det["bbox"]

        box_w = bbox[2] - bbox[0]
        box_h = bbox[3] - bbox[1]
        box_area = box_w * box_h
        img_area = w * h

        if box_h > h * 0.40 or box_w > w * 0.95 or box_area > (img_area * 0.30):
            continue

        clean_text = text.strip()
        if not clean_text:
            continue

        is_phi = False
        reason = []
        classified = False
        val_upper = clean_text.upper()

        # ── Step 1: Stanford De-ID Model (Radiology-Native) ──────────────────
        model_flags_phi = False
        model_phi_labels = []
        if deid_model:
            try:
                deid_entities = deid_model(clean_text)
                for ent in deid_entities:
                    score = float(ent.get("score", 0.0))
                    raw_group = str(ent.get("entity_group") or ent.get("entity") or "").strip()
                    clean_group = re.sub(r'^[BILOU]-', '', raw_group, flags=re.IGNORECASE)
                    if score > 0.30 and clean_group and clean_group.upper() not in {"O", "OUTSIDE"}:
                        if clean_group.upper() in {"UNIQUE_ID", "ID", "MEDICAL_RECORD_NUMBER", "SSN", "PHONE"} and not re.search(r'\d', clean_text):
                            continue
                        model_flags_phi = True
                        model_phi_labels.append(clean_group)
            except Exception as e:
                log.debug(f"Stanford De-ID model classification error: {e}")

        # ── Step 2: Regex Pattern Matching ──────────────────────────────────
        regex_flags_phi = False
        regex_reasons = []

        generic_demographic_labels = [
            "NAME", "PATIENT", "DATE", "DOB", "D0B", "0B:", "BIRTH", "MRN", "UHID", "PID", "AGE", "SEX", "GENDER",
            "MALE", "FEMALE", "DR.", "DOCTOR", "PHYSICIAN", "HOSPITAL", "HOSP", "CLINIC", "INSTITUT",
            "CONFIDENTIAL", "CONFIDCNTTIAC", "RESTRICTED", "PROPRIETARY", "SECRET"
        ]
        has_id_label = bool(re.search(r'\bID\b', val_upper))
        has_demographic_label = any(label in val_upper for label in generic_demographic_labels) or has_id_label
        if has_demographic_label:
            regex_flags_phi = True
            regex_reasons.append("regex:demographic_label")

        for pat_name, pat_regex in PII_PATTERNS.items():
            if re.search(pat_regex, clean_text, re.IGNORECASE):
                regex_flags_phi = True
                regex_reasons.append(f"regex:pii_pattern_{pat_name}")
                break

        # ── Step 3: Combined Decision ────────────────────────────────────────
        if model_flags_phi and regex_flags_phi:
            is_phi = True
            reason.append(f"hybrid:model+regex_agree(model={','.join(model_phi_labels)}, {', '.join(regex_reasons)})")
            classified = True
        elif regex_flags_phi:
            is_phi = True
            reason.extend(regex_reasons)
            classified = True
        elif model_flags_phi:
            if _is_clinical(clean_text) or _is_text_pure_clinical(clean_text):
                reason.append(f"hybrid:model_flagged({','.join(model_phi_labels)})_but_clinical_override")
            else:
                is_phi = True
                reason.append(f"hybrid:model_only({','.join(model_phi_labels)})")
                classified = True

        # ── Presidio Analyzer Check (if active) ──────────────────────────────
        if not classified and analyzer:
            try:
                results = analyzer.analyze(text=clean_text, language="en")
                if any(r.entity_type in {"PERSON", "DATE_TIME", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"} and r.score > 0.40 for r in results):
                    is_phi = True
                    classified = True
                    reason.append("presidio_nlp")
            except Exception:
                pass

        # ── Phase 2: Medical Entity Preservation (d4data + GLiNER-BioMed) ────
        if not classified and medical_ner:
            try:
                entities = medical_ner(clean_text)
                for ent in entities:
                    ent_group = ent.get("entity_group", "")
                    if ent_group in {
                        "Anatomical_structure", "Diagnostic_procedure", "Disease_disorder",
                        "Sign_symptom", "Lab_value", "Medication",
                        "Biological_structure", "Therapeutic_procedure"
                    }:
                        is_phi = False
                        classified = True
                        reason.append(f"medical_ner:{ent_group}")
                        break
            except Exception as e:
                log.debug(f"Medical NER prediction failed: {e}")

        if not classified and gliner_model:
            try:
                labels = [
                    "thoracic anatomy", "chest anatomy", "body part", "organ",
                    "cardiac structure", "pulmonary structure", "skeletal structure",
                    "lung pathology", "cardiac finding", "disease", "medical condition",
                    "clinical finding", "radiological finding", "medical procedure",
                    "diagnostic test", "imaging technique", "patient positioning",
                    "scan parameter", "exposure setting", "medical device", "medical equipment",
                    "imaging modality", "medication", "drug name", "vital sign", "lab value",
                    "measurement", "anatomical direction", "laterality marker",
                ]
                entities = gliner_model.predict_entities(clean_text, labels, threshold=0.35)
                for ent in entities:
                    if ent["label"] in labels:
                        is_phi = False
                        classified = True
                        reason.append(f"gliner_biomed:{ent['label']}")
                        break
            except Exception as e:
                log.debug(f"GLiNER-BioMed clinical check failed: {e}")

        # ── Phase 3: Exposure / Number Guard ─────────────────────────────────
        if not classified:
            if re.match(r'^\d{1,4}(?:\.\d+)?(?:\s*-\s*\d{1,4}(?:\.\d+)?)?$', clean_text):
                is_phi = False
                classified = True
                reason.append("clinical:exposure_number")

        # ── Phase 4: Spatial / Default Fallback ──────────────────────────────
        if not classified:
            in_border = _zone_for_bbox(bbox, w, h) == "border"
            if in_border and re.search(r'[A-Za-z]', clean_text):
                is_phi = True
                reason.append("fallback:suspect_border")
            else:
                is_phi = False
                reason.append("fallback:safe_anatomy")

        if is_phi:
            zone = _zone_for_bbox(bbox, w, h)
            log.info(f"    REDACT-{zone.upper()} ({', '.join(reason)}): '{text}' @ {bbox}")
            phi_regions.append({"text": text, "bbox": bbox, "zone": zone})
        else:
            log.info(f"    KEEP ({', '.join(reason) if reason else 'unclassified'}): '{text}' @ {bbox}")

    return phi_regions
