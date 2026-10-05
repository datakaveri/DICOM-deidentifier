"""
classify.py — Stage 4: merge overlapping OCR detections and classify each
one as PHI (redact) or safe (keep).
"""

import re
from difflib import SequenceMatcher

import numpy as np

from config import log, CLINICAL_ALLOWLIST, PII_PATTERNS, NON_PII_EXCLUSIONS


# ── OCR Error Normalization & Fuzzy Clinical Matching ─────────────────────────────────
# PaddleOCR frequently misreads burned-in text characters on low-resolution
# medical images (e.g. 'NORMAL' -> 'N0RMAL', 'ERECT' -> 'ERECI'). These
# functions catch OCR-corrupted clinical terms before they reach the AI models.

_OCR_SUBSTITUTIONS = {
    '0': 'O', '1': 'I', '5': 'S', '8': 'B', '6': 'G',
    '|': 'I', '!': 'I', '$': 'S', '@': 'A',
}


def _normalize_ocr(text):
    """Normalize common OCR character misreads for clinical term matching."""
    t = text.strip().upper()
    return ''.join(_OCR_SUBSTITUTIONS.get(c, c) for c in t)


def _is_fuzzy_clinical(text, threshold=0.78):
    """
    Fuzzy-match OCR-corrupted text against the expanded CLINICAL_ALLOWLIST.
    Catches 'N0RMAL', 'NCRMAL', 'NORMA', 'IORMA', 'DIAPHGRAM', 'CONSOLRA', 'DATION', etc.

    Strategy:
      1. OCR-normalize the text (fix 0->O, 1->I, etc.) and try exact match.
      2. Check common clinical OCR misspellings & truncations directly.
      3. Tokenize and check if clinical keywords match.
      4. For terms (≥4 chars), use SequenceMatcher fuzzy similarity.
    """
    val_raw = text.strip().upper()
    # 0. Direct exact match (before OCR normalization — preserves L5, T12, C7 etc.)
    if val_raw in CLINICAL_ALLOWLIST:
        return True

    # Common radiological OCR fragments and spelling variations
    known_clinical_fragments = {
        "NORMA", "IORMA", "NORMALI", "DIAPHGRAM", "CONSOLRA", "DATION",
        "VESSEL", "VESSELS", "PRUNING", "FLATTENED", "FLATTENING",
        "HAZINESS", "TENTING", "HEMIDIAPHRAGM"
    }
    if val_raw in known_clinical_fragments:
        return True

    val = _normalize_ocr(text)
    # 1. Exact match on OCR-normalized text
    if val in CLINICAL_ALLOWLIST or val in known_clinical_fragments:
        return True

    # 2. Token-level check (catches multi-word clinical phrases like "FLATTENED DIAPHGRAM")
    tokens = re.findall(r'[A-Z0-9]+', val)
    if tokens:
        clinical_set = CLINICAL_ALLOWLIST.union(known_clinical_fragments)
        if all(
            t in clinical_set
            or re.match(r'^(?:L|R|C|T|S|D)\d{0,2}$', t)
            or re.match(r'^\d{1,4}$', t)
            for t in tokens
        ):
            return True
        # If all alphabetic tokens are clinical
        alpha_tokens = [t for t in tokens if re.search(r'[A-Z]', t)]
        if alpha_tokens and all(t in clinical_set for t in alpha_tokens):
            return True

    # 3. Fuzzy match for terms (≥4 chars) to catch OCR corruption
    if len(val) >= 4:
        for term in CLINICAL_ALLOWLIST:
            if len(term) < 4:
                continue
            if abs(len(val) - len(term)) > 3:
                continue
            if SequenceMatcher(None, val, term).ratio() >= threshold:
                return True
    return False


def _iou(a, b):
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = (a[2]-a[0]) * (a[3]-a[1])
    areaB = (b[2]-b[0]) * (b[3]-b[1])
    union = float(areaA + areaB - inter)
    return inter / union if union > 0 else 0.0


def _is_pure_clinical_fast(text):
    return _is_clinical(text)


def merge_horizontal_lines(detections, max_gap_factor=2.5, min_v_overlap=0.45):
    """
    Merges detections that are on the same horizontal line and close to each other.
    Allows catching labeled PII like 'PATIENT:' + 'MEERA IYER' as a single entity,
    while keeping pure clinical terms (e.g. 'CXR', 'L26') separate so they remain preserved.
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
                
                # Do not merge if one box is pure clinical text (e.g. 'CXR', 'L26') and the other is not
                is_clin1 = _is_pure_clinical_fast(sorted_det[i]["text"])
                is_clin2 = _is_pure_clinical_fast(sorted_det[j]["text"])
                if is_clin1 != is_clin2:
                    j += 1
                    continue

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



def _containment(a, b):
    """Returns fraction of box `a` that is contained inside box `b`."""
    xA = max(a[0], b[0]); yA = max(a[1], b[1])
    xB = min(a[2], b[2]); yB = min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    areaA = max(1, (a[2]-a[0]) * (a[3]-a[1]))
    return inter / float(areaA)


def _should_merge(box_a, box_b, iou_thresh=0.3):
    """Merge if IoU > threshold OR if one box contains >70% of the other."""
    if _iou(box_a, box_b) > iou_thresh:
        return True
    if _containment(box_a, box_b) > 0.70 or _containment(box_b, box_a) > 0.70:
        return True
    return False


def merge_detections(raw, iou_thresh=0.3):
    """NMS-style merge of overlapping boxes from all OCR engine+variant passes.
    Uses both IoU and containment checks to catch near-duplicate bboxes."""
    if not raw:
        return []
    sorted_det = sorted(raw, key=lambda x: x["confidence"], reverse=True)
    merged = []
    while sorted_det:
        cur = sorted_det.pop(0)
        group = [cur]
        remaining = []
        for d in sorted_det:
            if _should_merge(cur["bbox"], d["bbox"], iou_thresh):
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

    # Post-merge dedup: remove any remaining near-duplicate boxes
    deduped = []
    for det in merged:
        is_dup = False
        for existing in deduped:
            if _containment(det["bbox"], existing["bbox"]) > 0.80:
                # det is mostly inside existing — skip it, but widen existing
                existing["bbox"][0] = min(existing["bbox"][0], det["bbox"][0])
                existing["bbox"][1] = min(existing["bbox"][1], det["bbox"][1])
                existing["bbox"][2] = max(existing["bbox"][2], det["bbox"][2])
                existing["bbox"][3] = max(existing["bbox"][3], det["bbox"][3])
                if len(det["text"]) > len(existing["text"]):
                    existing["text"] = det["text"]
                is_dup = True
                break
        if not is_dup:
            deduped.append(det)

    deduped = merge_horizontal_lines(deduped)
    return deduped




def expand_phi_blocks(merged, phi_regions, image_shape):
    """
    Groups `merged` detections into vertically-stacked text blocks (adjacent
    lines, overlapping horizontally -- e.g. a burned-in "Name:" / "ID:" /
    "Date:" header) and, if any line in a block was classified as PHI,
    redacts the rest of that block too (unless the sibling line is a confirmed
    clinical term).
    """
    if not merged:
        return phi_regions

    h, w = image_shape[:2]
    boxes = [d["bbox"] for d in merged]
    n = len(boxes)

    # Connected-component graph clustering for multi-line column blocks
    # Prevents distant detections (e.g. on opposite side of image) from splitting clusters
    adj = {i: [] for i in range(n)}
    for i in range(n):
        b1 = boxes[i]
        w1 = max(1, b1[2] - b1[0])
        h1 = max(1, b1[3] - b1[1])
        for j in range(i + 1, n):
            b2 = boxes[j]
            w2 = max(1, b2[2] - b2[0])
            h2 = max(1, b2[3] - b2[1])
            x_overlap = max(0, min(b1[2], b2[2]) - max(b1[0], b2[0]))
            min_w = min(w1, w2)
            if min_w > 0 and (x_overlap / min_w) >= 0.25:
                v_gap = max(0, max(b1[1], b2[1]) - min(b1[3], b2[3]))
                max_h = max(h1, h2)
                if v_gap <= max(45, int(1.8 * max_h)):
                    adj[i].append(j)
                    adj[j].append(i)

    visited = set()
    clusters = []
    for i in range(n):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for neighbor in adj[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(comp)

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
                    sibling_text = merged[i]["text"]
                    if _is_clinical(sibling_text):
                        log.info(f"    KEEP-SIBLING (block_expand_override): '{sibling_text}' @ {list(b)}")
                        continue
                    log.info(f"    REDACT (block_expand): '{sibling_text}' @ {list(b)}")
                    phi_regions.append({"text": sibling_text, "bbox": list(b), "reason": "block_expand:sibling_line"})
                    phi_bboxes.add(b)
                    added += 1

        # If this cluster contains PHI, align its left edge so truncated words at the margin are fully covered
        if has_phi or all_phi:
            min_cluster_x = min(boxes[i][0] for i in cluster)
            max_cluster_x2 = max(boxes[i][2] for i in cluster)
            target_x = max(0, min_cluster_x - 35) if min_cluster_x < 150 else min_cluster_x

            # Bridge vertical gaps in multi-line PHI blocks where OCR missed intermediate lines
            sorted_cluster = sorted(cluster, key=lambda idx: boxes[idx][1])
            for c_i in range(len(sorted_cluster) - 1):
                top_box = boxes[sorted_cluster[c_i]]
                bot_box = boxes[sorted_cluster[c_i + 1]]
                gap_y1 = top_box[3]
                gap_y2 = bot_box[1]
                gap_h = gap_y2 - gap_y1
                # If there is a small gap (1 to 35px) between consecutive PHI lines
                if 0 < gap_h <= 35:
                    bridge_x1 = min(top_box[0], bot_box[0])
                    bridge_x2 = max(top_box[2], bot_box[2])
                    bridge_bbox = [bridge_x1, gap_y1, bridge_x2, gap_y2]
                    # Ensure no confirmed clinical text is inside this gap
                    if not any(_is_clinical(m["text"]) and _containment(bridge_bbox, m["bbox"]) > 0.5 for m in merged):
                        phi_regions.append({"text": "[gap_bridge]", "bbox": bridge_bbox, "reason": "block_expand:gap_bridge"})
                        phi_bboxes.add(tuple(bridge_bbox))
                        added += 1

            for r in phi_regions:
                b_curr = tuple(r["bbox"])
                if b_curr in cluster_bboxes or any(_containment(r["bbox"], boxes[i]) > 0.6 for i in cluster):
                    r["bbox"][0] = min(r["bbox"][0], target_x)
                    r["bbox"][2] = min(w, r["bbox"][2] + 8)
                    r["bbox"][1] = max(0, r["bbox"][1] - 3)
                    r["bbox"][3] = min(h, r["bbox"][3] + 3)
                    # Extend truncated lines in demographic blocks at margin
                    if (r["bbox"][2] - r["bbox"][0]) < 0.60 * (max_cluster_x2 - min_cluster_x) and max_cluster_x2 > 300:
                        r["bbox"][2] = min(w, max(r["bbox"][2], int(max_cluster_x2 * 0.90)))
    if added:
        log.info(f"    Block-expand: redacting {added} additional sibling line(s) in flagged text blocks")
    return phi_regions



def _is_clinical(text):
    val = text.strip().upper()
    if re.match(r'^(?:L|R|LT|RT|LA|RA|LP|RP|PA|AP|LL|RL|A|P)\s*\d{0,3}$', val):
        return True
        
    clean = re.sub(r'[^A-Z0-9]', '', val)
    if clean in CLINICAL_ALLOWLIST or clean in {"L26", "R26", "L1", "R1", "L2", "R2", "L3", "R3", "CXR"}:
        return True

    # Demographics check — if it carries demographic markers, it is NOT pure clinical
    demog_markers = ["PATIENT", "NAME", "DOB", "D0B", "DR.", "DOCTOR", "PHYSICIAN", "HOSPITAL", "CLINIC", "MRN", "PID", "UHID", "AGE", "SEX"]
    if any(m in val for m in demog_markers):
        return False

    tokens = re.findall(r'[A-Z0-9]+', val)
    if tokens:
        expanded_safe = CLINICAL_ALLOWLIST.union({
            "ANTEROPOSTERIOR", "POSTEROANTERIOR", "ANIIEROPOSTERIOR", "ANIIEROPAOSIIERIORR",
            "PROJECTION", "ANTERIOR", "POSTERIOR", "ERECT", "SUPINE", "CHEST", "PORTABLE",
            "L26", "R26", "L12", "R12", "CXR", "NORMAL", "CLEAR", "COSTOPHRENIC", "ANGLES",
            "SHARP", "RECESSES", "SULCI", "CARDIOMEGALY", "EFFUSION", "VERTEBRAE", "TRACHEA",
            "MIDLINE", "CENTRAL", "PULMONARY", "INFILTRATE", "AORTIC", "KNOB", "ELONGATION",
            "BRONCHOVASCULAR", "SILHOUETTE", "EXPANDED", "PNEUMOTHORAX", "MEDIASTINAL",
            "MEDIASTINUM", "RADIOGRAPH", "BONY", "CAGE", "INTACT", "VASCULATURE", "RETICULAR",
            "OPACITIES", "OPACITY", "CARDIOTHORACIC", "RATIO", "PLEURA", "BLUNTING", "CP",
            "ANGLE", "LESION", "PARENCHYMA", "INFILTRATION", "DIAPHRAGMATIC", "OUTLINES",
            "TB", "TUBERCULOSIS", "THORACIC", "SKELETON", "HYPERINFLATED", "RETICULONODULAR",
            "ACUTE", "CONSOLIDATION", "PATTERN", "BILATERAL", "EXAMINATION", "STUDY", "VIEW",
            "KVP", "MAS", "SID", "CM"
        })
        clinical_matches = sum(1 for t in tokens if t in expanded_safe)
        # If all tokens or a majority of clinical keywords match
        if clinical_matches >= 2 and (clinical_matches / len(tokens)) >= 0.5:
            return True
        
    return False


def classify_phi(merged, image_shape, analyzer=None, gliner_model=None, medical_ner=None, deid_model=None, fallback_medical_ner=None, indian_ner=None, metadata_values=None, return_details=False):
    """
    Classifies each merged detection as PHI (redact) or safe (keep) using
    GROUND-TRUTH METADATA CROSS-VERIFICATION.

    Core Strategy:
      1. Candidate text is cross-verified against the DICOM metadata ground truth
         (PatientName, PatientID, DOB, Doctor, Hospital, Address).
      2. Universal PII patterns (Aadhaar, PAN, Phone, Email, ABHA) are checked.
      3. Explicit demographic labels with values (PATIENT:, MRN:, DOB:, DR:) are checked.
      4. General NER / Presidio false positives (e.g. tagging anatomy/findings as PERSON or LOCATION)
         are filtered out unless corroborated by metadata or explicit demographic label.
      5. Clinical findings, anatomy, and scan text that do not match metadata PII are NEVER touched.
         (Clinical allowlist is commented out in favor of dynamic metadata cross-verification).
    """
    h, w = image_shape[:2]
    phi_regions = []
    kept_regions = []

    # ── Prepare normalized metadata ground truth cache ────────────────────────
    normalized_metadata = set()
    metadata_tokens = set()
    for sv in (metadata_values or []):
        sv_clean = sv.strip().upper()
        if len(sv_clean) >= 3:
            normalized_metadata.add(sv_clean)
            for t in re.findall(r'[A-Z0-9]+', sv_clean):
                if len(t) >= 3 and not re.match(r'^\d+$', t):
                    metadata_tokens.add(t)
                elif len(t) >= 4:
                    metadata_tokens.add(t)

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

        val_upper = clean_text.upper()
        reason = []

        # ── 1. METADATA CROSS-VERIFICATION SIGNAL ─────────────────────────────
        val_tokens = [t for t in re.findall(r'[A-Z0-9]+', val_upper) if len(t) >= 3]
        matches_metadata = False
        matched_tag_val = None

        for sv in normalized_metadata:
            if val_upper == sv or (len(sv) >= 4 and sv in val_upper) or (len(val_upper) >= 4 and val_upper in sv):
                matches_metadata = True
                matched_tag_val = sv
                break
        if not matches_metadata and metadata_tokens:
            for t in val_tokens:
                if t in metadata_tokens:
                    matches_metadata = True
                    matched_tag_val = t
                    break

        # ── 2. UNIVERSAL PII REGEX PATTERNS ───────────────────────────────────
        matches_universal_pii = False
        regex_reasons = []
        for pat_name, pat_regex in PII_PATTERNS.items():
            if re.search(pat_regex, clean_text, re.IGNORECASE):
                matches_universal_pii = True
                regex_reasons.append(f"regex:{pat_name}")
                break

        # ── 3. EXPLICIT DEMOGRAPHIC LABELS ────────────────────────────────────
        generic_demographic_labels = [
            "PATIENT", "NAME", "DOB", "D0B", "0B:", "BIRTH", "MRN", "UHID", "PID",
            "DR.", "DOCTOR", "PHYSICIAN", "HOSPITAL", "CLINIC", "INSTITUT",
            "CONFIDENTIAL", "MEDICAL RECORD"
        ]
        has_id_label = bool(re.search(r'\bID\b', val_upper))
        has_demographic_label = any(label in val_upper for label in generic_demographic_labels) or has_id_label
        has_demographic_value = bool(has_demographic_label and len(clean_text) >= 5)

        # ── 4. GENERAL NER & PRESIDIO ANALYSIS ────────────────────────────────
        ner_phi_labels = []
        ner_engine = indian_ner if indian_ner is not None else deid_model
        if ner_engine:
            try:
                if hasattr(ner_engine, "predict_entities"):
                    entities = ner_engine.predict_entities(clean_text)
                    for ent in entities:
                        score = float(ent.get("score", 0.0))
                        cat = str(ent.get("cat", "")).upper()
                        if score >= 0.45 and cat in {"PERSON", "LOCATION", "ORGANIZATION", "PER", "LOC", "ORG"}:
                            ner_phi_labels.append(f"ner_{cat.lower()}")
                else:
                    entities = ner_engine(clean_text)
                    for ent in entities:
                        score = float(ent.get("score", 0.0))
                        raw_group = str(ent.get("entity_group") or ent.get("entity") or "").strip()
                        clean_group = re.sub(r'^[BILOU]-', '', raw_group, flags=re.IGNORECASE).upper()
                        if score > 0.50 and clean_group in {"PER", "LOC", "ORG", "PERSON", "LOCATION", "ORGANIZATION"}:
                            ner_phi_labels.append(clean_group)
            except Exception as e:
                log.debug(f"NER error: {e}")

        presidio_flagged = False
        if analyzer:
            try:
                results = analyzer.analyze(text=clean_text, language="en")
                if any(r.entity_type in {"PERSON", "DATE_TIME", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"} and r.score > 0.50 for r in results):
                    presidio_flagged = True
            except Exception:
                pass

        # ── 5. CROSS-VERIFICATION DECISION MATRIX ──────────────────────────────
        # Rule 1: Text directly matches DICOM metadata ground truth (PatientName, ID, Doctor, Hospital, DOB)
        if matches_metadata:
            is_phi = True
            reason.append(f"cross_verify:metadata_match({matched_tag_val})")

        # Rule 2: Text matches universal structured PII pattern (Aadhaar, PAN, phone, email)
        elif matches_universal_pii:
            is_phi = True
            reason.append(f"cross_verify:universal_pii({','.join(regex_reasons)})")

        # Rule 3: Text has explicit demographic label with value (e.g. "PATIENT: ...", "DOB: ...")
        elif has_demographic_value and (ner_phi_labels or presidio_flagged or any(c.isdigit() for c in clean_text) or len(val_tokens) >= 2):
            is_phi = True
            reason.append("cross_verify:demographic_labeled_pii")

        # Rule 4: NER / Presidio detected person or location
        # MUST BE CORROBORATED by metadata or demographic label to avoid destroying medical text
        elif ner_phi_labels or presidio_flagged:
            if has_demographic_label or matches_metadata:
                is_phi = True
                reason.append("cross_verify:ner_corroborated_pii")
            else:
                # Discard generic NER false-positives on medical findings / anatomy
                is_phi = False
                reason.append("cross_verify:unverified_ner_discarded_clinical_kept")

        # Rule 5: Default Fallback — Everything else is KEPT!
        else:
            is_phi = False
            reason.append("cross_verify:clinical_or_non_pii_kept")

        reason_str = ", ".join(reason) if reason else "unspecified"
        if is_phi:
            log.info(f"    REDACT ({reason_str}): '{text}' @ {bbox}")
            phi_regions.append({"text": text, "bbox": bbox, "reason": reason_str})
        else:
            log.info(f"    KEEP ({reason_str}): '{text}' @ {bbox}")
            kept_regions.append({"text": text, "bbox": bbox, "reason": reason_str})

    if return_details:
        return phi_regions, kept_regions
    return phi_regions
