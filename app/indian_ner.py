"""
indian_ner.py — Hybrid Indic NER ensemble and Truecasing preprocessor
for Indian demographic entity recognition (Patient Names, Doctor Names, Hospitals).

Ensemble Architecture:
  1. HiNER (IIT Bombay / MuRIL): cfilt/HiNER-original-muril-base-cased
  2. IndicNER (AI4Bharat): ai4bharat/IndicNER
  3. XLM-RoBERTa (Multilingual): Babelscape/wikineural-multilingual-ner
  4. BERT (General fallback): dslim/bert-base-NER

Includes:
  - truecase_line: Length-preserving casing correction for ALL-CAPS X-ray burned-in text.
  - Dual-pass inference: Evaluates verbatim and truecased strings to maximize sensitivity.
  - Subword/word boundary snapping & clinical/non-PII filtering.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional, Tuple, Any

from config import log, CLINICAL_ALLOWLIST, NON_PII_EXCLUSIONS

try:
    import torch
    TRANSFORMERS_TORCH_AVAILABLE = True
except ImportError:
    TRANSFORMERS_TORCH_AVAILABLE = False

try:
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    from transformers import pipeline as tf_pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


INDIAN_NER_MODEL_SPECS = [
    ("HiNER", "cfilt/HiNER-original-muril-base-cased", False, 0.45),
    ("IndicNER", "ai4bharat/IndicNER", False, 0.50),
    ("XLM_RoBERTa", "Babelscape/wikineural-multilingual-ner", True, 0.45),
]


def truecase_line(text: str) -> str:
    """
    Performs a 1-to-1 length-preserving title-casing transformation on words.
    Enables cased Transformer models (MuRIL, IndicNER, XLM-RoBERTa) to recognize
    entities in ALL-CAPS medical X-ray text without altering character offsets.
    """
    if not text:
        return ""
    return re.sub(r"\b[A-Za-z]+\b", lambda m: m.group(0).capitalize(), text)


def normalize_entity_label(label: Optional[str]) -> Optional[str]:
    """Normalizes entity category names to standard PERSON, LOCATION, ORGANIZATION or None."""
    if not label:
        return None
    lbl = str(label).upper().strip()
    lbl = re.sub(r"^[BI]-", "", lbl)
    if lbl in ("PER", "PERSON", "NAME"):
        return "PERSON"
    if lbl in ("LOC", "LOCATION", "GPE"):
        return "LOCATION"
    if lbl in ("ORG", "ORGANIZATION"):
        return "ORGANIZATION"
    if lbl in ("ALL", "ANY", "*", "NONE"):
        return None
    return lbl


def _snap_to_word_boundary(text: str, start: int, end: int) -> Tuple[int, int]:
    """Snaps token character spans outwards to complete word boundaries."""
    STRICT_DELIMITERS = set(" /\\:;,()[]{}<>\"'\t\n\r")
    while start > 0:
        prev = text[start - 1]
        if prev in STRICT_DELIMITERS:
            break
        if prev.isalnum():
            start -= 1
        elif (
            prev == "."
            and start > 1
            and text[start - 2].isalpha()
            and (start == 2 or text[start - 3] in STRICT_DELIMITERS or text[start - 3].isspace())
        ):
            start -= 1
        else:
            break

    while end < len(text):
        nxt = text[end]
        if nxt in STRICT_DELIMITERS:
            break
        if nxt.isalnum():
            end += 1
        elif nxt == "." and end + 1 < len(text) and text[end + 1].isalnum():
            end += 1
        else:
            break

    return start, end


def _clean_entity_text(text: str, start: int, end: int) -> Tuple[str, int, int]:
    """Trims leading/trailing punctuation and prefixes/suffixes from an extracted span."""
    val = text[start:end]
    match_prefix = re.match(r"^([\s:\-.,'\"/()]+|[A-Za-z]/[a-zA-Z]?\.?\s*|[a-zA-Z]\.\s*)", val)
    while match_prefix and match_prefix.end() > 0:
        cut = match_prefix.end()
        if cut >= len(val):
            break
        start += cut
        val = text[start:end]
        match_prefix = re.match(r"^([\s:\-.,'\"/()]+|[A-Za-z]/[a-zA-Z]?\.?\s*|[a-zA-Z]\.\s*)", val)

    match_suffix = re.search(r"(\s+[a-zA-Z]|[\s:\-.,'\"/()]+)$", val)
    while match_suffix and match_suffix.start() < len(val):
        cut = len(val) - match_suffix.start()
        if cut >= len(val):
            break
        end -= cut
        val = text[start:end]
        match_suffix = re.search(r"(\s+[a-zA-Z]|[\s:\-.,'\"/()]+)$", val)

    return val.strip(), start, end


class IndianHybridNER:
    """
    Manages loading and dual-pass inference for Indian-focused NER pipelines.
    Runs HiNER, IndicNER, and XLM-RoBERTa (with graceful fallback if any model is unavailable).
    """

    def __init__(self, token: Optional[str] = None, device: int = -1):
        self.device = device
        self.token = token or os.getenv("HF_TOKEN") or None
        self.pipelines: Dict[str, Tuple[Any, float]] = {}
        self._init_models()

    def _init_models(self):
        if not TRANSFORMERS_AVAILABLE:
            log.warning("Transformers not installed — Indian Hybrid NER unavailable.")
            return

        enabled_models_env = os.getenv("INDIAN_NER_MODELS", "").strip()
        filter_names = [m.strip().lower() for m in enabled_models_env.split(",") if m.strip()]

        for label, model_id, use_fast, min_score in INDIAN_NER_MODEL_SPECS:
            if filter_names and label.lower() not in filter_names:
                continue

            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_id, token=self.token, use_fast=use_fast
                )
                model = AutoModelForTokenClassification.from_pretrained(
                    model_id, token=self.token
                )
                model.eval()

                pipe = tf_pipeline(
                    task="ner",
                    model=model,
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    device=self.device,
                )
                self.pipelines[label] = (pipe, min_score)
                log.info(f"  [OK] Indian NER loaded: {label} ({model_id})")
            except Exception as exc:
                log.warning(f"  [SKIP] Indian NER model '{label}' ({model_id}) unavailable: {exc}")

        # Always check general fallback BERT if nothing else loaded
        if not self.pipelines:
            try:
                fallback_pipe = tf_pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple",
                    device=self.device,
                )
                self.pipelines["BERT_Base"] = (fallback_pipe, 0.50)
                log.info("  [OK] General fallback NER loaded: BERT_Base (dslim/bert-base-NER)")
            except Exception as e:
                log.warning(f"  [WARN] General fallback BERT init failed: {e}")

    def is_available(self) -> bool:
        return len(self.pipelines) > 0

    def _parse_pipe_output(self, results: Any, line_str: str, min_score: float, model_name: str) -> List[Dict]:
        if not results:
            return []
        raw_spans = []
        for item in results:
            if not isinstance(item, dict):
                continue
            score = float(item.get("score", 0.0))
            if score < min_score:
                continue

            raw_cat = item.get("entity_group", item.get("entity", ""))
            norm_cat = normalize_entity_label(raw_cat)
            if not norm_cat:
                continue

            start = int(item.get("start", 0))
            end = int(item.get("end", 0))
            if start < end:
                val = line_str[start:end]
                raw_spans.append({
                    "cat": norm_cat,
                    "start": start,
                    "end": end,
                    "score": score,
                    "text": val,
                    "model": model_name,
                })
        return raw_spans

    def _run_single_pass(self, pipe, text: str, min_score: float, model_name: str) -> List[Dict]:
        if not text.strip():
            return []
        try:
            if TRANSFORMERS_TORCH_AVAILABLE and hasattr(torch, "inference_mode"):
                with torch.inference_mode():
                    res = pipe(text)
            else:
                res = pipe(text)
            return self._parse_pipe_output(res, text, min_score, model_name)
        except Exception as exc:
            log.debug(f"NER pass failed on '{text[:30]}...' with {model_name}: {exc}")
            return []

    def predict_entities(self, text: str, min_score_override: Optional[float] = None) -> List[Dict]:
        """
        Runs dual-pass (verbatim + truecase) entity extraction across all loaded models.
        Returns deduplicated entities:
          [{'cat': 'PERSON', 'text': 'Ramesh Kumar', 'score': 0.95, 'model': 'HiNER'}]
        """
        if not text or not text.strip() or not self.pipelines:
            return []

        clean_text = text.strip()
        tc_text = truecase_line(clean_text)
        is_all_caps = clean_text.isupper()

        raw_candidates: List[Dict] = []

        for model_label, (pipe, default_min_score) in self.pipelines.items():
            threshold = min_score_override if min_score_override is not None else default_min_score

            # Pass 1: Verbatim string
            spans_v = self._run_single_pass(pipe, clean_text, threshold, model_label)
            raw_candidates.extend(spans_v)

            # Pass 2: Truecased string (if different)
            if tc_text != clean_text:
                spans_tc = self._run_single_pass(pipe, tc_text, threshold, model_label)
                # Map extracted text back to clean_text slice
                for s in spans_tc:
                    st, en = s["start"], s["end"]
                    s["text"] = clean_text[st:en]
                    # If line was all caps and verbatim had no hit, prioritize PERSON over ORG/LOC
                    if is_all_caps and not spans_v and s["cat"] != "PERSON":
                        continue
                    raw_candidates.append(s)

        if not raw_candidates:
            return []

        # Process, snap boundaries, and filter
        final_entities = []
        seen = set()

        for cand in raw_candidates:
            cat = cand["cat"]
            st, en = _snap_to_word_boundary(clean_text, cand["start"], cand["end"])
            cleaned_val, st, en = _clean_entity_text(clean_text, st, en)

            if not cleaned_val or len(cleaned_val) < 2:
                continue

            val_upper = cleaned_val.upper()

            # Ignore pure numbers or single initials
            if re.match(r"^\d+$", val_upper) or re.match(r"^[A-Z]\.?$", val_upper):
                continue

            # Clinical & Non-PII exclusions check
            if val_upper in CLINICAL_ALLOWLIST or val_upper in NON_PII_EXCLUSIONS:
                continue

            # If tokens are all purely safe clinical terms, skip
            tokens = re.findall(r"[A-Z0-9]+", val_upper)
            if tokens and all(t in CLINICAL_ALLOWLIST or re.match(r"^(?:L|R|C|T|S)\d{1,2}$", t) for t in tokens):
                continue

            key = (cat, cleaned_val.lower(), st, en)
            if key not in seen:
                seen.add(key)
                final_entities.append({
                    "cat": cat,
                    "text": cleaned_val,
                    "score": round(cand["score"], 4),
                    "model": cand["model"],
                    "start": st,
                    "end": en,
                })

        return final_entities

    def __call__(self, text: str) -> List[Dict[str, Any]]:
        """
        HuggingFace pipeline compatible interface.
        Returns [{'entity_group': cat, 'score': score, 'word': text, 'start': start, 'end': end}]
        """
        entities = self.predict_entities(text)
        return [
            {
                "entity_group": ent["cat"],
                "score": ent["score"],
                "word": ent["text"],
                "start": ent.get("start", 0),
                "end": ent.get("end", 0),
                "model": ent.get("model", ""),
            }
            for ent in entities
        ]

