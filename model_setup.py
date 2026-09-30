#!/usr/bin/env python3
# ╔══════════════════════════════════════════════════════════════════════════════╗
# ║  MODEL_SETUP.PY — Indian Hybrid NER & Biomedical Model Setup                 ║
# ║                                                                              ║
# ║  Downloads and caches all required models for offline pipeline execution:    ║
# ║    1. HiNER (IIT Bombay / MuRIL)          cfilt/HiNER-original-muril-base-cased
# ║    2. IndicNER (AI4Bharat)                ai4bharat/IndicNER                 ║
# ║    3. XLM-RoBERTa (Multilingual)          Babelscape/wikineural-multilingual ║
# ║    4. General NER (English Fallback)      dslim/bert-base-NER                ║
# ║    5. Biomedical NER (Primary)            d4data/biomedical-ner-all          ║
# ║    6. Clinical Apollo (Medical Fallback)  Clinical-AI-Apollo/Medical-NER     ║
# ╚══════════════════════════════════════════════════════════════════════════════╝

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor

MODELS = [
    (
        "HiNER (IIT Bombay / MuRIL)",
        "cfilt/HiNER-original-muril-base-cased",
        False,
        "~900 MB"
    ),
    (
        "IndicNER (AI4Bharat)",
        "ai4bharat/IndicNER",
        False,
        "~900 MB"
    ),
    (
        "XLM-RoBERTa (WikiNEural)",
        "Babelscape/wikineural-multilingual-ner",
        True,
        "~1.1 GB"
    ),
    (
        "General English NER",
        "dslim/bert-base-NER",
        True,
        "~400 MB"
    ),
    (
        "Biomedical NER (Primary)",
        "d4data/biomedical-ner-all",
        True,
        "~400 MB"
    ),
    (
        "Clinical Apollo NER (Fallback)",
        "Clinical-AI-Apollo/Medical-NER",
        True,
        "~400 MB"
    ),
]


def prefetch_models(token: str | None = None, max_workers: int = 3) -> int:
    """Downloads all models in parallel to the local HuggingFace cache."""
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except ImportError:
        print("  [ERROR] transformers library not found. Install it first: pip install transformers", flush=True)
        return 1

    print("\n" + "=" * 75)
    print("  PREFETCHING INDIAN HYBRID NER & BIOMEDICAL MODELS")
    print(f"  Downloading {len(MODELS)} models with {max_workers} worker threads...")
    print("=" * 75 + "\n", flush=True)

    hf_token = token or os.getenv("HF_TOKEN") or None

    def _fetch_one(spec):
        desc, model_id, use_fast, size = spec
        print(f"  --> Starting download: {desc} ({model_id}) [{size}]", flush=True)
        try:
            AutoTokenizer.from_pretrained(model_id, token=hf_token, use_fast=use_fast)
            AutoModelForTokenClassification.from_pretrained(model_id, token=hf_token)
            return model_id, None
        except Exception as exc:
            return model_id, exc

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        results = list(pool.map(_fetch_one, MODELS))

    failed = 0
    print("\n" + "-" * 75)
    print("  DOWNLOAD SUMMARY")
    print("-" * 75)
    for model_id, exc in results:
        if exc is None:
            print(f"  [CACHED] {model_id}")
        else:
            failed += 1
            print(f"  [FAILED] {model_id}: {exc}")

    if failed:
        print("\n  [NOTICE] Some gated or restricted models failed to download.")
        print("           Provide your Hugging Face access token via: --token <YOUR_HF_TOKEN>")
        print("           or by setting the HF_TOKEN environment variable.")
    else:
        print("\n  [SUCCESS] All models cached successfully. Ready for offline inference!")
    print("=" * 75 + "\n")
    return failed


def parse_args():
    parser = argparse.ArgumentParser(description="Prefetch and cache Indian Hybrid NER & Biomedical models.")
    parser.add_argument("--prefetch", action="store_true", default=True, help="Download all model weights to cache.")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN", ""), help="Hugging Face User Access Token.")
    parser.add_argument("--workers", type=int, default=3, help="Concurrent download workers.")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero code if any model fails.")
    return parser.parse_args()


def main():
    args = parse_args()
    failed = prefetch_models(token=args.token, max_workers=args.workers)
    if failed and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
