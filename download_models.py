#!/usr/bin/env python3
"""
download_models.py — One-Click Model Downloader for Indian Hybrid NER & Biomedical Stack.

How to use:
  1. Paste your Hugging Face token into the HF_TOKEN variable below, OR just run the script:
     python download_models.py
  2. If the variable is left blank, the script will prompt you to paste it in the terminal.
"""

import os
import sys

# ==============================================================================
# 🔑 PASTE YOUR HUGGING FACE TOKEN HERE (Optional: or enter it when prompted)
# ==============================================================================
HF_TOKEN = ""
# ==============================================================================


def get_token():
    # Check .env first
    if not os.getenv("HF_TOKEN") and os.path.exists(".env"):
        try:
            with open(".env") as f:
                for line in f:
                    if line.strip().startswith("HF_TOKEN="):
                        os.environ["HF_TOKEN"] = line.strip().split("=", 1)[1].strip().strip('"\'')
        except Exception:
            pass

    token = HF_TOKEN.strip()
    if not token:
        token = os.getenv("HF_TOKEN", "").strip()

    if not token:
        print("\n" + "=" * 70)
        print("🔑 Hugging Face Authentication")
        print("=" * 70)
        try:
            token = input("Paste your Hugging Face token (starts with hf_...) [or press Enter to skip]: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted.")
            sys.exit(0)
    return token or None


def main():
    token = get_token()
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
        print("✓ Token loaded.\n")
    else:
        print("⚠ No token provided. Public models will download; gated models may be skipped.\n")

    # Import dependencies
    try:
        from transformers import AutoTokenizer, AutoModelForTokenClassification
    except ImportError:
        print("Installing required dependencies (transformers, sentencepiece, protobuf)...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers>=4.40.0", "sentencepiece", "protobuf"])
        from transformers import AutoTokenizer, AutoModelForTokenClassification

    models = [
        ("HiNER (IIT Bombay / MuRIL - Indian Multilingual)", "cfilt/HiNER-original-muril-base-cased", False),
        ("IndicNER (AI4Bharat - South Asian NER)", "ai4bharat/IndicNER", False),
        ("XLM-RoBERTa (WikiNEural Multilingual NER)", "Babelscape/wikineural-multilingual-ner", True),
        ("General English NER (dslim/bert-base-NER)", "dslim/bert-base-NER", True),
        ("Primary Biomedical NER (d4data)", "d4data/biomedical-ner-all", True),
        ("Fallback Clinical NER (Clinical-AI-Apollo)", "Clinical-AI-Apollo/Medical-NER", True),
    ]

    print("=" * 70)
    print(f"Downloading {len(models)} models into local cache (~/.cache/huggingface/hub/)")
    print("=" * 70)

    success_count = 0
    for idx, (label, model_id, use_fast) in enumerate(models, 1):
        print(f"\n[{idx}/{len(models)}] Downloading: {label}")
        print(f"      Repository: {model_id} ...", end=" ", flush=True)
        try:
            AutoTokenizer.from_pretrained(model_id, token=token, use_fast=use_fast)
            AutoModelForTokenClassification.from_pretrained(model_id, token=token)
            print("✓ DONE")
            success_count += 1
        except Exception as e:
            print("✗ FAILED")
            print(f"      Reason: {e}")
            if "gated" in str(e).lower() or "401" in str(e) or "403" in str(e):
                print(f"      💡 Note: Visit https://huggingface.co/{model_id} and click 'Agree / Access repository'")

    print("\n" + "=" * 70)
    print(f"Result: {success_count}/{len(models)} models successfully downloaded and cached.")
    print("=" * 70)
    if success_count == len(models):
        print("🎉 All models are ready! You can now run the de-identification pipeline completely offline.")
    else:
        print("Pipeline will use cached models and fallbacks automatically.")


if __name__ == "__main__":
    main()
