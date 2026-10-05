"""
engines.py — OCR/NLP engine imports, GPU auto-detection, and
engine initialization (PaddleOCR, Presidio Analyzer, Stanford De-ID, Biomedical NER, GLiNER).

Optimizations for parallel processing:
  - PaddleOCR initialized with MKLDNN disabled and CPU threads capped.
  - pre_warm_model_cache() downloads all model weights to local disk cache
    once in the main process, so workers load from cache instead of re-downloading.
"""

import os
import logging
import time

from config import log

import paddle_env  # noqa: F401 — must be first to set MKLDNN and thread env flags

# ─── Optional engine imports ──────────────────────────────────────────────────
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception as e:
    log.warning(f"Failed to import PaddleOCR: {e}")
    PADDLE_AVAILABLE = False

try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except Exception:
    PRESIDIO_AVAILABLE = False

try:
    from transformers import pipeline as tf_pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

try:
    from gliner import GLiNER
    GLINER_AVAILABLE = True
except Exception:
    GLINER_AVAILABLE = False


def check_gpu_available():
    """Checks if a GPU is available via PyTorch or Paddle."""
    try:
        import torch
        if torch.cuda.is_available():
            return True
    except ImportError:
        pass
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            return True
    except Exception:
        pass
    return False


def _get_cpu_threads():
    """Returns the per-worker thread cap from env (set by paddle_env / main_parallel)."""
    return int(os.environ.get("SKALD_PADDLE_THREADS",
               os.environ.get("SKALD_THREADS_PER_WORKER", "2")))


def initialize_engines(use_gpu=False):
    """
    Initializes PaddleOCR as the sole OCR engine.
    Also initializes Presidio Analyzer, Stanford De-ID, Biomedical NER, and GLiNER.

    PaddleOCR is initialized with MKLDNN disabled and CPU threads capped
    to prevent thread explosion under multiprocessing.
    """
    cpu_threads = _get_cpu_threads()
    worker_pid = os.getpid()
    t0 = time.time()
    print(f"\n[Worker {worker_pid}] Initializing OCR and AI engines (GPU={use_gpu}, cpu_threads={cpu_threads})...")

    paddle_ocr = None
    if PADDLE_AVAILABLE:
        det_limit = int(os.environ.get("PADDLE_DET_LIMIT_SIDE_LEN", "2048"))
        det_db_thresh = float(os.environ.get("PADDLE_DET_DB_THRESH", "0.25"))
        try:
            paddle_ocr = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                use_gpu=use_gpu,
                enable_mkldnn=False,
                cpu_threads=cpu_threads,
                det_limit_side_len=det_limit,
                det_db_thresh=det_db_thresh,
            )
        except Exception as e1:
            log.info(f"Tuned PaddleOCR init failed ({e1}), falling back to standard init...")
            try:
                paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en')
            except Exception as e2:
                log.warning(f"PaddleOCR init failed: {e2}")
                print(f"  [WARN] PaddleOCR init failed: {e2}")

        if paddle_ocr is not None:
            elapsed = round(time.time() - t0, 1)
            print(f"  [OK] PaddleOCR initialized ({elapsed}s)")

    analyzer = None
    if PRESIDIO_AVAILABLE:
        try:
            analyzer = AnalyzerEngine()
            print("  [OK] Presidio NLP Analyzer initialized")
        except Exception as e:
            print(f"  [WARN] Presidio failed to init: {e}")

    deid_model = None
    if TRANSFORMERS_AVAILABLE:
        try:
            from indian_ner import IndianHybridNER
            indian_ner_engine = IndianHybridNER()
            if indian_ner_engine.is_available():
                deid_model = indian_ner_engine
                print("  [OK] Indian Hybrid NER Ensemble (HiNER + IndicNER + XLM-RoBERTa) initialized")
            else:
                print("  [INFO] Indian Hybrid NER models not loaded; falling back to dslim/bert-base-NER")
        except Exception as e:
            print(f"  [WARN] Indian Hybrid NER init failed ({e}), falling back to BERT...")

        if deid_model is None:
            try:
                deid_model = tf_pipeline(
                    "ner",
                    model="dslim/bert-base-NER",
                    aggregation_strategy="simple"
                )
                print("  [OK] General NER model (dslim/bert-base-NER) initialized")
            except Exception as e:
                print(f"  [WARN] General NER model failed to init: {e}")

    medical_ner = None
    fallback_medical_ner = None
    if TRANSFORMERS_AVAILABLE:
        try:
            medical_ner = tf_pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple"
            )
            print("  [OK] Primary Biomedical NER (d4data/biomedical-ner-all) initialized")
        except Exception as e:
            print(f"  [WARN] Primary Biomedical NER failed to init: {e}")

        try:
            fallback_medical_ner = tf_pipeline(
                "ner",
                model="Clinical-AI-Apollo/Medical-NER",
                aggregation_strategy="simple"
            )
            print("  [OK] Fallback Medical NER (Clinical-AI-Apollo/Medical-NER) initialized")
        except Exception as e:
            print(f"  [WARN] Fallback Medical NER failed to init: {e}")

    gliner_model = None  # GLiNER disabled: medical preservation handled by BioNER + expanded allowlist
    print("  [SKIP] GLiNER disabled — medical preservation via BioNER + expanded clinical allowlist")

    if not paddle_ocr:
        print("\n[ERROR] No OCR engine available! Run: pip install paddlepaddle paddleocr")
        print("Metadata anonymization will still run.\n")

    total_init = round(time.time() - t0, 1)
    print(f"[Worker {worker_pid}] All engines ready in {total_init}s\n")

    return paddle_ocr, analyzer, deid_model, medical_ner, gliner_model, fallback_medical_ner



def pre_warm_model_cache(use_gpu=False):
    """
    Downloads / caches all model weights to local disk in the MAIN process
    BEFORE spawning workers. Workers then load from the local cache instead
    of re-downloading, cutting worker init time by 60-80%.

    This function initializes all models once, then immediately discards them.
    The on-disk cache (HuggingFace hub cache, PaddleOCR model dir) persists.
    """
    print("\n" + "=" * 70)
    print("PRE-WARMING MODEL CACHE (main process)")
    print("  Downloading / verifying model weights to local disk cache...")
    print("  Workers will load from cache — no re-download needed.")
    print("=" * 70)

    t0 = time.time()

    # 1. PaddleOCR — downloads PP-OCRv4 models to ~/.paddleocr/
    if PADDLE_AVAILABLE:
        try:
            try:
                _warmup_ocr = PaddleOCR(
                    use_angle_cls=False, lang='en', use_gpu=use_gpu,
                    enable_mkldnn=False, cpu_threads=1,
                )
            except Exception:
                _warmup_ocr = PaddleOCR(use_angle_cls=False, lang='en')
            del _warmup_ocr
            print("  [CACHED] PaddleOCR model weights")
        except Exception as e:
            print(f"  [SKIP] PaddleOCR cache failed: {e}")

    # 2. HuggingFace Transformers — downloads to ~/.cache/huggingface/
    if TRANSFORMERS_AVAILABLE:
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            token = os.getenv("HF_TOKEN") or None
            models_to_cache = [
                ("cfilt/HiNER-original-muril-base-cased", False),
                ("ai4bharat/IndicNER", False),
                ("Babelscape/wikineural-multilingual-ner", True),
                ("dslim/bert-base-NER", True),
                ("d4data/biomedical-ner-all", True),
                ("Clinical-AI-Apollo/Medical-NER", True),
            ]
            for model_name, use_fast in models_to_cache:
                try:
                    AutoTokenizer.from_pretrained(model_name, token=token, use_fast=use_fast)
                    AutoModelForTokenClassification.from_pretrained(model_name, token=token)
                    print(f"  [CACHED] {model_name}")
                except Exception as me:
                    print(f"  [SKIP] Model cache failed for {model_name}: {me}")
        except Exception as e:
            print(f"  [SKIP] Transformers cache failed: {e}")


    # 3. GLiNER — DISABLED (medical preservation handled by BioNER + expanded allowlist)
    #    Saves ~1.4 GB RAM per worker and ~0.3s classification time per file.
    print("  [SKIP] GLiNER disabled — using BioNER + expanded clinical allowlist instead")

    elapsed = round(time.time() - t0, 1)
    print(f"\nModel cache warm-up complete in {elapsed}s")
    print("=" * 70 + "\n")

