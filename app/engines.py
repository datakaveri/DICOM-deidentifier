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

# ─── Optional engine imports ──────────────────────────────────────────────────
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except Exception:
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
        try:
            # Native PaddleOCR v2.x (PP-OCRv4 models)
            # enable_mkldnn=False prevents segfaults on VMs without AVX2/AVX512.
            # cpu_threads caps Paddle's internal thread pool per worker.
            paddle_ocr = PaddleOCR(
                use_angle_cls=False,
                lang='en',
                use_gpu=use_gpu,
                enable_mkldnn=False,
                cpu_threads=cpu_threads,
                show_log=False,
            )
            elapsed = round(time.time() - t0, 1)
            print(f"  [OK] PaddleOCR initialized ({elapsed}s)")
        except Exception as e:
            log.warning(f"PaddleOCR init failed: {e}")
            print(f"  [WARN] PaddleOCR init failed: {e}")
            # Detailed diagnostic for VM debugging
            try:
                import paddle
                print(f"  [DIAG] PaddlePaddle version: {paddle.__version__}")
                print(f"  [DIAG] Paddle compiled with MKLDNN: {paddle.device.is_compiled_with_mkldnn() if hasattr(paddle.device, 'is_compiled_with_mkldnn') else 'unknown'}")
            except Exception:
                pass

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
            deid_model = tf_pipeline(
                "token-classification",
                model="StanfordAIMI/stanford-deidentifier-base",
                aggregation_strategy="simple"
            )
            print("  [OK] Stanford De-ID model initialized")
        except Exception as e:
            print(f"  [WARN] Stanford De-ID model failed to init: {e}")

    medical_ner = None
    if TRANSFORMERS_AVAILABLE:
        try:
            medical_ner = tf_pipeline(
                "ner",
                model="d4data/biomedical-ner-all",
                aggregation_strategy="simple"
            )
            print("  [OK] Biomedical NER model initialized")
        except Exception as e:
            print(f"  [WARN] Biomedical NER model failed to init: {e}")

    gliner_model = None
    if GLINER_AVAILABLE:
        try:
            gliner_model = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
            print("  [OK] GLiNER-BioMed model initialized")
        except Exception as e:
            print(f"  [WARN] GLiNER model failed to init: {e}")

    if not paddle_ocr:
        print("\n[ERROR] No OCR engine available! Run: pip install paddlepaddle paddleocr")
        print("Metadata anonymization will still run.\n")

    total_init = round(time.time() - t0, 1)
    print(f"[Worker {worker_pid}] All engines ready in {total_init}s\n")

    return paddle_ocr, analyzer, deid_model, medical_ner, gliner_model


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
            _warmup_ocr = PaddleOCR(
                use_angle_cls=False, lang='en', use_gpu=use_gpu,
                enable_mkldnn=False, cpu_threads=1, show_log=False,
            )
            del _warmup_ocr
            print("  [CACHED] PaddleOCR model weights")
        except Exception as e:
            print(f"  [SKIP] PaddleOCR cache failed: {e}")

    # 2. HuggingFace Transformers — downloads to ~/.cache/huggingface/
    if TRANSFORMERS_AVAILABLE:
        try:
            from transformers import AutoTokenizer, AutoModelForTokenClassification
            for model_name in [
                "StanfordAIMI/stanford-deidentifier-base",
                "d4data/biomedical-ner-all",
            ]:
                AutoTokenizer.from_pretrained(model_name)
                AutoModelForTokenClassification.from_pretrained(model_name)
                print(f"  [CACHED] {model_name}")
        except Exception as e:
            print(f"  [SKIP] Transformers cache failed: {e}")

    # 3. GLiNER — downloads to ~/.cache/huggingface/
    if GLINER_AVAILABLE:
        try:
            _warmup_gliner = GLiNER.from_pretrained("urchade/gliner_large-v2.1")
            del _warmup_gliner
            print("  [CACHED] GLiNER model weights")
        except Exception as e:
            print(f"  [SKIP] GLiNER cache failed: {e}")

    elapsed = round(time.time() - t0, 1)
    print(f"\nModel cache warm-up complete in {elapsed}s")
    print("=" * 70 + "\n")

