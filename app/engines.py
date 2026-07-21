"""
engines.py — optional OCR/NLP engine imports, GPU auto-detection, and
engine initialization (PaddleOCR, EasyOCR, Presidio Analyzer).
"""

import logging

from config import log

# ─── Optional engine imports ──────────────────────────────────────────────────
try:
    from paddleocr import PaddleOCR, logger as paddle_logger
    paddle_logger.setLevel(logging.ERROR)
    PADDLE_AVAILABLE = True
except Exception:
    PADDLE_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except Exception:
    EASYOCR_AVAILABLE = False

try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except Exception:
    PRESIDIO_AVAILABLE = False


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
    except ImportError:
        pass
    return False


def initialize_engines(use_gpu=False):
    """Initializes PaddleOCR, EasyOCR, and Presidio Analyzer with GPU option."""
    print(f"\nInitializing OCR engines (GPU={use_gpu})...")

    paddle_ocr = None
    if PADDLE_AVAILABLE:
        try:
            # First attempt: Try with GPU and MKL-DNN toggles
            paddle_ocr = PaddleOCR(lang='en', enable_mkldnn=False, use_gpu=use_gpu)
            print(f"  [OK] PaddleOCR initialized (use_gpu={use_gpu})")
        except Exception as e1:
            # Second attempt: Try without use_gpu (if not accepted by this version)
            try:
                paddle_ocr = PaddleOCR(lang='en', enable_mkldnn=False)
                print("  [OK] PaddleOCR initialized (without explicit use_gpu)")
            except Exception as e2:
                # Third attempt: Try with bare minimum
                try:
                    paddle_ocr = PaddleOCR(lang='en')
                    print("  [OK] PaddleOCR initialized (default settings)")
                except Exception as e3:
                    print(f"  [WARN] PaddleOCR failed to init: {e3}")

    easy_ocr = None
    if EASYOCR_AVAILABLE:
        try:
            easy_ocr = easyocr.Reader(['en'], gpu=use_gpu)
            print("  [OK] EasyOCR initialized")
        except Exception as e:
            print(f"  [WARN] EasyOCR failed to init: {e}")

    analyzer = None
    if PRESIDIO_AVAILABLE:
        try:
            # AnalyzerEngine() defaults to spacy's en_core_web_lg (~600MB) if
            # not told otherwise. We only install en_core_web_sm (~12MB), so
            # pin the NLP engine to it explicitly instead of triggering an
            # unwanted large-model download.
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            nlp_engine = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}],
            }).create_engine()
            analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            print("  [OK] Presidio NLP Analyzer initialized (en_core_web_sm)")
        except Exception as e:
            print(f"  [WARN] Presidio failed to init: {e}")

    if not paddle_ocr and not easy_ocr:
        print("\n[ERROR] No OCR engine available! Install at least one:")
        print("  pip install paddleocr   OR   pip install easyocr")
        print("Metadata anonymization will still run.\n")

    return paddle_ocr, easy_ocr, analyzer
