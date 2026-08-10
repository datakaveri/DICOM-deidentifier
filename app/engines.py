"""
engines.py — optional OCR/NLP engine imports, GPU auto-detection, and
engine initialization (PaddleOCR, EasyOCR, Presidio Analyzer).
"""

import logging

from config import log

# ─── Optional engine imports ──────────────────────────────────────────────────
# Note (per README): running PaddlePaddle and PyTorch in the same Python process
# corrupts the heap once both load model weights on Windows/CPU. EasyOCR handles OCR.
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
    except ImportError:
        pass
    return False


def initialize_engines(use_gpu=False):
    """Initializes PaddleOCR, EasyOCR, Presidio Analyzer, Stanford De-ID, Biomedical NER, and GLiNER with GPU option."""
    print(f"\nInitializing OCR and AI engines (GPU={use_gpu})...")

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

    if not paddle_ocr and not easy_ocr:
        print("\n[ERROR] No OCR engine available! Install at least one:")
        print("  pip install paddleocr   OR   pip install easyocr")
        print("Metadata anonymization will still run.\n")

    return paddle_ocr, easy_ocr, analyzer, deid_model, medical_ner, gliner_model

