"""
engines.py — OCR/NLP engine imports, GPU auto-detection, and
engine initialization (PaddleOCR, Presidio Analyzer, Stanford De-ID, Biomedical NER, GLiNER).
"""

import logging

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


def initialize_engines(use_gpu=False):
    """
    Initializes PaddleOCR as the sole OCR engine.
    Also initializes Presidio Analyzer, Stanford De-ID, Biomedical NER, and GLiNER.
    """
    print(f"\nInitializing OCR and AI engines (GPU={use_gpu})...")

    paddle_ocr = None
    if PADDLE_AVAILABLE:
        try:
            # Native PaddleOCR (Linux/Ubuntu - PP-OCRv4 models)
            # Try with use_gpu parameter first (older PaddleOCR)
            paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', use_gpu=use_gpu)
            print("  [OK] PaddleOCR (Native) initialized as PRIMARY OCR")
        except (TypeError, ValueError) as e:
            try:
                # Newer PaddleOCR versions may accept device='gpu'/'cpu'
                device_str = 'gpu' if use_gpu else 'cpu'
                paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en', device=device_str)
                print(f"  [OK] PaddleOCR (Native, device={device_str}) initialized as PRIMARY OCR")
            except Exception:
                try:
                    # Or simple init without use_gpu/device
                    paddle_ocr = PaddleOCR(use_angle_cls=False, lang='en')
                    print("  [OK] PaddleOCR (Native) initialized as PRIMARY OCR")
                except Exception as ex:
                    log.warning(f"Native PaddleOCR init fallback failed: {ex}")
                    print(f"  [WARN] PaddleOCR init failed: {ex}")
        except Exception as e:
            log.warning(f"Native PaddleOCR init failed: {e}")
            print(f"  [WARN] PaddleOCR init failed: {e}")

    analyzer = None
    if PRESIDIO_AVAILABLE:
        try:
            from presidio_analyzer.nlp_engine import NlpEngineProvider
            provider = NlpEngineProvider(nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": "en_core_web_sm"}]
            })
            nlp_engine = provider.create_engine()
            analyzer = AnalyzerEngine(nlp_engine=nlp_engine)
            print("  [OK] Presidio NLP Analyzer initialized (en_core_web_sm)")
        except Exception as e:
            try:
                analyzer = AnalyzerEngine()
                print("  [OK] Presidio NLP Analyzer initialized (default)")
            except Exception as ex:
                print(f"  [WARN] Presidio failed to init: {ex}")

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

    return paddle_ocr, analyzer, deid_model, medical_ner, gliner_model

