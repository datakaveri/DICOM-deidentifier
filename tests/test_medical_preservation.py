import pytest
import os
import sys

# Ensure app is on sys.path
app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, app_dir)

from classify import classify_phi

def test_medical_markers_preserved():
    """Verify that clinical markers and anatomy are preserved."""
    test_cases = [
        {"text": "L", "bbox": [10, 10, 30, 30]},
        {"text": "R", "bbox": [10, 10, 30, 30]},
        {"text": "L5", "bbox": [10, 10, 40, 30]},
        {"text": "CHEST PA ERECT", "bbox": [10, 10, 150, 30]},
        {"text": "KVP 120 MAS 3.2", "bbox": [10, 10, 150, 30]},
        {"text": "NORMAL STUDY", "bbox": [10, 10, 150, 30]},
        {"text": "CARDIOMEGALY", "bbox": [10, 10, 150, 30]},
    ]
    # classify_phi without optional ML models relies on allowlist/regex
    redacted = classify_phi(test_cases, (1024, 1024))
    redacted_texts = [r["text"] for r in redacted]
    for tc in test_cases:
        assert tc["text"] not in redacted_texts, f"Expected '{tc['text']}' to be KEPT, but it was redacted!"

def test_pii_redacted():
    """Verify that demographic and PII patterns are flagged for redaction."""
    test_cases = [
        {"text": "PATIENT: RAMESH KUMAR", "bbox": [10, 10, 200, 30]},
        {"text": "AIIMS HOSPITAL DELHI", "bbox": [10, 10, 200, 30]},
        {"text": "UHID: 98472910", "bbox": [10, 10, 200, 30]},
        {"text": "DR. RAJESH SHARMA", "bbox": [10, 10, 200, 30]},
        {"text": "DOB: 14/08/1981 AGE: 45Y", "bbox": [10, 10, 200, 30]},
        {"text": "DATE: 28/09/2026", "bbox": [10, 10, 200, 30]},
    ]
    redacted = classify_phi(test_cases, (1024, 1024))
    redacted_texts = [r["text"] for r in redacted]
    for tc in test_cases:
        assert tc["text"] in redacted_texts, f"Expected '{tc['text']}' to be REDACTED, but it was kept!"

def test_apollo_fallback_preservation():
    """Verify that fallback_medical_ner is triggered when primary NER finds nothing."""
    mock_primary_ner = lambda txt: [] # Primary finds nothing
    mock_fallback_ner = lambda txt: [
        {"word": "pneumothorax", "entity_group": "DISEASE_DISORDER", "score": 0.85}
    ]
    test_case = [{"text": "PNEUMOTHORAX", "bbox": [10, 10, 100, 30]}]
    redacted = classify_phi(
        test_case, (1024, 1024),
        medical_ner=mock_primary_ner,
        fallback_medical_ner=mock_fallback_ner
    )
    redacted_texts = [r["text"] for r in redacted]
    assert "PNEUMOTHORAX" not in redacted_texts, "Fallback NER entity should have been preserved!"
