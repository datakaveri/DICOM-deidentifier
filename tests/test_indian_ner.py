import os
import sys
import pytest

# Ensure app is on sys.path
app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, app_dir)

from indian_ner import truecase_line, normalize_entity_label, _snap_to_word_boundary, _clean_entity_text
from config import PII_PATTERNS, NON_PII_EXCLUSIONS
from classify import classify_phi
import re


def test_truecase_line():
    """Verify truecasing preserves exact string length and casing structure for transformers."""
    all_caps = "PATIENT: RAMESHWAR DESHMUKH"
    tc = truecase_line(all_caps)
    assert len(tc) == len(all_caps), "Truecased string length must match original string"
    assert tc == "Patient: Rameshwar Deshmukh"

    single_word = "HYDERABAD"
    assert truecase_line(single_word) == "Hyderabad"


def test_normalize_entity_label():
    """Verify BIO tag stripping and entity normalization."""
    assert normalize_entity_label("B-PER") == "PERSON"
    assert normalize_entity_label("I-PER") == "PERSON"
    assert normalize_entity_label("PERSON") == "PERSON"
    assert normalize_entity_label("B-LOC") == "LOCATION"
    assert normalize_entity_label("ORG") == "ORGANIZATION"
    assert normalize_entity_label("MISC") == "MISC"
    assert normalize_entity_label(None) is None


def test_snap_and_clean_entity_text():
    """Verify boundary snapping and punctuation trimming."""
    line = "PATIENT: MEERA IYER, AGE 34"
    # span for 'MEERA' is 9..14
    st, en = _snap_to_word_boundary(line, 9, 14)
    assert line[st:en] == "MEERA"

    cleaned, c_st, c_en = _clean_entity_text(line, 7, 19)
    assert cleaned == "MEERA IYER"


def test_indian_pii_patterns():
    """Verify regex patterns for Indian demographic and health identifiers."""
    # PAN Card
    pan_text = "PAN NO: ABCDE1234F"
    assert re.search(PII_PATTERNS["pan"], pan_text) is not None

    # Aadhaar Number
    aadhaar_text = "AADHAAR: 2345 6789 0123"
    assert re.search(PII_PATTERNS["aadhaar"], aadhaar_text) is not None

    # Indian Phone (+91 and 10 digits)
    phone_text_1 = "PHONE: +91 9876543210"
    phone_text_2 = "CONTACT: 9876543210"
    assert re.search(PII_PATTERNS["phone"], phone_text_1) is not None
    assert re.search(PII_PATTERNS["phone"], phone_text_2) is not None

    # Indian Voter ID
    voter_text = "EPIC NO: WBD1234567"
    assert re.search(PII_PATTERNS["voter_id"], voter_text) is not None

    # Indian Driving License
    dl_text = "DL: MH-1420110012345"
    assert re.search(PII_PATTERNS["driving_license"], dl_text) is not None

    # Indian Pincode
    pin_text = "NAGPUR 440010"
    assert re.search(PII_PATTERNS["pincode"], pin_text) is not None

    # Age pattern
    age_text = "AGED ABOUT 52 YEARS"
    assert re.search(PII_PATTERNS["age_years"], age_text) is not None


def test_non_pii_exclusions():
    """Verify non-PII clerical terms do not trigger demographic redaction."""
    for token in ["NO", "DATE", "STATUS", "REF", "PLEASE", "SL.NO", "SR.NO"]:
        assert token in NON_PII_EXCLUSIONS

    # Test that a safe line containing clerical words is kept
    clerical_cases = [
        {"text": "SL.NO 102", "bbox": [10, 10, 80, 30]},
        {"text": "STATUS NORMAL", "bbox": [10, 10, 120, 30]},
    ]
    redacted = classify_phi(clerical_cases, (1024, 1024))
    redacted_texts = [r["text"] for r in redacted]
    for c in clerical_cases:
        assert c["text"] not in redacted_texts, f"Expected '{c['text']}' to be KEPT, but it was redacted!"


def test_mock_indian_hybrid_ner_in_classify_phi():
    """Verify that Indian names detected by the Indian NER engine are redacted."""
    class MockIndianNER:
        def predict_entities(self, text):
            if "VENKATA SATYANARAYANA" in text.upper() or "Venkata Satyanarayana" in text:
                return [{"cat": "PERSON", "text": "VENKATA SATYANARAYANA", "score": 0.94, "model": "HiNER"}]
            if "KAVITHA REDDY" in text.upper() or "Kavitha Reddy" in text:
                return [{"cat": "PERSON", "text": "KAVITHA REDDY", "score": 0.91, "model": "IndicNER"}]
            if "NIZAMS INSTITUTE" in text.upper():
                return [{"cat": "ORGANIZATION", "text": "NIZAMS INSTITUTE", "score": 0.88, "model": "XLM_RoBERTa"}]
            return []

    mock_ner = MockIndianNER()

    test_cases = [
        {"text": "VENKATA SATYANARAYANA", "bbox": [10, 10, 220, 30]},
        {"text": "DR. KAVITHA REDDY", "bbox": [10, 10, 180, 30]},
        {"text": "NIZAMS INSTITUTE OF MEDICAL SCIENCES", "bbox": [10, 10, 300, 30]},
        {"text": "CHEST PA ERECT", "bbox": [10, 10, 150, 30]},
        {"text": "L26", "bbox": [10, 10, 40, 30]},
    ]

    redacted = classify_phi(test_cases, (1024, 1024), indian_ner=mock_ner)
    redacted_texts = [r["text"] for r in redacted]

    assert "VENKATA SATYANARAYANA" in redacted_texts
    assert "DR. KAVITHA REDDY" in redacted_texts
    assert "NIZAMS INSTITUTE OF MEDICAL SCIENCES" in redacted_texts
    assert "CHEST PA ERECT" not in redacted_texts
    assert "L26" not in redacted_texts
