"""
Integration test for the DICOM de-identification pipeline components.
"""
import re
from config import CLINICAL_ALLOWLIST, PII_PATTERNS
from classify import _is_clinical


def test_clinical_allowlist_integrity():
    assert "LEFT" in CLINICAL_ALLOWLIST
    assert "RIGHT" in CLINICAL_ALLOWLIST
    assert "CHEST" in CLINICAL_ALLOWLIST
    assert _is_clinical("CHEST") is True
    assert _is_clinical("LEFT") is True
    assert _is_clinical("RIGHT") is True


def test_pii_regex_patterns():
    assert "phone" in PII_PATTERNS
    assert "aadhaar" in PII_PATTERNS
    assert "abha" in PII_PATTERNS

    phone_pattern = PII_PATTERNS["phone"]
    aadhaar_pattern = PII_PATTERNS["aadhaar"]

    assert re.search(phone_pattern, "Mobile: 9876543210") is not None
    assert re.search(aadhaar_pattern, "UID: 1234 5678 9012") is not None
