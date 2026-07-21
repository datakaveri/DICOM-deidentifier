"""
de_identification — DICOM metadata (tag-value) anonymization, driven by the
tag -> technique mapping in tag_mapping.py and implemented with the
crypto/FPE primitives ported from k-anonymisation/SKALD/src/pipeline/preprocess.

This complements, and is independent of, the pixel/burned-in-text
anonymization pipeline in the parent app package (metadata.py,
masking.py, verify.py, etc.) — that pipeline deliberately leaves tag
VALUES untouched; this package is what actually anonymizes them.
"""
