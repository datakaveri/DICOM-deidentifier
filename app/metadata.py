"""
metadata.py — no longer used by pipeline.py.

Tag-level de-identification (hash PatientID, mask dates, suppress direct
identifiers, etc., per DICOM tag) is now handled by the more complete
de_identification.deidentify.deidentify_dataset(), driven by the
tag -> technique mapping in de_identification/tag_mapping.py. This module is
kept only so any external caller still importing it doesn't hard-fail, but
sanitize_metadata() here is a no-op beyond private-tag stripping.
"""

from config import log


def sanitize_metadata(ds):
    """Strips private/vendor tags only. See de_identification/deidentify.py
    for actual PHI tag-value anonymization (hash/mask/suppress per tag)."""
    log.info("  [Stage 1] Stripping private tags...")
    ds.remove_private_tags()
    log.info("  [Stage 1] Done.")
    return ds
