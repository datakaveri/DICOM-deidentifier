"""
metadata.py — Stage 1: metadata sanitization.
"""

from config import log


def sanitize_metadata(ds):
    """
    Per policy, tag VALUES are left exactly as-is -- no tag is hashed,
    removed, generalized, or regenerated here or anywhere else in this
    pipeline (see phi_tags.identify_phi_tags(), which only reads tags, never
    writes them). This only strips private/vendor tags. UIDs
    (StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID, ...) are left
    untouched too. PHI is removed from the pixel data instead (masking.py /
    verify.py); the tag values identified in Step 1 are used purely to help
    find that PHI in the image's burned-in text (Step 3).
    """
    log.info("  [Stage 1] Stripping private tags (all other tag values, including UIDs, left unchanged)...")

    # Remove all private / vendor tags (group numbers are odd)
    ds.remove_private_tags()

    log.info("  [Stage 1] Done.")
    return ds
