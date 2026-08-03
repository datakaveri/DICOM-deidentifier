"""
deidentify.py — walks a pydicom Dataset and applies the DICOM tag ->
technique mapping (tag_mapping.TAG_MAPPING) to every matching element,
mutating the dataset in place.

Policy (per the PDF's own notes):
  - Any tag in TAG_MAPPING gets its declared technique applied.
  - A tag mapped to "suppress" (including whole Sequences, e.g.
    (0400,0561) Original Attributes Sequence) is deleted outright.
  - A tag mapped to a non-suppress technique whose VR is SQ (e.g.
    (0010,1002) Other Patient IDs Sequence) is *not* hashed/encrypted as a
    blob — there's no scalar value to transform. Instead its items are
    recursed into, and each nested element is re-matched against
    TAG_MAPPING by its own tag id (so a nested (0010,0020) PatientID gets
    hashed exactly like a top-level one).
  - Private tags (odd group number) are always suppressed, matching the
    PDF's default private-tag policy, and this pipeline's existing
    metadata.py Stage-1 behaviour (ds.remove_private_tags()).
  - Standard tags with no entry in TAG_MAPPING are left completely
    untouched (safer than guessing at an unlisted tag's sensitivity).

Unlike combine.py (which deliberately never mutates tag values, relying
only on private-tag stripping + pixel redaction), this module performs the
tag-VALUE-level anonymization described in the PDF, and is intended to run
as an additional metadata pass alongside — not instead of — the pixel
pipeline.
"""

from .tag_mapping import TAG_MAPPING, SUPPRESS
from .operations import apply_technique
from .crypto import should_skip_value


def deidentify_dataset(ds, keystore) -> list:
    """
    Applies TAG_MAPPING to `ds` (mutated in place, including nested
    sequence items). Returns an audit trail: a list of
    {"tag", "field", "technique", "action"} dicts. No original or new
    values are logged, so the audit file itself carries no PHI.
    """
    audit = []
    _walk(ds, keystore, audit)
    return audit


def _walk(ds, keystore, audit) -> None:
    for elem in list(ds):  # list(...) snapshot: safe to delete while iterating
        tag_id = (elem.tag.group, elem.tag.element)
        is_private = (elem.tag.group % 2) == 1

        if elem.VR == "SQ":
            mapping = TAG_MAPPING.get(tag_id)
            if (mapping and mapping["technique"] == SUPPRESS) or is_private:
                audit.append({
                    "tag": str(elem.tag),
                    "field": elem.keyword or "(private)",
                    "technique": SUPPRESS,
                    "action": "deleted-sequence",
                })
                del ds[elem.tag]
                continue
            for item in elem.value:
                _walk(item, keystore, audit)
            continue

        if is_private:
            audit.append({
                "tag": str(elem.tag),
                "field": "(private)",
                "technique": SUPPRESS,
                "action": "deleted-private-tag",
            })
            del ds[elem.tag]
            continue

        mapping = TAG_MAPPING.get(tag_id)
        if not mapping:
            continue  # unmapped standard tag: left untouched

        technique = mapping["technique"]

        if technique == SUPPRESS:
            audit.append({
                "tag": str(elem.tag), "field": elem.keyword,
                "technique": SUPPRESS, "action": "deleted",
            })
            del ds[elem.tag]
            continue

        raw_value = elem.value
        value = str(raw_value).strip() if raw_value is not None else ""
        if should_skip_value(value):
            audit.append({
                "tag": str(elem.tag), "field": elem.keyword,
                "technique": technique, "action": "skipped-empty",
            })
            continue

        elem.value = apply_technique(technique, value, elem, keystore, tag_id)
        audit.append({
            "tag": str(elem.tag), "field": elem.keyword,
            "technique": technique,
            "action": "retained" if technique == "retain" else "transformed",
        })
