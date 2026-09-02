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
  - Standard tags with no entry in TAG_MAPPING are left untouched, EXCEPT
    any date-shaped tag (VR "DA"/"DT", or keyword containing "date") -- the
    DICOM standard defines ~100 of these (StudyArrivalDate,
    DateOfLastCalibration, ScheduledProcedureStepStartDate, ...) and
    TAG_MAPPING only lists the handful that show up routinely, so an
    unlisted one must still be masked/suppressed rather than silently
    leaking an exact date. See _fallback_date_mapping() below.

Unlike combine.py (which deliberately never mutates tag values, relying
only on private-tag stripping + pixel redaction), this module performs the
tag-VALUE-level anonymization described in the PDF, and is intended to run
as an additional metadata pass alongside — not instead of — the pixel
pipeline.
"""

from pydicom.multival import MultiValue

from .tag_mapping import TAG_MAPPING, SUPPRESS, MASK
from .operations import apply_technique
from .crypto import should_skip_value


def _fallback_date_mapping(elem) -> dict:
    """
    Catch-all for date-shaped tags with no TAG_MAPPING entry: VR "DA"
    (date) and "DT" (datetime) are masked the same way as the explicitly
    listed date tags (retain year, blank month/day) since they're always
    an 8+-char YYYYMMDD[...] string. A tag whose keyword merely contains
    "date" but isn't DA/DT VR (e.g. PatientBirthDateInAlternativeCalendar,
    a free-text LO field) isn't a fixed format masking can safely target,
    so it's suppressed outright instead of guessing at a partial redaction.
    Returns None for anything that isn't date-shaped.
    """
    if elem.VR in ("DA", "DT"):
        return {"technique": MASK}
    if "date" in (elem.keyword or "").lower():
        return {"technique": SUPPRESS}
    return None


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

        mapping = TAG_MAPPING.get(tag_id) or _fallback_date_mapping(elem)
        if not mapping:
            continue  # unmapped, non-date standard tag: left untouched

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


# =============================================================================
# Config-driven path (SPIDEr "skald_dicom" contract)
# =============================================================================
# Everything above is the pre-contract fixed policy, driven by TAG_MAPPING and
# reached when the job config carries no `dicom_deidentify` block. Everything
# below is driven by that block's `tag_actions` instead, and differs from it in
# two ways that matter:
#
#   - `suppress` BLANKS the attribute (zero length, tag retained) rather than
#     deleting it. Deleting breaks readers that expect a Type 2 attribute to be
#     present; the fixed-policy path above deletes, and stays that way.
#   - a tag absent from `tag_actions` is written through UNCHANGED
#     (default_action "keep"), with the two exceptions documented on
#     deidentify_dataset_from_config.

from .job_config import SUPPRESS as ACTION_SUPPRESS
from .actions import apply_action
from .tag_mapping import HASH, UID_VALUED_TAGS


def _blank_value(elem):
    """The zero-length value for `elem`'s VR: '' for text/UI VRs, None for the
    binary and numeric ones, where an empty string is not a valid value."""
    if elem.VR in ("US", "SS", "UL", "SL", "FL", "FD", "OB", "OW", "OF", "AT", "UN"):
        return None
    return ""


def _suppress_element(ds, elem, audit, keyword):
    """Blanks `elem` in place -- zero length, tag retained."""
    if elem.VR == "SQ":
        elem.value = []
        note = "blanked-sequence"
    else:
        elem.value = _blank_value(elem)
        note = "blanked"
    audit.append({
        "tag": str(elem.tag), "field": keyword,
        "technique": ACTION_SUPPRESS, "action": note, "changed": True,
    })


def _apply_to_value(tag_action, raw_value, elem, tag_id, keystore, secrets):
    """Applies `tag_action` to `raw_value`, mapping over the items of a
    multi-valued element (VM > 1) rather than stringifying the whole list --
    OtherPatientNames and PatientTelephoneNumbers routinely carry several
    values, and hashing their repr() would produce one nonsense value where
    there were three real ones."""
    if isinstance(raw_value, (list, MultiValue)):
        out = []
        for item in raw_value:
            text = str(item).strip()
            if should_skip_value(text):
                out.append(item)
            else:
                out.append(apply_action(tag_action, text, elem, tag_id, keystore, secrets))
        return out, True

    text = str(raw_value).strip() if raw_value is not None else ""
    if should_skip_value(text):
        return raw_value, False
    return apply_action(tag_action, text, elem, tag_id, keystore, secrets), True


def deidentify_dataset_from_config(ds, keystore, secrets, job_config) -> list:
    """Applies `job_config.tag_actions` to `ds`, mutated in place (including
    nested sequence items). Returns the same audit-trail shape as
    deidentify_dataset -- {"tag", "field", "technique", "action", "changed"}
    dicts, with `technique` set to the wire action identifier so the manifest's
    `tags_by_technique` keys line up with the config's -- and no original or
    new values, so the audit itself carries no PHI.

    default_action is "keep", so a tag with no `tag_actions` entry is written
    through unchanged. In particular StudyDescription and SeriesDescription
    pass through byte-identical unless named: with no free-text redaction the
    only outcomes available are keeping the field whole or losing it whole, and
    deleting takes "CHEST PA" along with "Jane Doe".

    Two categories are not "unchanged", both to avoid making this path weaker
    than the fixed policy it replaces (contract section 7):

      - UIDs not named in tag_actions get the existing UID policy -- the
        deterministic hash-to-UID remap from TAG_MAPPING, which keeps a study's
        objects linked to each other while breaking the link to the source.
        None of the six actions can act on a UID correctly, which is exactly
        why the UI does not send one.
      - Private tags (odd group number) are stripped, as they always have been.
        They are invisible to the UI, so a user cannot name them, and vendor
        private blocks routinely carry a copy of the patient demographics.
    """
    audit = []
    _walk_config(ds, keystore, secrets, job_config, audit)
    _sync_file_meta(ds, audit)
    return audit


def _sync_file_meta(ds, audit) -> None:
    """Keeps file_meta.MediaStorageSOPInstanceUID equal to the (possibly
    remapped) SOPInstanceUID. The two are required to match; leaving the old
    value in the meta header both invalidates the file and leaks the original
    UID that was just remapped."""
    file_meta = getattr(ds, "file_meta", None)
    if file_meta is None or "SOPInstanceUID" not in ds:
        return
    if getattr(file_meta, "MediaStorageSOPInstanceUID", None) == ds.SOPInstanceUID:
        return
    file_meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID
    audit.append({
        "tag": "(0002,0003)", "field": "MediaStorageSOPInstanceUID",
        "technique": "uid_policy", "action": "synced-to-sop-instance-uid",
        "changed": True,
    })


def _walk_config(ds, keystore, secrets, job_config, audit) -> None:
    for elem in list(ds):  # list(...) snapshot: safe to delete while iterating
        tag_id = (elem.tag.group, elem.tag.element)
        is_private = (elem.tag.group % 2) == 1

        if is_private:
            # Recorded under its own technique, NOT "suppress": the manifest's
            # tags_by_technique is what the UI shows back to the user, and
            # folding a few dozen stripped vendor tags into their two chosen
            # suppressions makes that number meaningless.
            audit.append({
                "tag": str(elem.tag), "field": "(private)",
                "technique": "private_tag_policy", "action": "deleted-private-tag",
                "changed": True,
            })
            del ds[elem.tag]
            continue

        tag_action = job_config.action_for(tag_id)

        if elem.VR == "SQ":
            if tag_action is not None and tag_action.action == ACTION_SUPPRESS:
                _suppress_element(ds, elem, audit, tag_action.keyword)
                continue
            # No scalar to transform: recurse so a nested PatientID is matched
            # by its own tag and treated exactly like a top-level one.
            for item in elem.value:
                _walk_config(item, keystore, secrets, job_config, audit)
            continue

        if tag_action is None:
            _apply_unlisted_policy(ds, elem, tag_id, keystore, audit)
            continue

        if tag_action.action == ACTION_SUPPRESS:
            _suppress_element(ds, elem, audit, tag_action.keyword)
            continue

        new_value, changed = _apply_to_value(
            tag_action, elem.value, elem, tag_id, keystore, secrets
        )
        if changed:
            elem.value = new_value
        audit.append({
            "tag": str(elem.tag), "field": tag_action.keyword,
            "technique": tag_action.action,
            "action": "transformed" if changed else "skipped-empty",
            "changed": changed,
        })


def _apply_unlisted_policy(ds, elem, tag_id, keystore, audit) -> None:
    """A tag with no tag_actions entry. Written through unchanged, except a
    UID, which gets the existing deterministic remap (see
    deidentify_dataset_from_config)."""
    if tag_id not in UID_VALUED_TAGS:
        return  # default_action "keep": untouched, byte-identical.

    raw_value = elem.value
    value = str(raw_value).strip() if raw_value is not None else ""
    if should_skip_value(value):
        return

    elem.value = apply_technique(HASH, value, elem, keystore, tag_id)
    audit.append({
        "tag": str(elem.tag), "field": elem.keyword,
        "technique": "uid_policy", "action": "remapped", "changed": True,
    })
