"""
uids.py -- remap every instance UID in a dataset with one keyed HMAC.

Why: the per-tag policy hashes a UID with a per-column key, so the same UID
gets different results in StudyInstanceUID vs a ReferencedSOPInstanceUID, and
UIDs that no policy names (referenced series/instances in sequences, Frame of
Reference in a Structured Report, ...) pass through and still point at the
source study. With one HMAC key for the whole job, the same input UID maps to
the same output UID in every tag of every file, in any worker, and the key can
be supplied by the caller (single_file.Keys.uid_key) instead of being minted
in a keystore directory.

Opt-in: with no uid_key nothing here runs and the CLI behaves exactly as
before.

Kept as-is on purpose: UIDs that name a *kind* of thing rather than an
instance (SOP Class, Transfer Syntax, Implementation Class, coding schemes),
which carry no identity and which readers need to interpret the file.
"""

import hashlib
import hmac

from pydicom.multival import MultiValue

from .crypto import should_skip_value
from .tag_mapping import TAG_MAPPING, UID_VALUED_TAGS

# Keyword fragments of UI elements that are not instance identifiers.
_KEEP_FRAGMENTS = ("ClassUID", "TransferSyntax", "CodingSchemeUID", "ContextGroupExtensionCreatorUID")
_KEEP_EXACT = frozenset({"PrivateInformationCreatorUID"})


def hmac_uid(key: str, uid: str) -> str:
    """'2.25.<int>' derived from HMAC-SHA256(key, uid): deterministic, keyed,
    one-way, and a valid UID (PS3.5 Annex B)."""
    digest = hmac.new(key.encode("utf-8"), uid.strip().encode("utf-8"), hashlib.sha256).digest()
    return ("2.25." + str(int.from_bytes(digest[:16], "big")))[:64]


def is_instance_uid(elem) -> bool:
    kw = elem.keyword or ""
    return elem.VR == "UI" and kw not in _KEEP_EXACT and not any(f in kw for f in _KEEP_FRAGMENTS)


def _remap_value(key, value):
    if isinstance(value, (list, MultiValue)):
        return [hmac_uid(key, str(v)) if not should_skip_value(v) else v for v in value]
    return hmac_uid(key, str(value)) if not should_skip_value(value) else value


def remap_unlisted_uids(ds, key, job_config=None, audit=None) -> list:
    """Remaps every instance UID the tag policy does not itself handle, in
    place, sequences included: those in UID_VALUED_TAGS are left for the
    policy (which uses the same HMAC when the KeyStore carries a uid_key), as
    is any tag the user assigned an action (job config) or TAG_MAPPING names
    (fixed policy). Returns audit entries (tag names only, never a UID)."""
    audit = [] if audit is None else audit
    _walk(ds, key, job_config, audit)
    return audit


def sync_file_meta_uid(ds):
    """file_meta's Media Storage SOP Instance UID must equal the (remapped)
    SOP Instance UID; the original there would leak the source UID."""
    meta = getattr(ds, "file_meta", None)
    if meta is not None and "SOPInstanceUID" in ds and hasattr(meta, "MediaStorageSOPInstanceUID"):
        meta.MediaStorageSOPInstanceUID = ds.SOPInstanceUID


def _walk(ds, key, job_config, audit):
    for elem in ds:
        if elem.VR == "SQ":
            for item in elem.value:
                _walk(item, key, job_config, audit)
            continue
        tag_id = (elem.tag.group, elem.tag.element)
        if elem.tag.group % 2 == 1 or not is_instance_uid(elem) or tag_id in UID_VALUED_TAGS:
            continue
        if job_config is not None:
            if job_config.action_for(tag_id) is not None:
                continue
        elif tag_id in TAG_MAPPING:
            continue
        elem.value = _remap_value(key, elem.value)
        audit.append(
            {
                "tag": str(elem.tag),
                "field": elem.keyword,
                "technique": "uid_policy",
                "action": "remapped-hmac",
                "changed": True,
            }
        )
