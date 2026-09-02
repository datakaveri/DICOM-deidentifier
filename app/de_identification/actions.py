"""
actions.py -- the six k-anonymisation actions the SPIDEr UI can assign to a
DICOM tag, executed against one element value.

Nothing here is a new technique. These are the same six techniques, with the
same names and meanings, as in the tabular (CSV) pipeline; every one of them
dispatches to the SKALD primitive in crypto.py / masking.py that already
implements it. This module is the wire-name -> primitive adapter plus the
DICOM-specific concerns those primitives have no notion of: fitting a result
back into its element's VR, and keeping UI-VR elements syntactically valid.

  suppress           handled by the caller (deidentify.py): blank the
                     attribute, zero length, TAG RETAINED. Deleting it breaks
                     readers that expect a Type 2 attribute to be present.
  hashing_with_salt  crypto.hash_hex over a PER-RUN salt (RunSecrets below).
  encrypt            crypto.format_preserving_encrypt_general when
                     format_preserving is true, else crypto.pseudo_encrypt.
  masking            masking.apply_masking_value's "characters" step, with the
                     config's own masking_char and 1-based positions.
  tokenization       an opaque generated value: no relation to the input and
                     no stability across records.
  charcloak          crypto.randomize_preserving_class.

Contrast with operations.py, which serves the pre-contract fixed-policy path
(tag_mapping.TAG_MAPPING) and uses SKALD's *hashing_with_key* technique with a
persisted per-tag key. That path is unchanged; this one is separate on purpose,
because the two have opposite stability requirements across runs.
"""

import os
import uuid

from config import log
from .crypto import (
    generate_random_salt_hex, hash_hex, pseudo_encrypt,
    format_preserving_encrypt_general, randomize_preserving_class,
)
from .job_config import (
    SUPPRESS, HASHING_WITH_SALT, ENCRYPT, MASKING, TOKENIZATION, CHARCLOAK,
)
from .masking import MaskingConfigLite, apply_masking_value
from .operations import VR_MAX_LENGTH, tag_column_key
from .tag_mapping import UID_VALUED_TAGS


class RunSecrets:
    """Per-run key material for the config-driven path.

    The salt behind `hashing_with_salt` is minted once per run from a CSPRNG,
    held only in memory, and never written anywhere -- not to the keystore, not
    to an audit, not to the output volume. That is the whole point of the
    technique as distinct from hashing_with_key:

      - within a run every occurrence of a value hashes identically, so records
        still join across the files of a batch;
      - across runs the salt is different, so the same PatientID produces an
        unrelated hash and there is no cross-run linkage to exploit.

    One instance per batch, passed to every file. Losing it at process exit is
    the intended behaviour, not an oversight.
    """

    def __init__(self, salt=None):
        # generate_random_salt_hex is SKALD's generator for exactly this
        # technique (os.urandom-backed), kept distinct from the
        # generate_random_key_hex used for the persisted hashing_with_key keys.
        self._salt = salt or generate_random_salt_hex()

    @property
    def salt(self):
        return self._salt


def _fit_to_vr(value, vr):
    """Truncates `value` to the max character length of `vr` (DICOM PS3.5
    Table 6.2-1), so a generated value still fits its declared VR."""
    max_len = VR_MAX_LENGTH.get(vr)
    return value[:max_len] if max_len else value


def _as_uid(digest_hex):
    """Reshapes a hex digest into the standard UUID-derived UID form
    '2.25.<int>' (PS3.5 Annex B). A UI-VR element must hold a dotted numeric
    string, so a raw hex digest cannot be written into one."""
    derived = uuid.UUID(bytes=bytes.fromhex(digest_hex)[:16])
    return ("2.25." + str(derived.int))[:64]


def _is_uid_valued(elem, tag_id):
    return elem.VR == "UI" or tag_id in UID_VALUED_TAGS


def hashing_with_salt(value, elem, tag_id, secrets):
    """Salted one-way hash: SHA-256(salt + value), with the run's salt.

    Single-pass salted hashing, matching SKALD's hashing_with_salt -- not the
    nested keyed double-hash of hashing_with_key, whose key is persisted and
    would defeat the per-run requirement.
    """
    digest = hash_hex(secrets.salt + value)
    if _is_uid_valued(elem, tag_id):
        # A hex digest is not a valid UID. The user assigned a hash to a UID
        # anyway and the UI warned them; honour it, but keep the element
        # syntactically valid rather than writing a file no reader will open.
        return _as_uid(digest)
    return _fit_to_vr(digest, elem.VR)


def tokenize(value, elem, tag_id):
    """An opaque generated value: no relation to the input, no stability
    across records. Deliberately NOT the keystore's sequential vault -- that
    is reversible, and its reverse map on the output volume would be a
    re-identification key sitting beside the data it de-identifies."""
    if _is_uid_valued(elem, tag_id):
        return ("2.25." + str(uuid.uuid4().int))[:64]
    token = "TK-" + os.urandom(16).hex().upper()
    return _fit_to_vr(token, elem.VR)


def encrypt(value, elem, tag_id, keystore, format_preserving):
    """Encrypts `value`.

    format_preserving=True keeps the input's shape and character classes
    (crypto.format_preserving_encrypt_general) -- the only variant safe to
    write into a typed element, since the result still fits the VR.

    format_preserving=False produces the 'ENC$<hex>' pseudo-encryption, which
    is longer than its input and will overrun most VRs' nominal length limit.
    It is written whole -- truncating ciphertext would destroy it -- and the
    overrun is logged.
    """
    column = tag_column_key(elem)

    if format_preserving:
        key = keystore.get_or_create_key(keystore.fpe_encrypt_keys, column)
        return format_preserving_encrypt_general(value, key, column)

    key = keystore.get_or_create_key(keystore.symmetric_keys, column)
    ciphertext = pseudo_encrypt(value, key, column)

    max_len = VR_MAX_LENGTH.get(elem.VR)
    if max_len and len(ciphertext) > max_len:
        log.warning(
            f"  [{elem.keyword}] encrypt with format_preserving=false produced "
            f"{len(ciphertext)} chars, over the {elem.VR} limit of {max_len}. "
            f"Written whole (the element is widened); set format_preserving "
            f"true for a value that fits."
        )
    return ciphertext


def mask(value, tag_action):
    """Replaces the characters at `characters_to_mask` (1-based positions)
    with `masking_char`, leaving the rest.

    This is also how a date is coarsened -- there is no generalisation action.
    DICOM dates are YYYYMMDD, so positions 5-8 are exactly the month and day
    and '19780412' becomes '1978****'. No date-awareness is needed or wanted
    here: it is an ordinary positional mask.
    """
    cfg = MaskingConfigLite(
        column=tag_action.keyword,
        masking_char=tag_action.masking_char,
        characters_to_mask=tag_action.characters_to_mask,
        apply_order=["characters"],
    )
    return apply_masking_value(value, cfg, randomize_preserving_class)


def apply_action(tag_action, value, elem, tag_id, keystore, secrets):
    """Applies `tag_action` to one scalar `value`, returning the new value.

    `suppress` never reaches here -- blanking an element is a dataset-level
    operation the caller performs, not a value transform.
    """
    action = tag_action.action

    if action == HASHING_WITH_SALT:
        return hashing_with_salt(value, elem, tag_id, secrets)

    if action == ENCRYPT:
        return encrypt(value, elem, tag_id, keystore, tag_action.format_preserving)

    if action == MASKING:
        return mask(value, tag_action)

    if action == TOKENIZATION:
        return tokenize(value, elem, tag_id)

    if action == CHARCLOAK:
        return randomize_preserving_class(value)

    if action == SUPPRESS:
        raise AssertionError("suppress is handled by the caller, not apply_action")

    # Unreachable: job_config rejects unsupported actions at parse time.
    raise AssertionError(f"unhandled action {action!r}")
