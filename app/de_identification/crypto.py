"""
crypto.py — Python port of SKALD's Rust `crypto` module
(k-anonymisation/SKALD/src/pipeline/preprocess/crypto.rs).

Provides the cryptographic primitives behind the anonymization techniques
from the DICOM tag mapping (hash, tokenise, encrypt, charcloak):
  - Random key/salt generation (os.urandom, not deterministic).
  - SHA-256 hashing (salted and unsalted).
  - HMAC-SHA256 key derivation (per-column/context sub-keys).
  - Format-preserving encryption (general alnum, PAN, digits) via pyffx_compat.
  - Pseudo-encryption (XOR keystream from HMAC blocks, "ENC$" prefix).
  - Class-preserving randomization (the "charcloak" technique).
  - JSON key-store I/O helpers used by operations.py / deidentify.py.
"""

import os
import json
import hmac
import hashlib
import secrets

from .pyffx_compat import fpe_encrypt

# ── Key / salt generation ────────────────────────────────────────────────────

def generate_random_key_hex(nbytes: int = 16) -> str:
    """Random 32-hex-char (16-byte) key, analogous to the Rust /dev/urandom read."""
    return os.urandom(nbytes).hex()


def generate_random_salt_hex(nbytes: int = 32) -> str:
    """Random 64-hex-char (32-byte) salt for salted SHA-256 hashing."""
    return os.urandom(nbytes).hex()


# ── Hashing ──────────────────────────────────────────────────────────────────

def hash_hex(value: str) -> str:
    """SHA-256 hex digest (matches Python hashlib.sha256(...).hexdigest())."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def hash_hex_keyed(key: str, message: str) -> str:
    """
    Keyed double-hash, matching SKALD's nested_hash_hex (hashing_with_key
    technique) exactly: hash(key + hash(key + message)) --
    SHA-256(key + SHA-256(key + message)). The inner hash binds key and
    message together first; the outer hash re-applies the key on top of that
    result, rather than a single hash_hex(key + message) pass. `key` should
    come from a CSPRNG (see keystore.get_or_create_hash_key / generate_random_key_hex,
    os.urandom-backed, matching SKALD's /dev/urandom key generation).
    """
    inner = hash_hex(key + message)
    return hash_hex(key + inner)


# ── Value guards ─────────────────────────────────────────────────────────────

def should_skip_value(value) -> bool:
    """True when `value` should be left unchanged (empty or 'nan')."""
    if value is None:
        return True
    v = str(value).strip()
    return v == "" or v.lower() == "nan"


# ── JSON key-store helpers ───────────────────────────────────────────────────

def read_json_map_string(path: str) -> dict:
    """Reads a JSON file as a flat str->str map; {} if the file is absent."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if isinstance(v, str)}


def write_json_pretty(path: str, value) -> None:
    """Writes `value` as pretty JSON via a temp-file + rename (atomic write)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)
    os.replace(tmp, path)


# ── HMAC key derivation ───────────────────────────────────────────────────────

def derive_key(master_key: str, context: str) -> bytes:
    """
    Derives a 16-byte sub-key from `master_key` and a domain `context` string
    via HMAC-SHA256 (first 16 bytes of the digest) — the shared building
    block for FPE and pseudo-encryption so each (column, class, length)
    triple gets a unique but deterministic key.
    """
    mac = hmac.new(master_key.encode("utf-8"), context.encode("utf-8"), hashlib.sha256)
    return mac.digest()[:16]


# ── Format-preserving encryption ─────────────────────────────────────────────

def format_preserving_encrypt_general(value: str, master_key: str, column: str) -> str:
    """
    Encrypts `value` in a format-preserving way, treating each run of
    identical character class (upper letter / lower letter / digit) as a
    separate FPE segment. Non-alphanumeric characters pass through
    unchanged so the value's overall structure is preserved.
    """
    out = []
    i = 0
    n = len(value)
    while i < n:
        ch = value[i]
        if ch.isascii() and ch.isupper():
            class_name, alphabet = "upper", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        elif ch.isascii() and ch.islower():
            class_name, alphabet = "lower", "abcdefghijklmnopqrstuvwxyz"
        elif ch.isascii() and ch.isdigit():
            class_name, alphabet = "digit", "0123456789"
        else:
            out.append(ch)
            i += 1
            continue

        j = i + 1
        while j < n:
            c = value[j]
            same = (
                (class_name == "upper" and c.isascii() and c.isupper())
                or (class_name == "lower" and c.isascii() and c.islower())
                or (class_name == "digit" and c.isascii() and c.isdigit())
            )
            if not same:
                break
            j += 1

        segment = value[i:j]
        ctx = f"{column}:{class_name}:{len(segment)}"
        key = derive_key(master_key, ctx)
        out.append(fpe_encrypt(key, segment, alphabet))
        i = j

    return "".join(out)


def pseudo_encrypt(value: str, key: str, column: str) -> str:
    """
    Encrypts `value` with a deterministic XOR keystream built from
    successive HMAC-SHA256 blocks (never repeats regardless of value
    length), hex-encoded with an "ENC$" prefix.
    """
    plaintext = value.encode("utf-8")
    keystream = bytearray()
    block_idx = 0
    while len(keystream) < len(plaintext):
        ctx = f"{column}:{block_idx}"
        keystream += derive_key(key, ctx)
        block_idx += 1

    out = ["ENC$"]
    for p, k in zip(plaintext, keystream):
        out.append(f"{p ^ k:02x}")
    return "".join(out)


def fpe_pan_encrypt(value: str, master_key: str) -> str:
    """
    FPE for 10-char Indian PAN numbers ([A-Z]{5}[0-9]{4}[A-Z]). Encrypts the
    three structural parts independently so the PAN shape is preserved.
    Returns `value` unchanged if it doesn't match the expected format.
    """
    if (
        len(value) != 10
        or not value[:5].isascii() or not value[:5].isupper() or not value[:5].isalpha()
        or not value[5:9].isascii() or not value[5:9].isdigit()
        or not value[9].isascii() or not value[9].isupper() or not value[9].isalpha()
    ):
        return value

    letters_key = derive_key(master_key, "pan_letters")
    digits_key = derive_key(master_key, "pan_digits")
    suffix_key = derive_key(master_key, "pan_suffix")

    e1 = fpe_encrypt(letters_key, value[:5], "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    e2 = fpe_encrypt(digits_key, value[5:9], "0123456789")
    e3 = fpe_encrypt(suffix_key, value[9:10], "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    return e1 + e2 + e3


def fpe_digits_encrypt(value: str, master_key: str) -> str:
    """
    FPE for digit-only strings (phone numbers, Aadhaar, etc.). Returns
    `value` unchanged if empty or containing any non-digit character.
    """
    if not value or not value.isdigit():
        return value
    key = derive_key(master_key, f"digits_len_{len(value)}")
    return fpe_encrypt(key, value, "0123456789")


# ── Class-preserving randomization ("charcloak") ─────────────────────────────

_DIGITS = "0123456789"
_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_LOWER = "abcdefghijklmnopqrstuvwxyz"


def randomize_preserving_class(value: str) -> str:
    """
    Replaces each character with a cryptographically random character of the
    same class (digit / upper letter / lower letter); other characters pass
    through unchanged. Uses `secrets` for true randomness — output differs
    on every call for the same input (matches Python `secrets.choice`).
    """
    out = []
    for c in value:
        if c.isascii() and c.isdigit():
            out.append(secrets.choice(_DIGITS))
        elif c.isascii() and c.isupper():
            out.append(secrets.choice(_UPPER))
        elif c.isascii() and c.islower():
            out.append(secrets.choice(_LOWER))
        else:
            out.append(c)
    return "".join(out)
