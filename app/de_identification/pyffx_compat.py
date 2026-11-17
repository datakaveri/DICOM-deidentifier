"""
pyffx_compat.py — pure-Python port of SKALD's Rust `pyffx_compat` module
(k-anonymisation/SKALD/src/pipeline/pyffx_compat.rs).

Implements a format-preserving Feistel cipher (HMAC-SHA1 round function,
pyffx-compatible) over an arbitrary alphabet. Used by crypto.py for the
"encrypt (FPE)" technique, which keeps a value's length/character-set
while making it unrecoverable without the key.

Ported line-for-line from the Rust implementation; verified against its
five unit-test vectors (see scratchpad test during development) before
being wired into the rest of this package.
"""

import hmac
import hashlib
import math

DEFAULT_ROUNDS = 10
SHA1_DIGEST_SIZE = 20


class AlphabetMap:
    """Maps characters of a fixed alphabet to/from their positional index."""

    def __init__(self, alphabet: str):
        chars = list(alphabet)
        if len(chars) < 2:
            raise ValueError("alphabet must contain at least 2 characters")
        index_map = {}
        for i, c in enumerate(chars):
            if c in index_map:
                raise ValueError(f"duplicate character in alphabet: {c}")
            index_map[c] = i
        self.alphabet = chars
        self.index_map = index_map

    def radix(self) -> int:
        return len(self.alphabet)

    def encode(self, text: str):
        try:
            return [self.index_map[c] for c in text]
        except KeyError as e:
            raise ValueError(f"non-alphabet character: {e.args[0]}")

    def decode(self, values) -> str:
        out = []
        for v in values:
            if v >= len(self.alphabet):
                raise ValueError(f"digit out of range for alphabet: {v}")
            out.append(self.alphabet[v])
        return "".join(out)


def _pyffx_chars_per_hash(radix: int) -> int:
    # int(digest_size * log(256, radix)) — matches the Rust/pyffx formula
    return int(SHA1_DIGEST_SIZE * (math.log(256) / math.log(radix)))


def _u32_le_bytes(v: int) -> bytes:
    return (v & 0xFFFFFFFF).to_bytes(4, "little")


def _divmod_be_bytes_in_place(barr: bytearray, divisor: int) -> int:
    """Long-divides the big-endian integer held in `barr` by `divisor` in
    place, returning the remainder (one output "digit" per call)."""
    rem = 0
    for i in range(len(barr)):
        cur = (rem << 8) | barr[i]
        barr[i] = cur // divisor
        rem = cur % divisor
    return rem


def _fill_round_digits(key: bytes, radix: int, round_index: int, right, out_len: int):
    key_buf = bytearray()
    key_buf += _u32_le_bytes(round_index)
    for d in right:
        key_buf += _u32_le_bytes(d)

    chars_per_hash = max(1, _pyffx_chars_per_hash(radix))
    out_digits = []
    counter = 0

    while len(out_digits) < out_len:
        mac = hmac.new(key, bytes(key_buf) + _u32_le_bytes(counter), hashlib.sha1)
        digest = mac.digest()
        d = bytearray(digest)  # working copy; the raw digest is reused below

        for _ in range(chars_per_hash):
            if len(out_digits) >= out_len:
                break
            rem = _divmod_be_bytes_in_place(d, radix)
            out_digits.append(rem)

        key_buf = bytearray(digest)  # next round's key material is the RAW digest
        counter = (counter + 1) & 0xFFFFFFFF

    return out_digits


def fpe_encrypt_checked(key: bytes, text: str, alphabet: str) -> str:
    amap = AlphabetMap(alphabet)
    v = amap.encode(text)
    if not v:
        return ""
    radix = amap.radix()
    split = len(v) // 2
    a = v[:split]
    b = v[split:]

    for i in range(DEFAULT_ROUNDS):
        round_digits = _fill_round_digits(key, radix, i, b, len(a))
        c = [(a[j] + round_digits[j]) % radix for j in range(len(a))]
        a, b = b, c

    return amap.decode(a + b)


def fpe_decrypt_checked(key: bytes, text: str, alphabet: str) -> str:
    amap = AlphabetMap(alphabet)
    v = amap.encode(text)
    if not v:
        return ""
    radix = amap.radix()
    split = len(v) // 2
    a = v[:split]
    b = v[split:]

    for i in reversed(range(DEFAULT_ROUNDS)):
        old_b = a
        c = b
        round_digits = _fill_round_digits(key, radix, i, old_b, len(c))
        old_a = [(c[j] + radix - (round_digits[j] % radix)) % radix for j in range(len(c))]
        a, b = old_a, old_b

    return amap.decode(a + b)


def fpe_encrypt(key: bytes, text: str, alphabet: str) -> str:
    """Encrypts `text`, falling back to the original value on any alphabet
    mismatch (mirrors the Rust `unwrap_or_else` fallback)."""
    try:
        return fpe_encrypt_checked(key, text, alphabet)
    except ValueError:
        return text


def fpe_decrypt(key: bytes, text: str, alphabet: str) -> str:
    try:
        return fpe_decrypt_checked(key, text, alphabet)
    except ValueError:
        return text
