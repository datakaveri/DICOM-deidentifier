"""
keystore.py — persistent state for the reversible techniques (tokenise,
encrypt, encrypt_fpe), consolidated into a single file (secured.json) so one
KeyStore instance can be shared across every DICOM file in a batch: the same
PatientID/AccessionNumber/etc. gets the same token or key no matter which
file in the batch it appears in, instead of a fresh, unrelated one per file.
"""

import os

from .crypto import generate_random_key_hex, write_json_pretty


class KeyStore:
    """
    Loads (or creates) all key material from a single JSON file at `path`,
    and persists it back via `save()`. Create one instance per batch run and
    pass it to every file processed in that batch — call `save()` once after
    the whole batch finishes, not per file.
    """

    def __init__(self, path: str):
        self.path = path
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)

        data = self._load()
        self._token_vault = data.get("token_vault", {})
        self.fpe_keys = data.get("fpe_keys", {})
        self.symmetric_keys = data.get("symmetric_keys", {})
        self.fpe_encrypt_keys = data.get("fpe_encrypt_keys", {})
        self.hash_keys = data.get("hash_keys", {})

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {}
        import json
        with open(self.path, "r", encoding="utf-8") as f:
            return json.load(f)

    def tokenise(self, column: str, value: str, prefix: str = "TK-", digits: int = 6) -> str:
        """
        Returns the existing token for (column, value) if already minted,
        otherwise mints the next sequential token and records both
        directions in the vault (so a controlled reversal is possible).
        """
        col_vault = self._token_vault.setdefault(column, {"forward": {}, "reverse": {}})
        forward = col_vault["forward"]
        if value in forward:
            return forward[value]

        next_id = len(col_vault["reverse"]) + 1
        token = f"{prefix}{next_id:0{digits}d}"
        forward[value] = token
        col_vault["reverse"][token] = value
        return token

    def get_or_create_key(self, store: dict, column: str) -> str:
        """Returns the persisted key for `column` in `store`, minting one if absent."""
        if column not in store:
            store[column] = generate_random_key_hex()
        return store[column]

    def get_or_create_hash_key(self, column: str) -> str:
        """
        Returns the persisted key for `column` (a DICOM tag/field), minting
        one via generate_random_key_hex() if absent. Matches SKALD's
        hashing_with_key (nested_hash_hex) exactly, including its key
        generator -- NOT generate_random_salt_hex(), which SKALD reserves for
        the separate, non-persisted hashing_with_salt technique. One key per
        column, reused for every row/file that column appears in for as long
        as this keystore file persists — so e.g. every PatientID across the
        whole batch (and future batches reusing this same secured.json)
        hashes identically.
        """
        if column not in self.hash_keys:
            self.hash_keys[column] = generate_random_key_hex()
        return self.hash_keys[column]

    def save(self) -> None:
        """Writes all key material back to the single secured.json file, atomically."""
        data = {
            "token_vault": self._token_vault,
            "fpe_keys": self.fpe_keys,
            "symmetric_keys": self.symmetric_keys,
            "fpe_encrypt_keys": self.fpe_encrypt_keys,
            "hash_keys": self.hash_keys,
        }
        write_json_pretty(self.path, data)
