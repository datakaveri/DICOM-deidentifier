"""
keystore.py — persistent state for the reversible techniques (tokenise,
encrypt, encrypt_fpe), mirroring SKALD's on-disk stores
(token_vault.json, fpe_keys.json, symmetric_keys.json, fpe_encrypt_keys.json)
so that re-running the de-identifier on more files reuses the same
mappings/keys instead of silently diverging.
"""

import os

from .crypto import generate_random_key_hex, read_json_map_string, write_json_pretty


class KeyStore:
    """
    Loads (or creates) the token vault and the three per-technique key
    stores from `directory`, and persists them back via `save()`.
    """

    def __init__(self, directory: str):
        self.directory = directory
        os.makedirs(directory, exist_ok=True)

        self.token_vault_path = os.path.join(directory, "token_vault.json")
        self.fpe_keys_path = os.path.join(directory, "fpe_keys.json")
        self.symmetric_keys_path = os.path.join(directory, "symmetric_keys.json")
        self.fpe_encrypt_keys_path = os.path.join(directory, "fpe_encrypt_keys.json")

        self._token_vault = self._load_token_vault()
        self.fpe_keys = read_json_map_string(self.fpe_keys_path)
        self.symmetric_keys = read_json_map_string(self.symmetric_keys_path)
        self.fpe_encrypt_keys = read_json_map_string(self.fpe_encrypt_keys_path)

    def _load_token_vault(self) -> dict:
        if not os.path.exists(self.token_vault_path):
            return {}
        import json
        with open(self.token_vault_path, "r", encoding="utf-8") as f:
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

    def save(self) -> None:
        write_json_pretty(self.token_vault_path, self._token_vault)
        write_json_pretty(self.fpe_keys_path, self.fpe_keys)
        write_json_pretty(self.symmetric_keys_path, self.symmetric_keys)
        write_json_pretty(self.fpe_encrypt_keys_path, self.fpe_encrypt_keys)
