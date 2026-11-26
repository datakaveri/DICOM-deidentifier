"""
single_file.py -- anonymise ONE DICOM file held in memory.

    from single_file import Engines, FileResult, process_file

    engines = Engines.load(config)               # once per worker / Ray actor
    result = process_file(data, "a/b.dcm", config, engines)   # per file

Why this exists: the CLI (main.py) loads the models once and walks a
directory. A Ray Data actor needs the same pipeline as a function of
(bytes, config) so the models stay loaded across thousands of files. This
module is that function. It composes the very stages the CLI uses
(pipeline.redact_dataset_pixels for the pixels, pipeline._deidentify_tags
for the header), so there is one implementation, not two.

process_file performs NO filesystem writes and NO network calls of its own
(NER, when used, goes through the injected client), holds no state of its own
and is deterministic given config + keys. A per-file failure comes back as
FileResult(ok=False, error_code=...); it never raises and never exits.

Keys: the fixed policy hashes with per-tag keys and the job-config path with a
per-run salt. The CLI mints and persists those; here they come in through
`keys` (a Keys) or config["keys"], and are generated in memory when absent
(so a lone call still works, but two actors then disagree: pass keys for any
job that spans more than one process).

`audit` is PHI-free by construction: counts, tag names, techniques, region
geometry and timings. Never an original value or recognised text.

CLI parity (verified on synthetic fixtures, see the commit message): the
pixels and tags are byte-identical to what main.py writes for the same file.
Two deliberate differences: a file that is not parseable DICOM fails with
NOT_DICOM even under the fixed policy (the CLI there reads it with force=True
and writes it out), and nothing is checkpointed to disk.
"""

import copy
import hashlib
import io
import json
import time
from dataclasses import dataclass, field
from typing import Optional

import pydicom

import file_errors as codes
from config import log
from file_errors import FileError


@dataclass
class FileResult:
    ok: bool
    output: Optional[bytes]  # anonymised file bytes; None when ok is False
    output_rel_path: Optional[str]  # == rel_path today; lets a caller rename
    audit: dict = field(default_factory=dict)  # PHI-free
    error_code: Optional[str] = None  # stable code, see file_errors.py


class Keys:
    """Key material for one job: a KeyStore (hash / encrypt / FPE keys and the
    tokenise vault), the per-run salt of hashing_with_salt, and (optionally)
    the HMAC key that remaps UIDs. Build it once per worker and pass it to
    every process_file call of the job."""

    def __init__(self, keystore=None, secrets=None, uid_key=None):
        from de_identification.keystore import KeyStore

        self.keystore = keystore if keystore is not None else KeyStore.in_memory()
        self._secrets = secrets
        self._salt = None
        # The HMAC key that remaps every UID the same way across the job
        # (de_identification/uids.py). None keeps the per-tag hash policy.
        self.uid_key = uid_key
        self.keystore.uid_key = uid_key

    @property
    def secrets(self):
        """The hashing_with_salt salt. Minted on first use, so the fixed
        policy (which has none) draws no randomness it did not draw before."""
        from de_identification.actions import RunSecrets

        if self._secrets is None:
            self._secrets = RunSecrets(self._salt)
        return self._secrets

    @classmethod
    def from_config(cls, config):
        """From config["keys"]: {"salt", "uid_key", "hash_keys",
        "symmetric_keys", "fpe_encrypt_keys", "fpe_keys"}; every part is
        optional and generated when absent."""
        from de_identification.keystore import KeyStore

        raw = (config or {}).get("keys") or {}
        store = KeyStore.in_memory(
            {
                k: raw[k]
                for k in ("hash_keys", "symmetric_keys", "fpe_encrypt_keys", "fpe_keys", "token_vault")
                if k in raw
            }
        )
        keys = cls(store, None, raw.get("uid_key"))
        keys._salt = raw.get("salt")
        return keys


class Engines:
    """The loaded models and clients, built once per worker. Passing the
    pieces yourself (tests, an actor with its own NER client) skips load().

    ner: an object with spans(texts) -> list[list[dict]] (span =
    {start, end, label[, confidence]}); None means the enclave's unix-socket
    client (shared_ner.SocketNerClient), which reports "unavailable" when the
    socket is absent, and then free-text tags are blanked, never kept.
    """

    def __init__(
        self,
        paddle_ocr=None,
        analyzer=None,
        deid_model=None,
        medical_ner=None,
        gliner_model=None,
        ner=None,
    ):
        self.paddle_ocr = paddle_ocr
        self.analyzer = analyzer
        self.deid_model = deid_model
        self.medical_ner = medical_ner
        self.gliner_model = gliner_model
        self.ner = ner
        self._job_configs = {}
        self._keys = {}

    @classmethod
    def load(cls, config, model_dir=None, ner=None, use_gpu=None, cpu_threads=None):
        """Loads PaddleOCR, Presidio, Stanford De-ID, the biomedical NER and
        GLiNER (lazily) via engines.initialize_engines, exactly as the CLI.
        `model_dir`, if given, becomes the Hugging Face cache (HF_HOME) unless
        the environment already sets one; PaddleOCR 2.x keeps its own cache
        under ~/.paddleocr and has no directory option, so bake those models
        into the image or HOME. `config` is accepted so a future config key
        can steer loading; nothing in it changes which models load today."""
        import os

        if model_dir:
            os.environ.setdefault("HF_HOME", os.path.join(model_dir, "huggingface"))
        from engines import check_gpu_available, initialize_engines

        if use_gpu is None:
            use_gpu = check_gpu_available()
        loaded = initialize_engines(use_gpu=use_gpu, cpu_threads=cpu_threads, lazy_gliner=True)
        return cls(*loaded, ner=ner)

    # -- per-config caches: parsing logs a block per call, and tokenise vaults
    #    must persist across the files one worker sees ------------------------
    @staticmethod
    def _config_key(config):
        public = {k: v for k, v in (config or {}).items() if k != "keys"}
        return hashlib.sha256(json.dumps(public, sort_keys=True, default=str).encode()).hexdigest()

    def job_config_for(self, config):
        from de_identification.job_config import parse_job_config

        key = self._config_key(config)
        if key not in self._job_configs:
            self._job_configs[key] = parse_job_config(config or {})
        return self._job_configs[key]

    def keys_for(self, config):
        key = self._config_key(config)
        if key not in self._keys:
            self._keys[key] = Keys.from_config(config)
        return self._keys[key]


def _phi_free_audit(raw, job_config, timings):
    """The caller-facing audit: nothing from `raw` that could hold a value."""
    tags = raw.get("deidentified_tags", [])
    by_technique = {}
    for entry in tags:
        if entry.get("changed", True) is False or entry.get("action") == "skipped-empty":
            continue
        by_technique[entry["technique"]] = by_technique.get(entry["technique"], 0) + 1
    return {
        "tag_policy": "job_config" if job_config is not None else "fixed",
        "modality": raw.get("modality"),
        "image_size": raw.get("image_size"),
        "verification_status": raw.get("verification_status"),
        "redacted_region_count": len(raw.get("redacted_regions", [])),
        # geometry only; the recognised text is dropped, as in the published CLI audit
        "redacted_region_boxes": [r.get("bbox") for r in raw.get("redacted_regions", [])],
        "source_transfer_syntax": raw.get("source_transfer_syntax"),
        "output_transfer_syntax": raw.get("output_transfer_syntax"),
        "overlay_planes_removed": raw.get("overlay_planes_removed", 0),
        "tags": tags,
        "tags_by_technique": by_technique,
        "timings_s": timings,
    }


def _fail(rel_path, code, raw, job_config, started, exc=None):
    audit = _phi_free_audit(raw, job_config, {"total": round(time.time() - started, 3)})
    audit["error_type"] = type(exc).__name__ if exc is not None else None
    return FileResult(False, None, None, audit, code)


def process_file(data, rel_path, config, engines, keys=None):
    """Anonymises `data` (the bytes of one DICOM file); see the module doc."""
    # Imported here: pipeline pulls in OCR/classification modules and the
    # engine libraries, which a caller importing only FileResult must not pay for.
    from de_identification.job_config import JobConfigError
    from de_identification.uids import remap_unlisted_uids, sync_file_meta_uid
    from phi_tags import original_tag_values, snapshot_original_tags
    from pipeline import (
        PRESERVE_TAGS,
        _deidentify_tags,
        _restore_preserved,
        deidentified_tag_filter,
        new_audit,
        read_dicom,
        redact_dataset_pixels,
    )

    started = time.time()
    raw = new_audit(rel_path.rsplit("/", 1)[-1])
    job_config = None

    try:
        job_config = engines.job_config_for(config)
    except JobConfigError as e:
        return _fail(rel_path, codes.INVALID_CONFIG, raw, None, started, e)
    keys = keys or engines.keys_for(config)

    try:
        # ── Read ─────────────────────────────────────────────────────────────
        try:
            ds = read_dicom(io.BytesIO(data), job_config)
            if "SOPClassUID" not in ds:
                raise ValueError("no SOPClassUID")
        except Exception as e:  # noqa: BLE001 -- pydicom raises many types
            return _fail(rel_path, codes.NOT_DICOM, raw, job_config, started, e)

        stored_values = original_tag_values(
            snapshot_original_tags(ds), keep=deidentified_tag_filter(job_config)
        )

        # ── Pixels (OCR, classify, redact, verify) ───────────────────────────
        t0 = time.time()
        ds, _ = redact_dataset_pixels(
            ds,
            raw["file"],
            raw,
            engines.paddle_ocr,
            engines.analyzer,
            stored_values,
            deid_model=engines.deid_model,
            medical_ner=engines.medical_ner,
            gliner_model=engines.gliner_model,
            job_config=job_config,
            ner=engines.ner,
        )
        t_pixels = time.time() - t0
    except Exception as e:  # noqa: BLE001
        log.error(f"  [ERROR] {raw['file']}: pixel stage failed ({type(e).__name__})")
        raw["verification_status"] = "ERROR"
        return _fail(rel_path, getattr(e, "code", codes.PIXEL_STAGE_FAILED), raw, job_config, started, e)

    try:
        # ── Tags ─────────────────────────────────────────────────────────────
        # Round-trip through bytes, as the CLI does through its checkpoint
        # file: the tag pass must see the dataset as written, or the output
        # would not be byte-identical to the CLI's.
        t0 = time.time()
        checkpoint = io.BytesIO()
        ds.save_as(checkpoint, write_like_original=False)
        checkpoint.seek(0)
        ds = pydicom.dcmread(checkpoint, force=True)
        uid_audit = []
        if keys.uid_key:
            # UIDs no policy names (referenced series/instances, ...) first;
            # the policy pass then maps the named ones with the same HMAC.
            remap_unlisted_uids(ds, keys.uid_key, job_config, uid_audit)
        # Tags that must survive exactly as they are (SKALD_PRESERVE_TAGS),
        # as in pipeline.deidentify_file_tags.
        kept = {kw: copy.deepcopy(ds[kw]) for kw in PRESERVE_TAGS if kw in ds}
        raw["deidentified_tags"] = _deidentify_tags(
            ds, keys.keystore, keys.secrets if job_config is not None else None, job_config, ner=engines.ner
        ) + uid_audit
        if kept:
            _restore_preserved(ds, kept, raw["deidentified_tags"])
        if keys.uid_key:
            sync_file_meta_uid(ds)
        out = io.BytesIO()
        ds.save_as(out, write_like_original=False)
        raw["output_transfer_syntax"] = str(ds.file_meta.TransferSyntaxUID)
        t_tags = time.time() - t0
    except Exception as e:  # noqa: BLE001
        log.error(f"  [ERROR] {raw['file']}: tag stage failed ({type(e).__name__})")
        raw["verification_status"] = "ERROR"
        return _fail(rel_path, codes.TAG_STAGE_FAILED, raw, job_config, started, e)

    timings = {
        "pixels": round(t_pixels, 3),
        "tags": round(t_tags, 3),
        "total": round(time.time() - started, 3),
    }
    return FileResult(True, out.getvalue(), rel_path, _phi_free_audit(raw, job_config, timings), None)
