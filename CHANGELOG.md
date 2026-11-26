# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The version in `pyproject.toml`
(mirrored in `app/__init__.py`) is the release version; tags are `v<version>`.
See [`docs/releasing.md`](docs/releasing.md).

## [Unreleased]

### Fixed

- **12- and 14-bit images were rewritten as 16-bit.** Writing the redacted
  pixels back set `BitsStored`/`HighBit` from the array type (16/15) instead
  of keeping the original (for example 12/11). Pixel values were unchanged,
  but viewers take the value range, and the MONOCHROME1 inversion, from
  `BitsStored`, so contrast and brightness shifted. The original values are
  now kept whenever the redacted pixels still fit. This affected 2,192 of
  2,953 PACS chest X-rays in the 2026-11-18 batch.
- **`after_preview.png` of MONOCHROME1 images was inverted**, and every
  preview used its own percentile stretch. Previews are now rendered as a
  viewer shows the file: rescale, the file's window, and MONOCHROME1
  inversion.

### Changed

- **Tag-first pixel classification.** Burned-in text that equals the value of
  a tag the tag pass de-identifies is redacted directly, and only the rest
  goes to the NER models. "De-identified" means the job config's
  `tag_actions` or, under the fixed policy, non-retained `tag_mapping.py`
  entries and dates; private tags count in both. Before, the text was
  matched against every tag (Manufacturer, StationName, ...) and only after
  NER, so burned-in copies of non-PHI values were blacked out. On the sample
  set, the output is pixel-identical to 2.0.0.
- With a scratch directory, each file's intermediates are deleted when the
  file finishes, not at the end of the batch.

### Added

- `SKALD_OUTPUT_LAYOUT=mirror` and `SKALD_REPORT_DIR`: the output keeps the
  input's relative paths and file names, and holds nothing else.
- `SKALD_PRESERVE_TAGS`: tags the tag pass leaves unchanged. The meta
  header follows a preserved `SOPInstanceUID`.
- `SKALD_DISCARD_INTERMEDIATES`, `SKALD_RESUME` and
  `SKALD_KEYSTORE_SAVE_EVERY` (default 25) for long batches.
- Benchmark logs: `benchmark_files.jsonl` and `benchmark_summary.json`, plus
  a `PROGRESS` line with ETA per file.
- `scripts/batch/inventory.py`: Excel inventory of every DICOM's tags and the
  chest selection. `scripts/vm/` contains VM setup, upload, batch runner and
  monitor scripts.
- The shared Anamika NER engine (2026-11-23), reached through a Unix socket
  the SPIDEr enclave mounts at `/run/anamika-ner/ner.sock`; the enclave sends
  the text to an attested NER TEE. See `app/shared_ner.py`.
  - `free_text_anonymization` tag action: the personal data inside a free-text
    tag is replaced (`"XR CHEST PA - [PERSON]"`) and the rest kept. If the
    engine cannot be asked, the tag is blanked, never written out unchecked.
  - `dicom_deidentify.shared_ner` (optional): `pixel_text` adds the engine's
    hits on burned-in text to what the classifier redacts (only ever more);
    `minimum_confidence` (0.75) and `model_labels` (`["PERSON"]`) tune both.
- Without the socket, or with neither setting used, a run is unchanged.

## [2.0.0] — 2026-11-17

The first release built by this repository's release pipeline. The images
`v1.0.0` to `v1.3` on GHCR were pushed by hand from other branches, and
`:latest` pointed at `v1.1`. This release moves `:latest`.

**Before deploying:** the image runs as uid 65532 by default. The SPIDEr
enclaves stage inputs root-owned (mode `0600`), so their compose file must
set `user: "0:0"`. datakaveri/Docker-Compose#2 does this.

**Behaviour that changes against v1.1:**
- Compressed (JPEG) DICOMs are de-identified instead of failing.
- Output is always uncompressed, which makes it about 1.5× larger for JPEG
  inputs.
- Colour images are rejected.
- Any failed file makes the container exit 1.
- Burned-in text detection is this repository's pipeline. It can differ
  from v1.1 on markers: for example, an "L A6" laterality marker that v1.1
  blacked out is now kept.

### Added

- **SPIDEr job config.** This brings over the per-tag de-identification
  that the deployed `:latest` (v1.1, built from the old
  `Integrate-Telangana-DICOM-Dataset` branch) already ran. A `config.json`
  with a `dicom_deidentify` block applies the user's `tag_actions` from the
  UI. The actions are `suppress`, `hashing_with_salt`, `encrypt`, `masking`,
  `tokenization` and `charcloak`. Unlisted tags are kept; UIDs are still
  remapped and private tags removed. Under a config:
  - burned-in text is filled solid black;
  - verification re-checks after escalation and can report `FAILED`;
  - an unparsable file is an error;
  - the still-identified intermediates and the run's keys go to a scratch
    directory under `/tmp` that is deleted at the end.
  An invalid config exits 1 before any file is read. Without a config, the
  fixed policy runs. Tag results were compared with v1.1 on the same input
  and config: the same tags, the same techniques, and different values only
  for per-run salted hashes and UID remaps.
- `manifest.json` in the output directory, with per-file status and
  `tags_by_technique`. The enclave attaches it to the job status.
- `tag_audit.json` per file.
- Files with the same name in different input folders get separate output
  directories (`name`, `name_2`) instead of overwriting each other.
- `app/runtime.py` sizes the run to the machine it lands on: CPU affinity,
  cgroup v1/v2 CPU and memory limits, and available memory. This makes the
  image safe on TEE VMs of unknown shape.
- Thread caps for torch, Paddle (`cpu_threads`, previously 10 per process),
  OpenCV, OpenBLAS and MKL.
- On large machines, the pixel stage runs in worker processes forked after
  the models load, so the workers share the weights. Tag de-identification
  stays in one process, in input order, so tokens and keys are minted
  exactly as in a serial run.
- The batch finishes serially if the worker pool cannot start, a worker is
  OOM-killed, or a file exceeds `SKALD_FILE_TIMEOUT_S`.
- New environment variables: `SKALD_WORKERS`, `SKALD_THREADS_PER_WORKER`,
  `SKALD_FILE_TIMEOUT_S`, `SKALD_MODEL_MEMORY_MB` and
  `SKALD_WORKER_MEMORY_MB`.

### Security

- `pipeline_audit.json` no longer carries the recognised burned-in text,
  only region boxes.
- `bbox_regions.png` shows the PHI before redaction, so it is only written
  with `SKALD_SAVE_BBOX_PREVIEW=1`.
- Verification no longer logs the residual text it found.
- The fixed policy now also masks or removes date tags that
  `tag_mapping.py` does not list (VR DA/DT, or keyword containing "date").

### Fixed

- **Compressed DICOMs (JPEG Lossless and others) are now de-identified
  instead of silently failing.** Pixel data is decoded right after reading,
  and the output is written uncompressed (Explicit VR Little Endian,
  `1.2.840.10008.1.2.1`). This applies to the PACS X-ray exports, which are
  JPEG Lossless SV1. Before, decoding failed, 0 tags were de-identified,
  "Final anonymized DICOM saved" was printed with nothing saved, and the
  container exited 0. Decoding is lossless: a JPEG Lossless input gives
  exactly the same output pixels as the same image uncompressed.
- pydicom 2.x `decompress()` leaves the JPEG bytes in Pixel Data. The
  dataset's Pixel Data, transfer syntax, VR and length encoding are now
  always rewritten together, so no output can say "uncompressed" while
  holding JPEG bytes, or the reverse.
- **Failures are loud.** A failed file prints its real error and leaves
  nothing at `after_deidentification.dcm`. The container exits 1 if any file
  failed, or if `/app/data` has no `.dcm` files.
- A file with image attributes but no readable Pixel Data (truncated or
  corrupt) is an error. Before, it was saved as a metadata-only output.
- `python -m de_identification.run` returns the same exit status as
  `main.py`.

### Changed

- Colour images (`SamplesPerPixel` > 1) are rejected with an error. The
  redaction stages read a 3-D array as frames, so a colour image was
  processed as hundreds of one-row frames and its burned-in text never
  really checked.
- `/app/keys` is no longer a `VOLUME`. Run through `docker compose` without
  a keys mount, as the enclaves do, every run left an anonymous volume
  holding `secured.json` behind after `docker compose down`.
- Output files from compressed inputs are larger: about 1.5× for the 8-bit
  PACS X-rays (9.0 MB in, 14.8 MB out for a 4248 × 3480 image).
- **About 11× faster end to end with identical output.** On the 6-file
  sample set, a batch went from 236 s to 20 s, and per-file processing from
  ~31 s to ~1.4 s. Pixels, redacted regions, verification status, tags and
  the key store were all compared against the previous version and match.
  See `docs/performance.md`.
- `classify_phi` runs each model only when its answer can change the
  decision. Before, every OCR string ran through all four NER models;
  GLiNER alone was ~80% of the runtime.
- `classify_phi` caches decisions per text within a run (in memory only).
- In a single-process run, GLiNER loads only if a string needs it. Peak
  memory drops from ~4.2 GB to ~1.2 GB.
- The unused shape-based Stage 2 pass is no longer computed. Full-image
  OCR never read its regions.
- PNG previews are encoded with fast compression; the pixels are the same.

### Since the Flask service (drafted as 1.0.0, never released)

First tagged release. It replaces the original Flask tag-only service
(`server.py`, `deidentifier.py`, `detect_pii.py`) with the batch pipeline in
`app/`, which also redacts burned-in pixel text.

#### Migrating from the Flask service

1. **No HTTP API.** The `/inspect_DICOM` and `/process_DICOM` endpoints are
   gone. The image is a one-shot batch job: mount DICOMs at `/app/data`, read
   results from `/app/output`.
2. **Key material moved out of the output directory.** `secured.json` (hash
   keys and the reverse token vault) is written to `SKALD_KEYS_DIR`
   (`/app/keys` in the image; the working directory locally, as before).
   Mount a persistent `keys/` volume and reuse it across batches so
   pseudonyms stay consistent. Never ship it.
3. **Container runs as uid 65532.** Host directories mounted at `/app/output`
   and `/app/keys` must be writable by that uid
   (`chown 65532:65532 output keys`).
4. **Python 3.12.** The image, the locks and CI use CPython 3.12
   (3.10 reached end of life in October 2026).
5. **Image name and tags.** Releases publish
   `ghcr.io/datakaveri/skald-dicom:<version>` and `:v<version>`, and move
   `:latest` to the highest release. Deployments run `:latest`; pin a version
   or digest instead when a run must be reproducible.

#### Security

- **Dependency advisories closed** (all seven open Dependabot alerts):
  - `Flask-Cors` CVE-2024-6221, CVE-2024-6839, CVE-2024-6844, CVE-2024-6866
    and `flask` CVE-2026-27205: Flask and flask-cors are removed; the service
    no longer has an HTTP layer.
  - `pydicom` CVE-2026-32711 (DICOMDIR/FileSet path traversal): 2.4.5.
  - `pytest` CVE-2025-71176: 9.x.
- **Transitive advisories closed**: Pillow 13 (40+ 2026 advisories),
  transformers 5 (4.x advisories with no 4.x fix), filelock, soupsieve and
  black upgraded. CI now audits both locks with `pip-audit` on every PR.
- **Known and accepted** (documented in `ci.yml`, tracked for removal):
  `paddlepaddle` 2.6.2 CVE-2024-0815 and CVE-2024-0817 are fixed only in
  paddlepaddle 3.x, which needs the PaddleOCR 3 migration. Neither affected
  function is called by `app/`, and the runtime image is offline.
- **Hardened image**: multi-stage build, base pinned by digest, non-root user,
  root-owned read-only code/venv/models, `keys/` volume at mode 0700,
  `HF_HUB_OFFLINE`/`TRANSFORMERS_OFFLINE` at run time, an allowlist
  `.dockerignore` (no `.dcm` or `secured.json` can enter a layer).
- **Supply chain**: hashed locks (`--require-hashes`), every GitHub Action
  pinned to a full commit SHA, least-privilege workflow permissions.
- **Sample data removed**: `sample_data/retinal_phi.dcm` and
  `sample_data/synthetic_phi.dcm` are no longer in the tree. Both are
  synthetic.

#### Added

- Burned-in pixel text redaction: PaddleOCR detection, multi-model PHI
  classification (Presidio, Stanford De-ID, biomedical NER, GLiNER, regex,
  clinical allowlist), stroke segmentation with Navier-Stokes inpainting,
  two-tier verification.
- Tag de-identification per `app/de_identification/tag_mapping.py`: keyed
  hashing, tokenisation, date masking, suppression, UID regeneration, with a
  batch-wide keystore.
- `SKALD_KEYS_DIR` to place the keystore.
- Release automation: `release.yml` verifies the tag against the package
  version and this file, publishes the GHCR image, and creates the GitHub
  Release from this file's section.
- `scripts/lock.sh` regenerates `requirements.lock` and
  `requirements-dev.lock` inside the pinned base image.
- Repository standards: CI (lint, format, tests, dependency audit, Dockerfile
  check), CodeQL, Dependabot (pip, docker, actions), issue/PR templates,
  CODEOWNERS, `LICENSE` (Apache-2.0), `SECURITY.md`, `CONTRIBUTING.md`,
  `.env.example`, pre-commit hooks.

#### Changed

- Dependencies declared once, in `pyproject.toml`, with upper bounds;
  `requirements.txt` is removed in favour of the hashed locks.
- Code formatted with black.
