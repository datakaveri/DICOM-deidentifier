# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/). The version in `pyproject.toml`
(mirrored in `app/__init__.py`) is the release version; tags are `v<version>`.
See [`docs/releasing.md`](docs/releasing.md).

## [Unreleased]

## [1.0.0] — unreleased

First tagged release. It replaces the original Flask tag-only service
(`server.py`, `deidentifier.py`, `detect_pii.py`) with the batch pipeline in
`app/`, which also redacts burned-in pixel text.

### Migrating from the Flask service

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

### Security

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

### Added

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

### Changed

- Dependencies declared once, in `pyproject.toml`, with upper bounds;
  `requirements.txt` is removed in favour of the hashed locks.
- Code formatted with black.
