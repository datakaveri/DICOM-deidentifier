# Changelog

All notable changes to this project will be documented in this file.
The format is based on [Keep a Changelog](https://keepachangelog.com/), and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-09-25

### Added
- Standard `.github/` workflows: CI (`ci.yml`), Security scanning (`codeql.yml`), Release (`release.yml`).
- Dependabot configuration for pip, docker, and actions (`dependabot.yml`).
- Apache 2.0 LICENSE file.
- Repository hygiene scaffolding: `SECURITY.md`, `CONTRIBUTING.md`, `pyproject.toml`, `.env.example`, `.pre-commit-config.yaml`.
- Docker non-root user configuration and healthcheck.
- Package version tracking in `app/__init__.py`.

### Changed
- Dependencies pinned with upper bounds in `pyproject.toml`.
