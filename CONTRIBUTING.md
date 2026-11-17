# Contributing Guide

## Branching Model
We use **GitHub Flow**:
- Trunk: `main` (always deployable, protected)
- Feature branches branch off `main` and live at most 2 weeks.

### Branch Naming Convention
Follow `<type>/<short-kebab-description>`; the "Branch naming" ruleset rejects
anything else (`^(feat|fix|chore|docs|refactor|test|perf|ci)/[a-z0-9]+([-._][a-z0-9]+)*$`):
- `feat/` — New feature or algorithm enhancement
- `fix/` — Bug fix
- `chore/` — Dependency upgrades, config adjustments
- `docs/` — Documentation updates
- `refactor/` — Code restructuring without functional changes
- `test/`, `perf/`, `ci/` — Tests, performance, CI configuration

**Prohibited**:
- Uppercase letters
- Underscores
- Version numbers in branch names
- Status words like `final`, `latest`, `updated`, `testing`

## Pull Request Process
1. Keep branches short-lived and focused on a single change.
2. Run linters and tests before submitting:
   ```bash
   ruff check . && black --check .
   pytest -q
   ```
3. Ensure no real patient DICOM files or secrets are ever committed.
4. Add a line under `## [Unreleased]` in `CHANGELOG.md` for any user-visible change.
5. Changing dependencies: edit the range in `pyproject.toml`, then run
   `scripts/lock.sh` (or `scripts/lock.sh --upgrade`) and commit both locks.
6. Squash-merge into `main` (the only merge method allowed) and delete the branch.

## Releases
See [`docs/releasing.md`](docs/releasing.md).
