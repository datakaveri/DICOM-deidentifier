# Contributing Guide

## Branching Model
We use **GitHub Flow**:
- Trunk: `main` (always deployable, protected)
- Feature branches branch off `main` and live at most 2 weeks.

### Branch Naming Convention
Follow `<type>/<short-kebab-description>`:
- `feat/` — New feature or algorithm enhancement
- `fix/` — Bug fix
- `chore/` — Dependency upgrades, config adjustments
- `docs/` — Documentation updates
- `refactor/` — Code restructuring without functional changes

**Prohibited**:
- Uppercase letters
- Underscores
- Version numbers in branch names
- Status words like `final`, `latest`, `updated`, `testing`

## Pull Request Process
1. Keep branches short-lived and focused on a single change.
2. Run linters and tests before submitting:
   ```bash
   ruff check app/ tests/
   pytest tests/
   ```
3. Ensure no real patient DICOM files or secrets are ever committed.
4. Squash-merge into `main` and delete the branch after merge.
