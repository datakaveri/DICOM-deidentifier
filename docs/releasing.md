# Releasing

Releases follow [Semantic Versioning](https://semver.org/) and the same
process as `datakaveri/k-anonymisation`.

| Change | Bump |
|---|---|
| Output format, tag mapping, CLI/env contract, image I/O layout, or anything that makes an existing deployment behave differently | MAJOR |
| New capability that existing deployments can ignore | MINOR |
| Bug or security fix with no contract change | PATCH |

When in doubt for a de-identification tool, treat a change in *what gets
removed or kept* as MAJOR: downstream data-sharing approvals are tied to it.

## Where the version lives

- `pyproject.toml` → `[project] version` (source of truth)
- `app/__init__.py` → `__version__` (must match; `tests/unit/test_version.py`
  fails otherwise)
- `CHANGELOG.md` → a `## [X.Y.Z] — YYYY-MM-DD` section

## Cutting a release

1. On a branch such as `chore/release-1-1-0`:
   - set the version in `pyproject.toml` and `app/__init__.py`;
   - in `CHANGELOG.md`, rename `## [Unreleased]` (or `## [X.Y.Z] — unreleased`)
     to `## [X.Y.Z] — YYYY-MM-DD` and add a fresh empty `## [Unreleased]`
     above it.
2. Open a PR, get it approved, squash-merge it into `main`.
3. Tag the merge commit on `main` and push the tag:

   ```bash
   git checkout main && git pull
   git tag -a vX.Y.Z -m "vX.Y.Z"
   git push origin vX.Y.Z
   ```

   The "Release tags" ruleset only accepts `vMAJOR.MINOR.PATCH` and makes the
   tag immutable: it cannot be moved or deleted. A mistake is fixed by a new
   patch release, never by re-tagging.
4. [`release.yml`](../.github/workflows/release.yml) then:
   - fails unless the tag equals `v<pyproject version>`, `app/__init__.py`
     agrees, and `CHANGELOG.md` has a dated section for it;
   - builds and pushes `ghcr.io/datakaveri/skald-dicom:X.Y.Z` and `:vX.Y.Z`;
   - creates the GitHub Release with that CHANGELOG section and the image
     digest.

## Deploying

Deployments pin `ghcr.io/datakaveri/skald-dicom:X.Y.Z`, or better its digest
from the release notes. There is deliberately no `latest` tag, so an image
never changes underneath a running deployment. Rolling back means deploying
the previous version's tag.

## Dependencies between releases

Edit ranges in `pyproject.toml`, then run `scripts/lock.sh` (keep current
pins where possible) or `scripts/lock.sh --upgrade`, and commit both locks.
CI's audit job must pass; an advisory can only be ignored with a written
reason next to its `--ignore-vuln` line in `.github/workflows/ci.yml`.
