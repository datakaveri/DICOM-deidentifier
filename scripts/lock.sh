#!/usr/bin/env bash
# Regenerate requirements.lock (runtime) and requirements-dev.lock (tooling).
#
# Resolution runs inside the same pinned base image the Dockerfile uses, so
# the lock matches the image's Python and platform (CPython 3.12, linux
# x86_64) whatever the host is. torch comes from the PyTorch CPU index, which
# keeps multi-GB CUDA wheels out of the image.
#
# Usage: scripts/lock.sh [--upgrade]   (needs Docker and network access)
#   Without --upgrade, versions already in the lock are kept where they still
#   satisfy pyproject.toml; with it, everything moves to the newest allowed.
set -euo pipefail

UPGRADE="${1:-}"

cd "$(dirname "$0")/.."

BASE_IMAGE="python:3.12.16-slim@sha256:5e2bdb1b0d542d5039cffb8181c65930093949194a65c365dfb1b0fe13a4fa5b"
UV_VERSION="0.8.22"

docker run --rm -v "$PWD":/src -w /src "$BASE_IMAGE" sh -euc "
  pip install --quiet --disable-pip-version-check uv==${UV_VERSION}
  common='--python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes --no-header --quiet ${UPGRADE}'
  uv pip compile \$common \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    --emit-index-url \
    pyproject.toml requirements.in -o requirements.lock
  uv pip compile \$common --extra dev --constraint requirements.lock \
    --index-url https://pypi.org/simple \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    --index-strategy unsafe-best-match \
    --emit-index-url \
    pyproject.toml requirements.in -o requirements-dev.lock
  chown $(id -u):$(id -g) requirements.lock requirements-dev.lock
"
echo "Wrote requirements.lock and requirements-dev.lock"
