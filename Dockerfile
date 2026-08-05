# SKALD-DICOM — DICOM burned-in text + tag de-identification pipeline.
#
# Runs as a batch job: reads every *.dcm file under /data, applies pixel
# (OCR-based burned-in text redaction) and tag-level (hash/tokenise/encrypt)
# de-identification, and writes results + audit logs under /output.
#
# CPU-only by default (see the torch install step below) — no CUDA/nvidia
# runtime required. To build a GPU variant, swap the CPU torch wheel below
# for a CUDA build.
#
# OCR uses EasyOCR only (PaddleOCR was dropped: PaddlePaddle and PyTorch
# each bundle conflicting native allocators/OpenMP runtimes, and running
# both in the same process reliably corrupted the heap once real model
# work started — see ocr_detect.py, which already tolerates a single OCR
# engine).

FROM python:3.10-slim

# System libraries required by opencv-python-headless / easyocr at
# import/runtime (image codecs, OpenMP, X11 stubs some wheels still link).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only PyTorch first (easyocr depends on torch/torchvision; installing
# the CPU wheel explicitly avoids pulling the much larger default CUDA build).
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
# Install the spaCy model as a direct wheel URL, pinned to a version
# compatible with spacy==3.7.4 above. `spacy download` shells out to a
# GitHub compatibility-check API that has proven unreliable in CI/Docker
# builds (it can return a malformed release URL); a pinned wheel avoids
# that lookup entirely.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

COPY app/ .

# Pre-download OCR/NLP model weights at BUILD time so the running container
# never needs outbound network access (required for air-gapped/TEE
# deployments). Needs network access during `docker build` only.
RUN python -c "from engines import check_gpu_available, initialize_engines; initialize_engines(use_gpu=check_gpu_available())"

RUN mkdir -p /data /app/config /output

ENV SKALD_DATA_DIR=/data \
    SKALD_CONFIG_DIR=/app/config \
    SKALD_OUTPUT_DIR=/output \
    PYTHONUNBUFFERED=1

CMD ["python", "-m", "de_identification.run"]
