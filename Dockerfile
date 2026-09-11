# ==============================================================================
# DICOM De-Identification Pipeline — Production Container
# ==============================================================================
#
# Processes radiology DICOM files (.dcm) at two levels in a single unified pass:
#   1. Burned-In Pixel Text Redaction:
#      - Primary OCR: PaddleOCR (PP-OCRv4)
#      - Multi-Model PHI Classification: Stanford De-ID, Biomedical NER, GLiNER,
#        Microsoft Presidio, Regex PII patterns, and Clinical Allowlist
#      - Character Stroke & Shadow Segmentation: Top-Hat + Drop-shadow capture
#      - Unified Inpainting: Navier-Stokes neighbor reconstruction
#      - Verification: Two-tier fast morphological stroke gate with OCR fallback
#   2. DICOM Header Tag De-Identification:
#      - Per-tag tokenization, hashing, AES-CBC format-preserving encryption,
#        masking, and UID regeneration per DICOM PS 3.15 Profile (secured.json).
#
# CPU-only by default — no CUDA runtime required. To build for GPU, swap the
# CPU PyTorch/Paddle wheels for CUDA builds.
# All model weights are baked in during build time for air-gapped/offline execution.
# ==============================================================================

FROM python:3.10-slim

# System libraries required by opencv-python-headless, PaddleOCR, and PyTorch
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CPU-only PyTorch and torchvision explicitly to avoid massive CUDA wheels
RUN pip install --no-cache-dir torch==2.2.2 torchvision==0.17.2 \
        --index-url https://download.pytorch.org/whl/cpu

# Install Python requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir \
        https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl

# Copy application source code
COPY app/ .

# Pre-download OCR (PaddleOCR), NLP, and Transformer model weights at BUILD time.
# Ensures the container can run in 100% air-gapped / offline healthcare environments.
RUN python -c "from engines import check_gpu_available, initialize_engines; initialize_engines(use_gpu=check_gpu_available())"

# Create standard runtime directories
RUN mkdir -p /app/data /app/config /app/output

ENV SKALD_DATA_DIR=/app/data \
    SKALD_CONFIG_DIR=/app/config \
    SKALD_OUTPUT_DIR=/app/output \
    PYTHONUNBUFFERED=1

# Default entry point runs the batch pipeline across all DICOMs in /app/data
CMD ["python", "main.py"]
