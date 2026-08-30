#!/bin/bash
set -e

echo "============================================================"
echo "  DICOM De-Identifier — Ubuntu/WSL Setup & Run Script"
echo "============================================================"
echo ""

# ── Step 1: System Packages ──────────────────────────────────────────────────
echo "=== Step 1: Installing System Packages ==="
sudo apt update
sudo apt install -y libgl1 libglib2.0-0 git

# ── Step 2: Python 3.12 Virtual Environment ──────────────────────────────────
echo ""
echo "=== Step 2: Creating Python 3.12 Virtual Environment ==="

if ! command -v python3.12 &> /dev/null; then
    echo "python3.12 not found. Installing from deadsnakes PPA..."
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt update
    sudo apt install -y python3.12 python3.12-venv python3.12-dev
fi

# Create venv inside Linux home (avoids NTFS/symlink issues on /mnt/c/)
VENV_PATH="$HOME/venv_deid"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating virtual environment at '$VENV_PATH'..."
    python3.12 -m venv "$VENV_PATH"
fi

echo "Activating virtual environment..."
source "$VENV_PATH/bin/activate"
pip install --upgrade pip

# ── Step 3: Install PaddlePaddle + PaddleOCR (Stable Combo) ─────────────────
echo ""
echo "=== Step 3: Installing PaddlePaddle 2.6.2 + PaddleOCR 2.9.1 ==="
pip install paddlepaddle==2.6.2 paddleocr==2.9.1

# ── Step 4: Install NLP, AI, and Project Dependencies ────────────────────────
echo ""
echo "=== Step 4: Installing NLP and Project Dependencies ==="
pip install pydicom opencv-python-headless numpy cryptography
pip install presidio-analyzer transformers gliner torch spacy
pip install sentencepiece protobuf safetensors
python -m spacy download en_core_web_lg

# ── Step 5: Run the Pipeline ─────────────────────────────────────────────────
echo ""
echo "=== Step 5: Running DICOM De-Identification Pipeline ==="
echo "============================================================"
python app/main.py

echo ""
echo "============================================================"
echo "  DONE! Check the app/output/ folder for results."
echo "============================================================"
