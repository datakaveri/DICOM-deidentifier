# DICOM De-Identifier — Ubuntu / WSL Setup & Demo Guide

## Quick Start (One Command)

If you have this project folder, just open Ubuntu terminal and run:

```bash
cd "/mnt/c/Users/YourUsername/path/to/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset"
bash setup_ubuntu.sh
```

This single command does **everything automatically** — installs Python 3.12, creates a virtual environment, installs PaddlePaddle + PaddleOCR + all AI models, and runs the pipeline.

---

## Step-by-Step Manual Setup (For Demo)

### Step 1: Open Ubuntu Terminal

**On Windows (WSL):**
- Press `Windows Key`, type **Ubuntu**, click to open
- OR: Open PowerShell and type `wsl`

**On native Ubuntu/Linux:**
- Open any terminal

### Step 2: Navigate to the Project Folder

```bash
# If project is on Windows Desktop (via WSL):
cd "/mnt/c/Users/Shashank/OneDrive/Desktop/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset (1)/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset"

# If project is on a native Linux path:
cd /path/to/DICOM-deidentifier-Integrate-Telangana-DICOM-Dataset
```

### Step 3: Install System Dependencies

```bash
sudo apt update
sudo apt install -y libgl1 libglib2.0-0
```

### Step 4: Install Python 3.12 (if not already installed)

```bash
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.12 python3.12-venv python3.12-dev
```

### Step 5: Create and Activate Virtual Environment

> **IMPORTANT**: Create the venv inside your Linux home directory (`~/venv_deid`), NOT on the Windows mount (`/mnt/c/...`). This avoids NTFS symlink errors.

```bash
python3.12 -m venv ~/venv_deid
source ~/venv_deid/bin/activate
pip install --upgrade pip
```

### Step 6: Install OCR Engine (PaddlePaddle + PaddleOCR)

```bash
pip install paddlepaddle==2.6.2 paddleocr==2.9.1
```

> **Version Note**: Use `paddlepaddle==2.6.2` + `paddleocr==2.9.1` (the stable combo). Newer versions (PaddleOCR 3.7+) have PIR engine bugs on CPU.

### Step 7: Install AI/NLP Dependencies

```bash
pip install pydicom opencv-python-headless numpy cryptography
pip install presidio-analyzer transformers gliner torch spacy
pip install sentencepiece protobuf safetensors
python -m spacy download en_core_web_lg
```

### Step 8: Run the Pipeline

```bash
python app/main.py
```

You should see output like:
```
Found 6 file(s) to process.
Initializing OCR and AI engines (GPU=False)...
  [OK] PaddleOCR (Native) initialized as PRIMARY OCR
  [OK] Presidio NLP Analyzer initialized
  [OK] Stanford De-ID model initialized
  [OK] Biomedical NER model initialized
  [OK] GLiNER-BioMed model initialized
...
Batch complete: 6 file(s) processed.
  File_000002_3.dcm              PASSED
  ...
```

---

## Re-Running After Setup

Once the setup is done, future runs only need 2 commands:

```bash
source ~/venv_deid/bin/activate
python app/main.py
```

The first run downloads AI model weights (~1.5GB total). All subsequent runs start instantly.

---

## Output Files

Results are saved in `app/output/<filename>/`:

| File | Description |
|------|-------------|
| `data.json` | Original DICOM tag values (backup) |
| `phi_tags.json` | Identified PHI-bearing tags |
| `before_deidentification.dcm` | Pixel-redacted DICOM (tags still original) |
| `after_deidentification.dcm` | Fully de-identified DICOM (pixels + tags) |
| `bbox_regions.png` | Visual audit with red bounding boxes |
| `pipeline_audit.json` | Complete pipeline execution audit log |

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `pip install` fails with "externally managed environment" | You forgot to activate the venv: `source ~/venv_deid/bin/activate` |
| `No OCR engine available!` | Install OCR: `pip install paddlepaddle==2.6.2 paddleocr==2.9.1` |
| `python3.12 not found` | Install it: `sudo apt install python3.12 python3.12-venv` |
| NTFS symlink error creating venv | Create venv in Linux home: `python3.12 -m venv ~/venv_deid` |
