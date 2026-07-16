# 🚀 Deployment & Run Guide

This guide describes how to run the de-identification pipeline locally and inside cloud GPU notebooks (Kaggle and Google Colab).

---

## 💻 Local Setup & Execution

### 1. Prerequisite Installations
Ensure Python 3.8–3.12 is installed, then install dependencies:
```bash
# Core medical and visual processing libraries
pip install pydicom easyocr presidio-analyzer opencv-python numpy matplotlib

# Install PaddlePaddle (CPU or GPU version depending on your OS)
# For CPU:
pip install paddlepaddle
# For GPU (CUDA must be configured):
pip install paddlepaddle-gpu

# Install PaddleOCR
pip install paddleocr
```

### 2. Execution CLI Arguments
You can run the script directly from your terminal:
```bash
python dicom_anonymizer_pipeline.py -i <INPUT_PATH> -o <OUTPUT_PATH> [OPTIONS]
```

* **Arguments**:
  * `-i`, `--input`: Path to input DICOM file OR directory of DICOM files.
  * `-o`, `--output`: Path to save anonymized DICOM file OR output directory.
  * `-a`, `--audit`: Custom path for the output audit JSON log.
  * `--gpu`: Force enable GPU acceleration.
  * `--no-gpu`: Force disable GPU acceleration.

* **Examples**:
  ```bash
  # Anonymize a single scan:
  python dicom_anonymizer_pipeline.py -i scan.dcm -o anonymized_scan.dcm --gpu

  # Anonymize an entire directory of chest X-rays:
  python dicom_anonymizer_pipeline.py -i ./raw_folder -o ./output_folder --gpu
  ```

---

## ⚡ Kaggle Setup & Running (GPU Enabled)

Running on Kaggle allows you to utilize free T4 GPUs to accelerate OCR processing by **5x to 10x**.

### Step 1: Upload Your Data & The Script
1. Create a new **Notebook** in Kaggle.
2. In the right panel under **Datasets**, click **Upload** or **Add Input** and upload:
   * Your raw DICOM scans (e.g. as a `.zip` file).
   * The python script [dicom_anonymizer_pipeline.py](file:///c:/Users/Shashank/OneDrive/Desktop/haryana%20problem%20statement/dicom_anonymizer_pipeline.py).
   * *Alternatively, you can just import the pre-configured [dicom_anonymizer_kaggle.ipynb](file:///c:/Users/Shashank/OneDrive/Desktop/haryana%20problem%20statement/dicom_anonymizer_kaggle.ipynb) notebook into Kaggle.*

### Step 2: Enable GPU Accelerator
1. Expand **Notebook options** in the right-hand panel of the Kaggle editor.
2. Under **Accelerator**, select **GPU T4 x2** or **GPU P100**. 
*Note: If the options are greyed out, make sure you have verified your phone number on your Kaggle Account Settings page (`https://www.kaggle.com/settings`) and refreshed the editor.*

### Step 3: Install Dependencies
Create a cell, paste this code, and run it:
```python
# Install libraries
!pip install -q pydicom easyocr presidio-analyzer opencv-python numpy matplotlib

# Install GPU-compatible PaddlePaddle
import torch
use_gpu = torch.cuda.is_available()
if use_gpu:
    !pip install -q paddlepaddle-gpu
else:
    !pip install -q paddlepaddle
!pip install -q paddleocr
```

### Step 4: Run the Script (Auto-Detection Mode)
If you paste the script [dicom_anonymizer_pipeline.py](file:///c:/Users/Shashank/OneDrive/Desktop/haryana%20problem%20statement/dicom_anonymizer_pipeline.py) into a cell and run it directly, it will use **Auto-Detection**:
* It scans `/kaggle/input` recursively to find the directory containing your `.dcm` files.
* It sets the input path to that directory automatically (e.g. `/kaggle/input/raw-dicom-samples`).
* It sets the output directory to `/kaggle/working/anonymized_output` and turns on GPU automatically.

If you are running via the command line, run:
```bash
!python dicom_anonymizer_pipeline.py --gpu
```

### Step 5: Download the Entire Output Folder
Because Kaggle does not support downloading whole folders directly, run this cell to zip the output directory and generate a direct download link:
```python
import shutil
from IPython.display import FileLink

# Zip folder
shutil.make_archive("anonymized_output", "zip", "/kaggle/working/anonymized_output")

# Generate download link
FileLink('anonymized_output.zip')
```
Click the blue **`anonymized_output.zip`** link that appears in the cell output to download everything.

---

## 🌐 Google Colab Setup (Google Drive Mounted)

Running in Google Colab lets you read and write files directly from your Google Drive.

### Step 1: Set Runtime to GPU
1. Go to **Runtime** -> **Change runtime type**.
2. Select **T4 GPU** under Hardware Accelerator.

### Step 2: Mount Google Drive
Run this cell to connect your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### Step 3: Install Dependencies & Run
Install dependencies using the same commands as Kaggle, then run the script pointing to your Drive folders:
```bash
# Run anonymizer
!python dicom_anonymizer_pipeline.py -i "/content/drive/MyDrive/raw_dicoms" -o "/content/drive/MyDrive/anonymized_dicoms" --gpu
```
Your de-identified files will be written directly back to your Google Drive folder, avoiding the need to manually download them.
