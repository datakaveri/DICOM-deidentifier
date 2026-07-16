"""
=============================================================================
CHEST X-RAY DICOM DOWNLOADER
=============================================================================
Downloads real open-source chest X-ray DICOM files from public sources:
  1. pydicom built-in test data (RG1/RG3 CR chest X-rays)
  2. Orthanc public DICOM server demo files
  3. Rubo Medical open DICOM samples

Output: ./raw_datasamples_xray_chest/ (UNMODIFIED originals)
=============================================================================
"""

import os
import shutil
import urllib.request
import urllib.error
import pydicom
from pydicom.data import get_testdata_file

OUTPUT_DIR = "./raw_datasamples_xray_chest"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("CHEST X-RAY DICOM DOWNLOADER")
print("Output directory:", os.path.abspath(OUTPUT_DIR))
print("=" * 70)

downloaded = 0
failed = 0

# =============================================================================
# SOURCE 1: pydicom built-in test data (RG = Radiograph = X-ray, CR modality)
# These are real anonymized clinical X-ray DICOM files bundled with pydicom
# =============================================================================
print("\n[SOURCE 1] pydicom built-in chest X-ray test data (CR/RG modality)...")

pydicom_xray_files = [
    ("RG1_UNCR.dcm",  "chest_xray_RG1_raw.dcm"),
    ("RG3_UNCR.dcm",  "chest_xray_RG3_raw.dcm"),
]

for src_name, dest_name in pydicom_xray_files:
    try:
        src_path = get_testdata_file(src_name)
        ds = pydicom.dcmread(src_path)
        modality = getattr(ds, "Modality", "Unknown")
        rows = getattr(ds, "Rows", "?")
        cols = getattr(ds, "Columns", "?")
        bits = getattr(ds, "BitsAllocated", "?")
        dest = os.path.join(OUTPUT_DIR, dest_name)
        shutil.copy(src_path, dest)
        print(f"  [OK] {dest_name}")
        print(f"       Modality={modality} | Size={rows}x{cols} | Bits={bits}")
        downloaded += 1
    except Exception as e:
        print(f"  [FAIL] {src_name}: {e}")
        failed += 1

# =============================================================================
# SOURCE 2: Orthanc public demo server — real DICOM chest X-rays
# Public DICOM test files from orthanc-server.com
# =============================================================================
print("\n[SOURCE 2] Orthanc public DICOM demo server...")

orthanc_urls = [
    (
        "https://orthanc.uclouvain.be/demo/instances/9c48cb16-d47f4faa-a4b96370-9d6d5a03-fcb11107/file",
        "chest_xray_orthanc_1.dcm"
    ),
    (
        "https://orthanc.uclouvain.be/demo/instances/66a662ce-7ef2a7f2-27252ae3-84543bb8-25fe4d15/file",
        "chest_xray_orthanc_2.dcm"
    ),
]

for url, dest_name in orthanc_urls:
    dest = os.path.join(OUTPUT_DIR, dest_name)
    try:
        print(f"  Downloading {dest_name} from Orthanc...")
        req = urllib.request.Request(url, headers={"Accept": "application/dicom"})
        with urllib.request.urlopen(req, timeout=20) as response:
            data = response.read()
        with open(dest, "wb") as f:
            f.write(data)
        # Validate it's a real DICOM
        ds = pydicom.dcmread(dest)
        modality = getattr(ds, "Modality", "Unknown")
        rows = getattr(ds, "Rows", "?")
        cols = getattr(ds, "Columns", "?")
        print(f"  [OK] {dest_name} | Modality={modality} | Size={rows}x{cols}")
        downloaded += 1
    except Exception as e:
        print(f"  [SKIP] {dest_name}: {e}")
        # Remove partial file if download failed
        if os.path.exists(dest):
            os.remove(dest)
        failed += 1

# =============================================================================
# SOURCE 3: Rubo Medical open DICOM samples (chest X-ray)
# =============================================================================
print("\n[SOURCE 3] Rubo Medical open DICOM samples...")

rubo_urls = [
    (
        "https://www.rubomedical.com/dicom_files/0001.DCM",
        "chest_xray_rubo_1.dcm"
    ),
    (
        "https://www.rubomedical.com/dicom_files/0002.DCM",
        "chest_xray_rubo_2.dcm"
    ),
]

for url, dest_name in rubo_urls:
    dest = os.path.join(OUTPUT_DIR, dest_name)
    try:
        print(f"  Downloading {dest_name} from Rubo Medical...")
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0", "Accept": "*/*"}
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
        with open(dest, "wb") as f:
            f.write(data)
        # Validate
        ds = pydicom.dcmread(dest, force=True)
        modality = getattr(ds, "Modality", "Unknown")
        rows = getattr(ds, "Rows", "?")
        cols = getattr(ds, "Columns", "?")
        bits = getattr(ds, "BitsAllocated", "?")
        print(f"  [OK] {dest_name} | Modality={modality} | Size={rows}x{cols} | Bits={bits}")
        downloaded += 1
    except Exception as e:
        print(f"  [SKIP] {dest_name}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        failed += 1

# =============================================================================
# SOURCE 4: DICOM sample files from sample-dicom-files.com
# =============================================================================
print("\n[SOURCE 4] Additional open DICOM X-ray samples...")

extra_urls = [
    (
        "https://github.com/pydicom/pydicom/raw/main/pydicom/data/test_files/CT_small.dcm",
        "chest_ct_small_raw.dcm"
    ),
]

for url, dest_name in extra_urls:
    dest = os.path.join(OUTPUT_DIR, dest_name)
    try:
        print(f"  Downloading {dest_name}...")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as response:
            data = response.read()
        with open(dest, "wb") as f:
            f.write(data)
        ds = pydicom.dcmread(dest)
        modality = getattr(ds, "Modality", "Unknown")
        rows = getattr(ds, "Rows", "?")
        cols = getattr(ds, "Columns", "?")
        print(f"  [OK] {dest_name} | Modality={modality} | Size={rows}x{cols}")
        downloaded += 1
    except Exception as e:
        print(f"  [SKIP] {dest_name}: {e}")
        if os.path.exists(dest):
            os.remove(dest)
        failed += 1

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("DOWNLOAD COMPLETE")
print("=" * 70)

all_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.dcm', '.DCM'))]
print(f"\nTotal DICOM files in raw_datasamples_xray_chest: {len(all_files)}")
print(f"  Successfully downloaded: {downloaded}")
print(f"  Failed/Skipped:          {failed}")
print()
for fname in sorted(all_files):
    fsize = os.path.getsize(os.path.join(OUTPUT_DIR, fname))
    print(f"  {fname:45s}  {fsize/1024/1024:.2f} MB")
print()
print(f"Location: {os.path.abspath(OUTPUT_DIR)}")
print("=" * 70)
