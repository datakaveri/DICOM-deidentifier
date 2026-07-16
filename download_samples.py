import os
import shutil
import pydicom
from pydicom.data import get_testdata_file

# Define the target directory for saving samples
TARGET_DIR = "./dicom_samples"
os.makedirs(TARGET_DIR, exist_ok=True)

# List of files we want to download/extract from pydicom's test database
# These represent CT, MRI (with overlays), and X-rays (RG/CR)
sample_requests = {
    "CT Scan (Small)": "CT_small.dcm",
    "MRI Brain (Siemens with Overlays)": "MR-SIEMENS-DICOM-WithOverlays.dcm",
    "MRI Brain (Small)": "MR_small.dcm",
    "X-Ray Image (RG1)": "RG1_UNCR.dcm",
    "X-Ray Image (RG3)": "RG3_UNCR.dcm",
    "Liver Scan (CT/MRI)": "liver.dcm"
}

print("=" * 60)
print("Downloading / extracting sample DICOM datasets...")
print("=" * 60)

for description, filename in sample_requests.items():
    try:
        print(f"Retrieving {description} ({filename})...")
        # get_testdata_file will automatically download the file from GitHub if it's not present locally
        file_path = get_testdata_file(filename)
        
        # Read the file to verify it's valid and print its metadata/modality
        ds = pydicom.dcmread(file_path)
        modality = getattr(ds, "Modality", "Unknown")
        rows = getattr(ds, "Rows", "Unknown")
        cols = getattr(ds, "Columns", "Unknown")
        
        # Determine output file name
        dest_filename = f"sample_{modality.lower()}_{filename}"
        dest_path = os.path.join(TARGET_DIR, dest_filename)
        
        # Copy to our local samples directory
        shutil.copy(file_path, dest_path)
        print(f"  [SUCCESS] Saved to: {dest_path}")
        print(f"            Modality: {modality} | Resolution: {rows}x{cols}")
        
    except Exception as e:
        print(f"  [ERROR] Failed to retrieve {filename}: {e}")

print("=" * 60)
print(f"All available samples have been copied to: {os.path.abspath(TARGET_DIR)}")
print("=" * 60)
