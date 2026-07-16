import os
import pydicom
import numpy as np
import cv2
from pixel_anonymizer import PixelAnonymizer

# Set up directories
SAMPLES_DIR = "./dicom_samples"
OUTPUT_DIR = "./test_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Instantiate our anonymizer
print("Initializing PixelAnonymizer...")
anonymizer = PixelAnonymizer(use_gpu=False)

# List of downloaded real DICOM files to test
test_files = [
    "sample_cr_RG1_UNCR.dcm",  # X-ray 1
    "sample_cr_RG3_UNCR.dcm",  # X-ray 2
    "sample_mr_MR-SIEMENS-DICOM-WithOverlays.dcm",  # MRI (has overlay layers)
    "sample_ct_CT_small.dcm"  # CT Scan
]

print("\n" + "="*60)
print("TESTING ANONYMIZER ON REAL MEDICAL SCANS")
print("="*60)

for filename in test_files:
    path = os.path.join(SAMPLES_DIR, filename)
    if not os.path.exists(path):
        print(f"Skipping {filename} (file not found)")
        continue
        
    print(f"\nProcessing real file: {filename}")
    try:
        # Read the DICOM dataset
        ds = pydicom.dcmread(path)
        
        # Decompress if compressed (pylibjpeg handles this)
        ts = ds.file_meta.TransferSyntaxUID
        if hasattr(ds, "decompress"):
            ds.decompress()
            
        if not hasattr(ds, "pixel_array"):
            print(f"  [WARN] No pixel data found in {filename}")
            continue
            
        pixels = ds.pixel_array
        
        # Strip overlay tags if MR (to test overlay removal)
        # MR-SIEMENS file has overlay data in group 6000
        has_overlays = False
        for group in range(0x6000, 0x601F, 2):
            for elem in list(ds):
                if elem.tag.group == group:
                    has_overlays = True
                    del ds[elem.tag]
        if has_overlays:
            print("  [INFO] Stripped overlay group tags (60xx) from dataset.")

        # Modality check and photometric interpretation flip
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        if photometric == "MONOCHROME1":
            print("  [INFO] MONOCHROME1 detected: Flipping pixel values for OCR correctness.")
            pixels = np.max(pixels) - pixels

        # Run anonymization on the pixels
        print("  Running pixel anonymizer...")
        clean_pixels, status = anonymizer.anonymize(pixels, modality=getattr(ds, "Modality", "CR"))
        print(f"  Anonymization Status: {status}")

        # Normalization helper for saving comparison PNGs (8-bit)
        def to_8bit(arr):
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                return ((arr - arr_min) / (arr_max - arr_min) * 255.0).astype(np.uint8)
            return arr.astype(np.uint8)

        orig_8bit = to_8bit(pixels)
        clean_8bit = to_8bit(clean_pixels)

        # Create side-by-side visual comparison
        # Resize high-resolution images down slightly for easy viewing (e.g. max width 800)
        max_view_w = 800
        h, w = orig_8bit.shape[:2]
        if w > max_view_w:
            scale = max_view_w / w
            new_size = (max_view_w, int(h * scale))
            orig_view = cv2.resize(orig_8bit, new_size)
            clean_view = cv2.resize(clean_8bit, new_size)
        else:
            orig_view = orig_8bit
            clean_view = clean_8bit

        # Draw division line and stack horizontally
        comparison = np.hstack([orig_view, clean_view])
        
        # Save output comparison image
        output_path = os.path.join(OUTPUT_DIR, f"compare_{filename.replace('.dcm', '.png')}")
        cv2.imwrite(output_path, comparison)
        print(f"  [SUCCESS] Saved visual comparison to: {output_path}")

    except Exception as e:
        print(f"  [ERROR] Failed to process {filename}: {e}")

print("\n" + "="*60)
print(f"Finished processing. Open the folder: {os.path.abspath(OUTPUT_DIR)}")
print("="*60)
