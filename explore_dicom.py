"""
=============================================================
DICOM FILE EXPLORER - See What's Inside a Medical Image
=============================================================
This script:
1. Downloads a sample DICOM chest X-ray file
2. Shows you ALL the metadata (patient info, hospital info, etc.)
3. Displays the actual X-ray image
4. Highlights which fields contain PII that needs anonymization
=============================================================
"""

import pydicom
from pydicom.data import get_testdata_file
import os
import sys

# ============================================================
# STEP 1: Get a sample DICOM file (built into pydicom library)
# ============================================================
print("=" * 70)
print("STEP 1: Loading Sample DICOM Files")
print("=" * 70)

# Pydicom comes with built-in test DICOM files
try:
    ct_path = get_testdata_file("CT_small.dcm")
    mr_path = get_testdata_file("MR_small.dcm")
    sample_files = {
        "CT Scan": pydicom.dcmread(ct_path),
        "MR Brain": pydicom.dcmread(mr_path),
    }
except Exception as e:
    print(f"Failed to load pydicom test files: {e}")
    sys.exit(1)


for name, ds in sample_files.items():
    print(f"\n{'='*70}")
    print(f"  EXPLORING: {name}")
    print(f"{'='*70}")
    
    # ============================================================
    # STEP 2: Show ALL metadata tags (this is what gets anonymized)
    # ============================================================
    print(f"\n--- ALL DICOM METADATA TAGS ---")
    print(f"Total number of tags: {len(ds)}")
    print()
    
    # Categorize tags into PII and Non-PII
    pii_tags = []
    clinical_tags = []
    
    # Known PII tag keywords
    pii_keywords = [
        'patient', 'name', 'birth', 'age', 'sex', 'address',
        'physician', 'doctor', 'operator', 'institution', 'hospital',
        'accession', 'referring', 'performing', 'requesting',
        'telephone', 'ethnic', 'occupation', 'insurance',
        'religious', 'responsible', 'station name'
    ]
    
    for elem in ds:
        tag_name = elem.keyword if elem.keyword else str(elem.tag)
        tag_value = str(elem.value)[:80] if elem.value else "N/A"
        
        # Skip pixel data (too large to print)
        if elem.keyword == 'PixelData':
            continue
        
        is_pii = any(kw in tag_name.lower() for kw in pii_keywords)
        
        if is_pii:
            pii_tags.append((str(elem.tag), tag_name, tag_value))
        else:
            clinical_tags.append((str(elem.tag), tag_name, tag_value))
    
    # Print PII tags (what needs to be REMOVED)
    print("[PII] TAGS (MUST BE ANONYMIZED):")
    print("-" * 70)
    if pii_tags:
        for tag_id, tag_name, tag_value in pii_tags:
            print(f"  {tag_id:20s} | {tag_name:35s} | {tag_value}")
    else:
        print("  (No PII tags found - this file may already be de-identified)")
    
    print()
    print("[SAFE] CLINICAL/TECHNICAL TAGS (SAFE TO KEEP):")
    print("-" * 70)
    for tag_id, tag_name, tag_value in clinical_tags[:25]:  # Show first 25
        print(f"  {tag_id:20s} | {tag_name:35s} | {tag_value}")
    if len(clinical_tags) > 25:
        print(f"  ... and {len(clinical_tags) - 25} more clinical tags")
    
    # ============================================================
    # STEP 3: Show specific dangerous PII fields
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  SPECIFIC PII FIELDS IN THIS {name.upper()} FILE:")
    print(f"{'='*70}")
    
    dangerous_fields = [
        ('PatientName', 'Patient Name'),
        ('PatientID', 'Patient ID'),
        ('PatientBirthDate', 'Date of Birth'),
        ('PatientSex', 'Patient Sex'),
        ('PatientAge', 'Patient Age'),
        ('PatientWeight', 'Patient Weight'),
        ('ReferringPhysicianName', 'Referring Doctor'),
        ('InstitutionName', 'Hospital/Institution'),
        ('StationName', 'Machine/Station Name'),
        ('AccessionNumber', 'Accession Number'),
        ('StudyDate', 'Study Date'),
        ('StudyTime', 'Study Time'),
        ('PerformingPhysicianName', 'Performing Doctor'),
        ('OperatorsName', 'Operator Name'),
        ('InstitutionalDepartmentName', 'Department'),
    ]
    
    for attr, label in dangerous_fields:
        value = getattr(ds, attr, 'NOT PRESENT')
        status = "[REMOVE]" if value and value != 'NOT PRESENT' else "[ N/A ]"
        print(f"  {status} | {label:30s} | {value}")
    
    # ============================================================
    # STEP 4: Show image info
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  IMAGE PROPERTIES:")
    print(f"{'='*70}")
    print(f"  Modality:        {getattr(ds, 'Modality', 'N/A')}")
    print(f"  Image Size:      {getattr(ds, 'Rows', 'N/A')} x {getattr(ds, 'Columns', 'N/A')} pixels")
    print(f"  Bits Allocated:  {getattr(ds, 'BitsAllocated', 'N/A')}")
    print(f"  Bits Stored:     {getattr(ds, 'BitsStored', 'N/A')}")
    print(f"  Samples/Pixel:   {getattr(ds, 'SamplesPerPixel', 'N/A')}")
    print(f"  Body Part:       {getattr(ds, 'BodyPartExamined', 'N/A')}")

# ============================================================
# STEP 5: Save the image as PNG so you can SEE it
# ============================================================
print(f"\n{'='*70}")
print("STEP 5: Saving DICOM images as viewable PNG files...")
print(f"{'='*70}")

try:
    import numpy as np
    from PIL import Image
    
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dicom_samples")
    os.makedirs(output_dir, exist_ok=True)
    
    for name, ds in sample_files.items():
        if hasattr(ds, 'pixel_array'):
            pixel_array = ds.pixel_array
            
            # Normalize to 0-255 for display
            img_min = pixel_array.min()
            img_max = pixel_array.max()
            if img_max > img_min:
                normalized = ((pixel_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                normalized = pixel_array.astype(np.uint8)
            
            img = Image.fromarray(normalized)
            filename = f"{name.replace(' ', '_').lower()}_sample.png"
            filepath = os.path.join(output_dir, filename)
            img.save(filepath)
            print(f"  [OK] Saved: {filepath}")
        else:
            print(f"  [WARN] {name}: No pixel data available")
    
    print(f"\n  [INFO] Open the 'dicom_samples' folder to see the images!")

except Exception as e:
    print(f"  [WARN] Could not save images: {e}")

# ============================================================
# STEP 6: DEMONSTRATION - How anonymization works
# ============================================================
print(f"\n{'='*70}")
print("STEP 6: ANONYMIZATION DEMO")
print("         Showing BEFORE vs AFTER anonymization")
print(f"{'='*70}")

import copy

# Take the CT scan example
original = sample_files["CT Scan"]
anonymized = copy.deepcopy(original)

print("\n  BEFORE ANONYMIZATION:")
print(f"  Patient Name:     {original.PatientName}")
print(f"  Patient ID:       {original.PatientID}")
print(f"  Patient Birth:    {getattr(original, 'PatientBirthDate', 'N/A')}")
print(f"  Institution:      {getattr(original, 'InstitutionName', 'N/A')}")
print(f"  Study Date:       {original.StudyDate}")
print(f"  Referring Doctor:  {getattr(original, 'ReferringPhysicianName', 'N/A')}")

anonymized.PatientName = ""
anonymized.PatientID = ""
if hasattr(anonymized, 'PatientBirthDate'):
    anonymized.PatientBirthDate = ""
if hasattr(anonymized, 'InstitutionName'):
    anonymized.InstitutionName = ""
anonymized.StudyDate = ""
if hasattr(anonymized, 'ReferringPhysicianName'):
    anonymized.ReferringPhysicianName = ""

print("\n  AFTER ANONYMIZATION:")
print(f"  Patient Name:     {anonymized.PatientName}")
print(f"  Patient ID:       {anonymized.PatientID}")
print(f"  Patient Birth:    {getattr(anonymized, 'PatientBirthDate', 'N/A')}")
print(f"  Institution:      {getattr(anonymized, 'InstitutionName', 'N/A')}")
print(f"  Study Date:       {anonymized.StudyDate}")
print(f"  Referring Doctor:  {getattr(anonymized, 'ReferringPhysicianName', 'N/A')}")

# Save anonymized DICOM
anon_path = os.path.join(output_dir, "anonymized_ct_sample.dcm")
anonymized.save_as(anon_path)
print(f"\n  [OK] Anonymized DICOM saved to: {anon_path}")

print(f"\n{'='*70}")
print("  [KEY TAKEAWAY]:")
print("  The metadata anonymization (shown above) is the EASY part.")
print("  The HARD part is detecting and removing 'burned-in' text")
print("  that's baked into the actual image pixels.")
print("  That's where AI models (PaddleOCR, VLMs) are needed!")
print(f"{'='*70}")

# ============================================================
# STEP 7: Show downloadable resources
# ============================================================
print(f"\n{'='*70}")
print("STEP 7: WHERE TO DOWNLOAD MORE DICOM FILES")
print(f"{'='*70}")
print("""
  [FREE DICOM SAMPLE SOURCES]:
  
  1. DICOM Library (view online + download):
     https://www.dicomlibrary.com/
     
  2. Saga IT DICOM Samples (curated catalog):
     https://dicom.sample-files.com/
     
  3. Rubo Medical (X-ray, CT, ultrasound demos):
     https://www.rubomedical.com/dicom_files/
     
  4. The Cancer Imaging Archive (large clinical datasets):
     https://www.cancerimagingarchive.net/
     
  5. MIMIC-CXR (377k+ chest X-rays, needs registration):
     https://physionet.org/content/mimic-cxr/2.1.0/
  
  6. NIH Chest X-ray (DICOM on Google Cloud):
     gs://gcs-public-data--healthcare-nih-chest-xray/dicom/
     
  7. Kaggle - Search "chest xray DICOM":
     https://www.kaggle.com/search?q=chest+xray+dicom
     
  [FREE DICOM VIEWERS]:
  
  1. 3D Slicer (Windows/Mac/Linux):   https://www.slicer.org/
  2. OHIF Viewer (web-based):          https://viewer.ohif.org/
  3. MicroDicom (Windows, lightweight): https://www.microdicom.com/
  4. Horos (Mac):                       https://horosproject.org/
""")
