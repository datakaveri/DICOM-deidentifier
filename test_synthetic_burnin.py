# -*- coding: utf-8 -*-
"""
=============================================================================
SYNTHETIC DICOM STRESS TEST — Proves burned-in text redaction works
=============================================================================
Creates a synthetic 16-bit chest X-ray DICOM with realistic burned-in PHI:
  - Patient Name, DOB, ID in top border
  - Hospital name in bottom border
  - Accession number on side
  - Clinical markers "L", "PA" (should be KEPT)

Then runs the full anonymization pipeline and saves the result as DICOM.
Saves both the RAW and ANONYMIZED .dcm to their respective folders so you
can open both in MicroDicom/3D Slicer and compare visually.

Output:
  raw_datasamples_xray_chest/synthetic_burnin_test_raw.dcm
  anonymized_xray_chest/anon_synthetic_burnin_test_raw.dcm
=============================================================================
"""

import os
import sys
import numpy as np
import cv2
import pydicom
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import generate_uid, ExplicitVRLittleEndian
from pydicom.sequence import Sequence
import datetime

RAW_DIR  = "./raw_datasamples_xray_chest"
ANON_DIR = "./anonymized_xray_chest"
os.makedirs(RAW_DIR,  exist_ok=True)
os.makedirs(ANON_DIR, exist_ok=True)


# =============================================================================
# STEP 1: Build a realistic synthetic 16-bit chest X-ray pixel array
# =============================================================================
def make_synthetic_xray(h=1024, w=1024):
    """
    Creates a 16-bit grayscale synthetic chest X-ray image.
    Background = lung field (low values), ribcage structure = bright values.
    Burned-in PHI text is drawn in the border regions.
    """
    # Background: simulate dark lung fields
    img = np.full((h, w), 800, dtype=np.uint16)

    # Add a rough circular lung field on each side (brighter = denser tissue)
    cx_left, cx_right = w // 4, (3 * w) // 4
    cy = h // 2
    Y, X = np.ogrid[:h, :w]

    # Lung fields (elliptical, darker = air)
    left_lung  = ((X - cx_left)**2 / (w//5)**2 + (Y - cy)**2 / (h//3)**2) < 1
    right_lung = ((X - cx_right)**2 / (w//5)**2 + (Y - cy)**2 / (h//3)**2) < 1
    img[left_lung]  = np.random.randint(200, 400, int(left_lung.sum()), dtype=np.uint16)
    img[right_lung] = np.random.randint(200, 400, int(right_lung.sum()), dtype=np.uint16)

    # Rib-like horizontal bands (bright = bone density)
    for rib_y in range(cy - h//3, cy + h//3, h // 18):
        thickness = np.random.randint(6, 14)
        img[rib_y:rib_y+thickness, w//6:5*w//6] = np.random.randint(2800, 3600)

    # Spine centre line
    img[cy-h//3:cy+h//3, w//2-12:w//2+12] = np.random.randint(3000, 4095)

    # Add natural radiographic noise
    noise = np.random.normal(0, 40, (h, w)).astype(np.int32)
    img = np.clip(img.astype(np.int32) + noise, 0, 4095).astype(np.uint16)

    # ── Burn-in PHI text into the image ──────────────────────────────────────
    # Scale to 8-bit for cv2.putText, draw, then scale the text ink back up
    img_8 = ((img.astype(float) / 4095.0) * 255.0).astype(np.uint8)

    font = cv2.FONT_HERSHEY_SIMPLEX

    # TOP BORDER — Patient info (bright white text on dark background)
    cv2.putText(img_8, "SHARMA RAJESH KUMAR",   (10,  35), font, 0.9,  255, 2)
    cv2.putText(img_8, "DOB: 15/03/1976",        (10,  65), font, 0.75, 255, 2)
    cv2.putText(img_8, "ID: UHID-20231187",       (10,  90), font, 0.75, 255, 2)
    cv2.putText(img_8, "Aadhaar: 4521 8834 7712", (10, 115), font, 0.65, 255, 2)
    cv2.putText(img_8, "Ph: +91 9876543210",       (10, 140), font, 0.65, 255, 2)

    # RIGHT BORDER — Accession + date
    cv2.putText(img_8, "ACC: RD20240788",   (w-240, 50), font, 0.65, 255, 2)
    cv2.putText(img_8, "15/06/2024",         (w-220, 80), font, 0.65, 255, 2)
    cv2.putText(img_8, "Dr. Priya Mehta",    (w-240,110), font, 0.65, 255, 2)

    # BOTTOM BORDER — Hospital name
    cv2.putText(img_8, "GOVT. MEDICAL COLLEGE & HOSPITAL, ROHTAK", (10, h-50), font, 0.7, 255, 2)
    cv2.putText(img_8, "Radiology Dept | HARSAC Project 2024",     (10, h-20), font, 0.65, 255, 2)

    # LEFT BORDER — Clinical markers that MUST be KEPT
    cv2.putText(img_8, "L",   (12, h//2),    font, 1.5, 255, 3)  # <- should be KEPT
    cv2.putText(img_8, "PA",  (12, h//2+60), font, 1.0, 255, 2)  # <- should be KEPT

    # ANATOMY ZONE PHI — should be redacted using hybrid masking (leaves ribs intact)
    cv2.putText(img_8, "CONFIDENTIAL STUDY", (w//2 - 180, h//2 - 100), font, 0.7, 255, 2)
    cv2.putText(img_8, "PATIENT_ID: 99120",   (w//2 - 140, h//2 + 100), font, 0.7, 255, 2)


    # Map burned-in text pixels back to 16-bit dynamic range
    # Text pixels appear at 255 in 8-bit → map to ~3800 in 16-bit (bright)
    text_mask = (img_8 > 200)  # pixels that were painted white by putText
    img[text_mask] = (img_8[text_mask].astype(float) / 255.0 * 4095.0).astype(np.uint16)

    return img, img_8


# =============================================================================
# STEP 2: Package pixel array into a valid DICOM file
# =============================================================================
def create_dicom(pixel_array, filename):
    """Wraps a 16-bit pixel array into a valid DICOM dataset and saves to disk."""
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID    = "1.2.840.10008.5.1.4.1.1.1"  # CR Image Storage
    file_meta.MediaStorageSOPInstanceUID = generate_uid()
    file_meta.TransferSyntaxUID          = ExplicitVRLittleEndian

    ds = FileDataset(filename, {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Timing
    now = datetime.datetime.now()
    ds.StudyDate       = now.strftime("%Y%m%d")
    ds.StudyTime       = now.strftime("%H%M%S")
    ds.ContentDate     = ds.StudyDate
    ds.ContentTime     = ds.StudyTime

    # Patient / Study info (intentionally filled — to be anonymized)
    ds.PatientName      = "SHARMA^RAJESH^KUMAR"
    ds.PatientID        = "UHID-20231187"
    ds.PatientBirthDate = "19760315"
    ds.PatientSex       = "M"
    ds.PatientAge       = "048Y"
    ds.InstitutionName  = "GOVT. MEDICAL COLLEGE ROHTAK"
    ds.ReferringPhysicianName = "MEHTA^PRIYA"
    ds.AccessionNumber  = "RD20240788"
    ds.StudyID          = "STU001"
    ds.SeriesNumber     = "1"
    ds.InstanceNumber   = "1"

    # Image geometry
    h, w = pixel_array.shape
    ds.Rows            = h
    ds.Columns         = w
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated   = 16
    ds.BitsStored      = 16
    ds.HighBit         = 15
    ds.PixelRepresentation = 0  # unsigned

    # Modality
    ds.Modality        = "CR"
    ds.SOPClassUID     = file_meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID  = generate_uid()
    ds.StudyInstanceUID  = generate_uid()
    ds.SeriesInstanceUID = generate_uid()
    ds.is_implicit_VR    = False
    ds.is_little_endian  = True

    # Pixel data
    ds.PixelData = pixel_array.tobytes()

    ds.save_as(filename, write_like_original=False)
    print(f"  [OK] Saved raw DICOM: {filename}")
    return ds


# =============================================================================
# STEP 3: Run the full anonymization pipeline on the synthetic DICOM
# =============================================================================
def run_pipeline(raw_path, anon_path):
    # Import pipeline functions
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dicom_anonymizer_pipeline import (
        sanitize_metadata, enhance_image, detect_text,
        merge_detections, classify_phi, redact_pixels,
        verify_redaction, write_pixels_to_dicom,
        PADDLE_AVAILABLE, EASYOCR_AVAILABLE, PRESIDIO_AVAILABLE
    )

    print("\n  Initializing OCR engines...")
    paddle_ocr = None
    easy_ocr   = None
    analyzer   = None

    if PADDLE_AVAILABLE:
        try:
            from paddleocr import PaddleOCR
            paddle_ocr = PaddleOCR(lang='en', enable_mkldnn=False)
            print("  [OK] PaddleOCR ready")
        except Exception as e:
            print(f"  [WARN] PaddleOCR: {e}")

    if EASYOCR_AVAILABLE:
        try:
            import easyocr
            easy_ocr = easyocr.Reader(['en'], gpu=False)
            print("  [OK] EasyOCR ready")
        except Exception as e:
            print(f"  [WARN] EasyOCR: {e}")

    if PRESIDIO_AVAILABLE:
        try:
            from presidio_analyzer import AnalyzerEngine
            analyzer = AnalyzerEngine()
            print("  [OK] Presidio NLP ready")
        except Exception as e:
            print(f"  [WARN] Presidio: {e}")

    if not paddle_ocr and not easy_ocr:
        print("  [ERROR] No OCR engine available. Cannot test pixel redaction.")
        return

    print(f"\n  Reading raw DICOM: {raw_path}")
    ds = pydicom.dcmread(raw_path)
    pixels = ds.pixel_array.copy()

    print(f"  Image shape: {pixels.shape}  dtype: {pixels.dtype}")

    # Stage 1: metadata
    print("\n  [Stage 1] Sanitizing metadata...")
    ds = sanitize_metadata(ds)

    # Normalize to 8-bit for OCR
    pmin, pmax = pixels.min(), pixels.max()
    norm_8 = ((pixels - pmin) / (pmax - pmin) * 255.0).astype(np.uint8)

    # Stage 2: enhance
    print("  [Stage 2] Generating image variants...")
    variants = enhance_image(norm_8)

    # Stage 3: OCR
    print("  [Stage 3] Running OCR...")
    raw_det = detect_text(variants, paddle_ocr, easy_ocr)
    print(f"            Raw detections: {len(raw_det)}")
    for d in raw_det[:10]:
        print(f"            -> '{d['text']}'  conf={d['confidence']:.2f}  variant={d['variant']}")

    # Stage 4: classify
    print("  [Stage 4] Classifying PHI regions...")
    merged     = merge_detections(raw_det)
    phi_regions = classify_phi(merged, norm_8.shape, analyzer)
    print(f"            PHI regions to redact: {len(phi_regions)}")
    for r in phi_regions:
        print(f"            -> REDACT: '{r['text']}'  @ {r['bbox']}")

    # Stage 5: redact
    print("  [Stage 5] Inpainting PHI pixels...")
    cleaned, mask = redact_pixels(pixels, phi_regions)
    print(f"            Inpainted pixels: {int(np.sum(mask > 0))}")

    # Stage 6: verify
    print("  [Stage 6] Verification pass...")
    cleaned_final, status = verify_redaction(cleaned, phi_regions, paddle_ocr, easy_ocr, analyzer)
    print(f"            Status: {status}")

    # Stage 7: write back
    print("  [Stage 7] Writing pixels to DICOM...")
    ds = write_pixels_to_dicom(ds, cleaned_final)
    ds.save_as(anon_path, write_like_original=False)
    print(f"  [OK] Anonymized DICOM saved: {anon_path}")

    return status, len(phi_regions)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SYNTHETIC CHEST X-RAY DICOM — BURNED-IN TEXT REDACTION TEST")
    print("=" * 70)

    raw_path  = os.path.join(RAW_DIR,  "synthetic_burnin_test_raw.dcm")
    anon_path = os.path.join(ANON_DIR, "anon_synthetic_burnin_test_raw.dcm")

    # Step 1: create synthetic X-ray with PHI burned in
    print("\n[STEP 1] Creating synthetic 16-bit chest X-ray with burned-in PHI...")
    pixel_array, _ = make_synthetic_xray(h=1024, w=1024)
    create_dicom(pixel_array, raw_path)

    print(f"\n  Raw DICOM (with PHI) saved to:")
    print(f"  {os.path.abspath(raw_path)}")

    # Step 2: run full pipeline
    print("\n[STEP 2] Running full anonymization pipeline...")
    result = run_pipeline(raw_path, anon_path)

    print("\n" + "=" * 70)
    print("SYNTHETIC STRESS TEST COMPLETE")
    print("=" * 70)
    if result:
        status, n_regions = result
        print(f"\n  PHI regions detected & redacted : {n_regions}")
        print(f"  Verification status             : {status}")
    print(f"\n  Compare these two files in MicroDicom or 3D Slicer:")
    print(f"  RAW  (with PHI) : {os.path.abspath(raw_path)}")
    print(f"  ANON (clean)    : {os.path.abspath(anon_path)}")
    print("\n  What to check:")
    print("  - Patient name / DOB / ID text in border -> should be GONE in ANON")
    print("  - 'L' and 'PA' clinical markers          -> should be KEPT in ANON")
    print("  - Lung/rib anatomy                       -> must be INTACT in ANON")
    print("  - File format                            -> both must open as valid DICOM")
    print("=" * 70)
