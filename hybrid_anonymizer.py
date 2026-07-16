"""
================================================================================
5-LAYER HYBRID MEDICAL IMAGE ANONYMIZER (100% PERMISSIVE STACK)
================================================================================
Libraries required:
    pip install pydicom opencv-python numpy paddleocr presidio-analyzer
================================================================================
"""

import os
import re
import json
import logging

# Disable mkldnn to prevent ConvertPirAttribute2RuntimeAttribute errors on CPU
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
import numpy as np
import cv2
import pydicom
from pydicom.uid import generate_uid

# Try importing PaddleOCR
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# Try importing Presidio
try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False

# Try importing dicomanonymizer
try:
    import dicomanonymizer
    DICOM_ANONYMIZER_AVAILABLE = True
except ImportError:
    DICOM_ANONYMIZER_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridDicomAnonymizer:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        
        # Initialize Layer 2 & 3 OCR Engine (Apache 2.0)
        if PADDLE_AVAILABLE:
            if use_gpu:
                try:
                    import paddle
                    paddle.device.set_device('gpu')
                except Exception:
                    pass
            os.environ["DISABLE_AUTO_LOGGING_CONFIG"] = "1"
            try:
                from paddleocr import logger as paddle_logger
                paddle_logger.setLevel(logging.ERROR)
            except Exception:
                pass
            self.ocr = PaddleOCR(lang='en', enable_mkldnn=False)
            logging.info("PaddleOCR engine loaded successfully.")
        else:
            self.ocr = None
            logging.warning("PaddleOCR not installed. OCR layers will be bypassed.")
            
        # Initialize Layer 3 NLP Engine (MIT)
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            logging.info("Presidio NLP engine loaded successfully.")
        else:
            self.analyzer = None
            logging.warning("Presidio Analyzer not installed. Falling back to pattern matching.")
            
        # Clinical Allowlist: Clinical tags that should NOT be redacted
        self.clinical_allowlist = {
            "L", "R", "PA", "AP", "LAT", "LEFT", "RIGHT", "PORTABLE", 
            "CHEST", "UPRIGHT", "DECUB", "APRIGHT", "ERECT", "SUPINE"
        }
        
    def _is_clinical_marker(self, text):
        """Checks if a string is a clinical direction/marker to avoid over-redaction."""
        clean_text = re.sub(r'[^A-Z]', '', text.upper())
        return clean_text in self.clinical_allowlist

    def _parse_ocr_results(self, results):
        """Parses OCR output from both legacy PaddleOCR formats and newer PaddleX OCRResult objects."""
        parsed = []
        if not results or len(results) == 0:
            return parsed
            
        first_item = results[0]
        
        # Check if it's the new PaddleX OCRResult object or a dictionary-like object
        if hasattr(first_item, 'get') or isinstance(first_item, dict) or (hasattr(first_item, '__getitem__') and not isinstance(first_item, list)):
            try:
                rec_texts = first_item.get('rec_texts', [])
                rec_polys = first_item.get('rec_polys', [])
                rec_scores = first_item.get('rec_scores', [])
                for text, points, score in zip(rec_texts, rec_polys, rec_scores):
                    parsed.append({
                        "text": str(text),
                        "points": np.array(points, dtype=np.int32),
                        "score": float(score)
                    })
                return parsed
            except Exception as e:
                logging.debug(f"Failed to parse new OCR format: {e}")
                
        # Legacy PaddleOCR format: list of [points, (text, score)]
        if isinstance(first_item, list):
            for line in first_item:
                try:
                    if len(line) == 2 and (isinstance(line[0], list) or isinstance(line[0], np.ndarray)) and isinstance(line[1], (tuple, list)):
                        points = np.array(line[0], dtype=np.int32)
                        text = line[1][0]
                        score = line[1][1]
                        parsed.append({
                            "text": str(text),
                            "points": points,
                            "score": float(score)
                        })
                except Exception as e:
                    logging.debug(f"Failed to parse legacy OCR format: {e}")
                    
        return parsed

    def _contains_pii(self, text):
        """Layer 3 Cognitive Verification: Checks if text matches PII properties."""
        if not text or len(text.strip()) < 2:
            return False
            
        # Check clinical allowlist first
        if self._is_clinical_marker(text):
            return False
            
        # 1. Regex checks for standard numeric PII (Aadhaar, Phone, DOB)
        # Aadhaar: 12 digits or spaced 12 digits
        if re.search(r'\b\d{4}\s?\d{4}\s?\d{4}\b', text):
            return True
        # Mobile/Phone: Indian style (10 digits)
        if re.search(r'\b(?:\+91|0)?[6-9]\d{9}\b', text):
            return True
        # Date patterns (DOB / Study Date)
        if re.search(r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b', text):
            return True
            
        # 2. NLP entity check using Presidio
        if self.analyzer:
            results = self.analyzer.analyze(text=text, language="en")
            # If entities like PERSON, DATE, or LOCATION are detected with high confidence
            for result in results:
                if result.entity_type in ["PERSON", "DATE_TIME", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"]:
                    if result.score > 0.45:
                        return True
                        
        # 3. Heuristic checks for text containing names/identifiers
        lower_text = text.lower()
        id_indicators = ["name", "patient", "id:", "uhid", "mrn", "hosp", "dr.", "age:", "sex:", "dob:"]
        if any(indicator in lower_text for indicator in id_indicators):
            return True
            
        # Default: If it's general text not in allowlist, treat as suspect to be safe
        return True

    def sanitize_metadata(self, ds):
        """Layer 1: Metadata Tag Sanitation (using dicomanonymizer if available)"""
        if DICOM_ANONYMIZER_AVAILABLE:
            try:
                from dicomanonymizer.simpledicomanonymizer import anonymize_dataset
                anonymize_dataset(ds)
                logging.info("Metadata anonymized using dicomanonymizer package.")
                return ds
            except Exception as e:
                logging.warning(f"dicomanonymizer failed: {e}. Falling back to manual sanitation.")
                
        # Critical tag de-identification list (Fallback)
        tags_to_clear = [
            'PatientName', 'PatientID', 'PatientBirthDate', 'PatientSex',
            'PatientAge', 'PatientAddress', 'PatientTelephoneNumbers',
            'ReferringPhysicianName', 'InstitutionName', 'InstitutionAddress',
            'StationName', 'AccessionNumber', 'StudyDate', 'SeriesDate',
            'AcquisitionDate', 'ContentDate', 'StudyTime', 'SeriesTime',
            'AcquisitionTime', 'ContentTime', 'PhysiciansOfRecord',
            'PerformingPhysicianName', 'NameOfPhysiciansReadingStudy',
            'OperatorsName', 'AdmittingDiagnosesDescription', 'PatientWeight'
        ]
        
        for attr in tags_to_clear:
            if hasattr(ds, attr):
                # We replace with empty string or dummy labels
                if attr == 'PatientName':
                    ds.PatientName = ""
                elif attr == 'PatientID':
                    ds.PatientID = ""
                else:
                    setattr(ds, attr, "")
                    
        # Replace UIDs to break linkability
        uid_tags = ['StudyInstanceUID', 'SeriesInstanceUID', 'SOPInstanceUID']
        for attr in uid_tags:
            if hasattr(ds, attr):
                setattr(ds, attr, generate_uid())
                
        # Purge vendor-specific private tags (group number is odd)
        ds.remove_private_tags()
        return ds

    def process_pixels(self, ds):
        """Layer 2 & 4: Text Detection and Pixel Reconstruction"""
        if not hasattr(ds, 'pixel_array'):
            return ds, []
            
        pixels = ds.pixel_array.copy()
        raw_min, raw_max = pixels.min(), pixels.max()
        
        # Normalize to 8-bit RGB for OpenCV/PaddleOCR processing
        if raw_max > raw_min:
            norm_img = ((pixels - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
        else:
            norm_img = pixels.astype(np.uint8)
            
        if len(norm_img.shape) == 2:
            rgb_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = norm_img
            
        # Create a black mask for redaction
        mask = np.zeros(rgb_img.shape[:2], dtype=np.uint8)
        redacted_log = []
        
        if self.ocr:
            # Layer 2: Text Boundary Detection (DBNet)
            results = self.ocr.ocr(rgb_img)
            parsed_lines = self._parse_ocr_results(results)
            
            for line in parsed_lines:
                text = line["text"]
                points = line["points"]
                confidence = line["score"]
                
                # Layer 3: Cognitive verification
                if self._contains_pii(text):
                    x, y, w, h = cv2.boundingRect(points)
                    # Add padding to boundary box
                    pad = 4
                    y1 = max(0, y - pad)
                    y2 = min(rgb_img.shape[0], y + h + pad)
                    x1 = max(0, x - pad)
                    x2 = min(rgb_img.shape[1], x + w + pad)
                    
                    # Add to mask
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    redacted_log.append({
                        "text": text,
                        "box": [x1, y1, x2, y2],
                        "confidence": float(confidence)
                    })
                        
        # Layer 4: Pixel Reconstruction / Inpainting
        if np.any(mask > 0):
            # Perform Telea inpainting on normalized array
            norm_img = cv2.inpaint(norm_img, mask, 3, cv2.INPAINT_TELEA)
            
            # Map back to original DICOM dynamic range scale
            if raw_max > raw_min:
                reconstructed = (norm_img.astype(float) / 255.0 * (raw_max - raw_min) + raw_min).astype(pixels.dtype)
            else:
                reconstructed = norm_img.astype(pixels.dtype)
                
            # Write pixel modifications back
            ds.PixelData = reconstructed.tobytes()
            
        return ds, redacted_log

    def verify_leakage(self, ds):
        """Layer 5: Dual-Pass Leakage Verification"""
        if not hasattr(ds, 'pixel_array') or not self.ocr:
            return "PASSED (No pixels or OCR bypassed)"
            
        pixels = ds.pixel_array
        raw_min, raw_max = pixels.min(), pixels.max()
        if raw_max > raw_min:
            norm_img = ((pixels - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
        else:
            norm_img = pixels.astype(np.uint8)
            
        if len(norm_img.shape) == 2:
            rgb_img = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = norm_img
            
        # Re-run OCR on the processed image
        results = self.ocr.ocr(rgb_img)
        parsed_lines = self._parse_ocr_results(results)
        
        leakage = []
        for line in parsed_lines:
            text = line["text"]
            # If we still detect suspicious text that is not a clinical marker
            if self._contains_pii(text):
                leakage.append(text)
                    
        if leakage:
            return f"FAILED (Suspicious text still detected: {leakage})"
        return "PASSED"

    def anonymize_file(self, input_path, output_path):
        """Ties all 5 layers together to process a single file."""
        logging.info(f"Processing DICOM: {input_path}")
        ds = pydicom.dcmread(input_path)
        
        # Layer 1
        ds = self.sanitize_metadata(ds)
        
        # Layer 2, 3, 4
        ds, log = self.process_pixels(ds)
        
        # Layer 5
        status = self.verify_leakage(ds)
        
        # Write back changes
        ds.save_as(output_path)
        logging.info(f"Saved anonymized DICOM: {output_path} | Audit Status: {status}")
        
        audit_report = {
            "file": os.path.basename(input_path),
            "redacted_items": log,
            "verification_status": status
        }
        return audit_report

# ================================================================================
# TEST RUN
# ================================================================================
if __name__ == "__main__":
    # Initialize the anonymizer
    anonymizer = HybridDicomAnonymizer(use_gpu=False)
    
    # Define test files
    sample_dir = "./dicom_samples"
    os.makedirs(sample_dir, exist_ok=True)
    
    # We will look for our previously saved CT sample in step 6
    test_file = os.path.join(sample_dir, "anonymized_ct_sample.dcm")
    output_file = os.path.join(sample_dir, "fully_anonymized_ct.dcm")
    
    if os.path.exists(test_file):
        report = anonymizer.anonymize_file(test_file, output_file)
        print("\n" + "=" * 50)
        print("ANONYMIZATION AUDIT REPORT:")
        print("=" * 50)
        print(json.dumps(report, indent=4))
        print("=" * 50)
    else:
        print(f"Test file {test_file} not found. Please run explore_dicom.py first.")
