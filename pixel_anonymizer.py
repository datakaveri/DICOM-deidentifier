import os
import re
import logging
import numpy as np
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Handle PaddleOCR imports
try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False

# Handle EasyOCR imports
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# Handle Presidio Analyzer imports
try:
    from presidio_analyzer import AnalyzerEngine
    PRESIDIO_AVAILABLE = True
except ImportError:
    PRESIDIO_AVAILABLE = False


class PixelAnonymizer:
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        
        # Initialize PaddleOCR
        if PADDLE_AVAILABLE:
            try:
                # Disable mkldnn to avoid potential CPU execution issues
                os.environ["FLAGS_use_mkldnn"] = "0"
                os.environ["DISABLE_AUTO_LOGGING_CONFIG"] = "1"
                from paddleocr import logger as paddle_logger
                paddle_logger.setLevel(logging.ERROR)
            except Exception:
                pass
            if self.use_gpu:
                try:
                    import paddle
                    paddle.device.set_device('gpu')
                except Exception:
                    pass
            self.paddle_ocr = PaddleOCR(lang='en', enable_mkldnn=False)
            logging.info("PaddleOCR initialized successfully.")
        else:
            self.paddle_ocr = None
            logging.warning("PaddleOCR not found. Skipping PaddleOCR detections.")

        # Initialize EasyOCR
        if EASYOCR_AVAILABLE:
            self.easy_ocr = easyocr.Reader(['en'], gpu=self.use_gpu)
            logging.info("EasyOCR initialized successfully.")
        else:
            self.easy_ocr = None
            logging.warning("EasyOCR not found. Skipping EasyOCR detections.")

        # Initialize Presidio NLP Analyzer
        if PRESIDIO_AVAILABLE:
            self.analyzer = AnalyzerEngine()
            logging.info("Presidio NLP Analyzer initialized successfully.")
        else:
            self.analyzer = None
            logging.warning("Presidio Analyzer not found. Falling back to pattern-only classification.")

        # Clinical Allowlist: Clinical labels, technical details, or directions to preserve
        self.clinical_allowlist = {
            "L", "R", "LT", "RT", "LEFT", "RIGHT",
            "PA", "AP", "LAT", "LL", "RL", "LATERAL",
            "ERECT", "SUPINE", "PRONE", "DECUBITUS", "UPRIGHT",
            "PORTABLE", "MOBILE", "STAT", "ROUTINE",
            "CHEST", "ABDOMEN", "PELVIS", "SKULL", "SPINE",
            "KVP", "MAS", "MA", "SEC", "CM", "MM", "FOV"
        }

        # Indian Health IDs & Standard PII Patterns
        self.pii_patterns = {
            "aadhaar": r'\b\d{4}\s?\d{4}\s?\d{4}\b',                     # Aadhaar (12 digits)
            "abha": r'\b\d{2}-\d{4}-\d{4}-\d{4}\b',                       # ABHA ID (Ayushman Bharat)
            "phone": r'\b(?:\+91|0)?[6-9]\d{9}\b',                        # Indian Mobile Numbers
            "date": r'\b\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}\b',               # Standard Date formats
            "uhid_mrn": r'\b(?:UHID|MRN|REG|IPD|OPD|CR)[\s:/-]?\d+\b',    # Medical registration IDs
            "age_sex": r'\b\d{1,3}\s*[/]\s*[MFO]\b',                     # "45/M", "28/F" Age/Sex markers
            "name_prefix": r'\b(?:DR\.?|MR\.?|MRS\.?|MS\.?|SHRI|SMT|S/O|D/O|W/O)\b' # Names/relations
        }

    # =========================================================================
    # STAGE 1: IMAGE ENHANCEMENT
    # =========================================================================
    def enhance_image(self, image_array, modality="CR"):
        """
        Generates 5 enhanced variants of the image to maximize OCR text visibility.
        Input image can be 8-bit or 16-bit.
        """
        # Ensure image is 8-bit normalized for OCR engines
        img_min, img_max = image_array.min(), image_array.max()
        if img_max > img_min:
            norm_img = ((image_array - img_min) / (img_max - img_min) * 255.0).astype(np.uint8)
        else:
            norm_img = image_array.astype(np.uint8)

        variants = {"standard": norm_img}

        # 1. CLAHE (Local Contrast Stretch)
        try:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            variants["clahe"] = clahe.apply(norm_img)
        except Exception as e:
            logging.debug(f"CLAHE failed: {e}")

        # 2. Adaptive Threshold (Binary text separation)
        try:
            variants["adaptive_thresh"] = cv2.adaptiveThreshold(
                norm_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 15, 5
            )
        except Exception as e:
            logging.debug(f"Adaptive Threshold failed: {e}")

        # 3. Inverse Threshold (Dark text on bright background)
        try:
            _, thresh_inv = cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            variants["inverse_thresh"] = thresh_inv
        except Exception as e:
            logging.debug(f"Inverse Threshold failed: {e}")

        # 4. Gamma Correction (Brightens shadow areas)
        try:
            gamma = 0.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            variants["gamma"] = cv2.LUT(norm_img, table)
        except Exception as e:
            logging.debug(f"Gamma Correction failed: {e}")

        # 5. Sharpening (Unsharp mask for blurry/scanned text)
        try:
            blurred = cv2.GaussianBlur(norm_img, (3, 3), 0)
            variants["sharpened"] = cv2.addWeighted(norm_img, 1.5, blurred, -0.5, 0)
        except Exception as e:
            logging.debug(f"Sharpening failed: {e}")

        return variants

    # =========================================================================
    # STAGE 2: OCR TEXT DETECTION
    # =========================================================================
    def detect_text(self, variants):
        """
        Runs the active OCR engines on all generated image variants.
        """
        raw_detections = []

        for name, img in variants.items():
            # Convert single channel to RGB if needed by engines
            if len(img.shape) == 2:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            else:
                rgb_img = img

            # 1. PaddleOCR Pass
            if self.paddle_ocr:
                try:
                    result = self.paddle_ocr.ocr(rgb_img, cls=False)
                    if result and result[0]:
                        for line in result[0]:
                            box = line[0]  # List of 4 points [[x, y], ...]
                            text = line[1][0]
                            conf = line[1][1]
                            
                            x_coords = [p[0] for p in box]
                            y_coords = [p[1] for p in box]
                            bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                            
                            raw_detections.append({
                                "text": text.strip(),
                                "bbox": bbox,
                                "confidence": float(conf),
                                "variant": name,
                                "engine": "paddle"
                            })
                except Exception as e:
                    logging.debug(f"PaddleOCR detection failed on variant {name}: {e}")

            # 2. EasyOCR Pass
            if self.easy_ocr:
                try:
                    results = self.easy_ocr.readtext(rgb_img)
                    for (bbox_pts, text, conf) in results:
                        x_coords = [p[0] for p in bbox_pts]
                        y_coords = [p[1] for p in bbox_pts]
                        bbox = [int(min(x_coords)), int(min(y_coords)), int(max(x_coords)), int(max(y_coords))]
                        
                        raw_detections.append({
                            "text": text.strip(),
                            "bbox": bbox,
                            "confidence": float(conf),
                            "variant": name,
                            "engine": "easyocr"
                        })
                except Exception as e:
                    logging.debug(f"EasyOCR detection failed on variant {name}: {e}")

        return raw_detections

    # =========================================================================
    # STAGE 3: MERGING & DEDUPLICATION (IoU + Non-Maximum Suppression)
    # =========================================================================
    def _compute_iou(self, boxA, boxB):
        """Computes Intersection over Union (IoU) of two bounding boxes."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        unionArea = float(boxAArea + boxBArea - interArea)
        if unionArea == 0:
            return 0.0
        return interArea / unionArea

    def merge_detections(self, raw_detections, iou_threshold=0.3):
        """
        Groups duplicate boxes, merges coordinates, and performs confidence fusion.
        """
        if not raw_detections:
            return []

        # Sort detections by confidence score descending
        sorted_det = sorted(raw_detections, key=lambda x: x["confidence"], reverse=True)
        merged = []

        while len(sorted_det) > 0:
            current = sorted_det.pop(0)
            overlapping = [current]
            
            # Find all other boxes that overlap significantly with this one
            remaining = []
            for det in sorted_det:
                if self._compute_iou(current["bbox"], det["bbox"]) > iou_threshold:
                    overlapping.append(det)
                else:
                    remaining.append(det)
            sorted_det = remaining

            # Merge overlapping boxes
            bboxes = np.array([o["bbox"] for o in overlapping])
            x1 = int(np.min(bboxes[:, 0]))
            y1 = int(np.min(bboxes[:, 1]))
            x2 = int(np.max(bboxes[:, 2]))
            y2 = int(np.max(bboxes[:, 3]))

            # Find the best text description (prefer the longest string or highest confidence)
            best_text = max(overlapping, key=lambda x: len(x["text"]))["text"]
            max_conf = max(o["confidence"] for o in overlapping)
            
            # Weighted confidence based on how many variants detected this block
            variant_count = len(set(o["variant"] for o in overlapping))
            fused_conf = max_conf * (variant_count / 5.0)  # normalized by max 5 variants

            merged.append({
                "text": best_text,
                "bbox": [x1, y1, x2, y2],
                "confidence": min(1.0, max(fused_conf, max_conf * 0.5)),
                "variant_count": variant_count
            })

        return merged

    # =========================================================================
    # STAGE 4: CLASSIFICATION (ALLOWLIST + REGEX + PRESIDIO NLP + SPATIAL)
    # =========================================================================
    def _is_clinical_term(self, text):
        clean_text = re.sub(r'[^A-Z]', '', text.upper())
        return clean_text in self.clinical_allowlist

    def classify_text(self, merged_detections, image_shape):
        """
        Decides whether each merged detection represents PHI or is safe to keep.
        """
        h, w = image_shape[:2]
        phi_regions = []

        for det in merged_detections:
            text = det["text"]
            bbox = det["bbox"]
            
            # Check Clinical Allowlist first
            if self._is_clinical_term(text):
                logging.info(f"Preserving clinical marker: '{text}' at {bbox}")
                continue

            # 1. Regex PII Check
            regex_match = False
            for pattern_name, pattern in self.pii_patterns.items():
                if re.search(pattern, text, re.IGNORECASE):
                    regex_match = True
                    break

            # 2. NLP Entity Check (Presidio)
            nlp_match = False
            if self.analyzer:
                try:
                    results = self.analyzer.analyze(text=text, language="en")
                    for result in results:
                        if result.entity_type in ["PERSON", "DATE_TIME", "LOCATION", "EMAIL_ADDRESS", "PHONE_NUMBER"]:
                            if result.score > 0.45:
                                nlp_match = True
                                break
                except Exception:
                    pass

            # 3. Spatial Heuristic (Is it in top 15% or bottom 15% border?)
            y_mid = (bbox[1] + bbox[3]) / 2.0
            x_mid = (bbox[0] + bbox[2]) / 2.0
            
            in_top_border = y_mid < (h * 0.15)
            in_bottom_border = y_mid > (h * 0.85)
            in_left_border = x_mid < (w * 0.10)
            in_right_border = x_mid > (w * 0.90)
            
            is_in_border = in_top_border or in_bottom_border or in_left_border or in_right_border

            # Decision Logic: Redact if it matches Regex/NLP, OR if it's unknown text in the borders
            if regex_match or nlp_match or is_in_border:
                logging.info(f"Flagged for Redaction: '{text}' at {bbox} | Regex={regex_match}, NLP={nlp_match}, Border={is_in_border}")
                phi_regions.append({
                    "text": text,
                    "bbox": bbox,
                    "zone": "border" if is_in_border else "anatomy"
                })
            else:
                logging.info(f"Preserved text: '{text}' at {bbox}")

        return phi_regions

    # =========================================================================
    # STAGE 5: PRECISE INK REDACTION / MASKING
    # =========================================================================
    def redact_image(self, image_array, phi_regions):
        """
        Redacts ONLY the letters/ink of the flagged text.
        Fills the text strokes with local background matching or Navier-Stokes inpainting.
        Works directly on native bit depth (8-bit or 16-bit).
        """
        cleaned_image = image_array.copy()
        h, w = cleaned_image.shape[:2]
        
        # Create a binary mask where 255 = text pixels to redact
        redaction_mask = np.zeros((h, w), dtype=np.uint8)

        for region in phi_regions:
            x1, y1, x2, y2 = region["bbox"]
            
            # Keep bounds safe
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            if (x2 - x1) <= 0 or (y2 - y1) <= 0:
                continue

            # Extract local region of interest (ROI)
            roi = cleaned_image[y1:y2, x1:x2]
            
            # Normalize ROI to 8-bit for thresholding
            roi_min, roi_max = roi.min(), roi.max()
            if roi_max > roi_min:
                roi_8bit = ((roi - roi_min) / (roi_max - roi_min) * 255.0).astype(np.uint8)
            else:
                roi_8bit = roi.astype(np.uint8)

            # Local binarization to find the ink strokes of the characters
            _, local_mask = cv2.threshold(roi_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Invert mask if background is brighter than the text
            if np.mean(roi_8bit) > 127:
                local_mask = cv2.bitwise_not(local_mask)

            # Dilate mask slightly (1-2px) to capture edges/outlines cleanly
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            local_mask_dilated = cv2.dilate(local_mask, kernel, iterations=1)
            
            # Add to global mask
            redaction_mask[y1:y2, x1:x2] = cv2.bitwise_or(redaction_mask[y1:y2, x1:x2], local_mask_dilated)

        # Apply precise masking on the final image
        if np.any(redaction_mask > 0):
            # For 16-bit images, we need to convert to 8-bit to run cv2.inpaint
            if cleaned_image.dtype == np.uint16:
                raw_min, raw_max = cleaned_image.min(), cleaned_image.max()
                if raw_max > raw_min:
                    temp_8bit = ((cleaned_image - raw_min) / (raw_max - raw_min) * 255.0).astype(np.uint8)
                    inpainted_8bit = cv2.inpaint(temp_8bit, redaction_mask, 3, cv2.INPAINT_NS)
                    # Convert back to original 16-bit dynamic range
                    cleaned_image = (inpainted_8bit.astype(float) / 255.0 * (raw_max - raw_min) + raw_min).astype(np.uint16)
                else:
                    cleaned_image = cv2.inpaint(cleaned_image.astype(np.uint8), redaction_mask, 3, cv2.INPAINT_NS).astype(np.uint16)
            else:
                # Direct 8-bit inpainting
                cleaned_image = cv2.inpaint(cleaned_image, redaction_mask, 3, cv2.INPAINT_NS)

        return cleaned_image, redaction_mask

    # =========================================================================
    # STAGE 6: VERIFICATION & RETRY ESCALATION
    # =========================================================================
    def verify_image(self, redacted_image, mask, phi_regions):
        """
        Verifies that no text remains. If residual text is found, applies an expanded
        dilated mask, or falls back to full box background fill.
        """
        # Run OCR on the redacted image
        test_variants = self.enhance_image(redacted_image)
        test_detections = self.detect_text(test_variants)
        test_merged = self.merge_detections(test_detections)
        test_phi = self.classify_text(test_merged, redacted_image.shape)

        if not test_phi:
            return redacted_image, "PASSED"

        # Escalation: If text is still detected, fall back to box background fill for those zones
        h, w = redacted_image.shape[:2]
        escalated_image = redacted_image.copy()

        logging.warning(f"Verification FAILED. Residual text detected: {[t['text'] for t in test_phi]}. Escalating to Box Redaction.")

        for region in test_phi:
            x1, y1, x2, y2 = region["bbox"]
            x1, y1 = max(0, x1 - 5), max(0, y1 - 5)
            x2, y2 = min(w, x2 + 5), min(h, y2 + 5)

            # Background fill box: Fill with the local median
            pad = 20
            roi_box = escalated_image[max(0, y1-pad):min(h, y2+pad), max(0, x1-pad):min(w, x2+pad)]
            median_val = np.median(roi_box)
            escalated_image[y1:y2, x1:x2] = median_val

        # Final check on escalated image
        final_variants = self.enhance_image(escalated_image)
        final_detections = self.detect_text(final_variants)
        final_phi = self.classify_text(final_detections, escalated_image.shape)

        if final_phi:
            # Nuclear Option: Black out the top 15% and bottom 15% margins
            logging.error("Escalation failed to clear text. Applying Nuclear Option (Border Blackout).")
            escalated_image[0:int(h * 0.15), :] = 0
            escalated_image[int(h * 0.85):, :] = 0
            return escalated_image, "FORCE PASSED (Nuclear Fallback)"

        return escalated_image, "PASSED (Escalated)"

    def anonymize(self, image_array, modality="CR"):
        """Main interface function."""
        # 1. Enhance
        variants = self.enhance_image(image_array, modality)
        # 2. Detect
        raw_detections = self.detect_text(variants)
        # 3. Merge
        merged_detections = self.merge_detections(raw_detections)
        # 4. Classify
        phi_regions = self.classify_text(merged_detections, image_array.shape)
        # 5. Redact
        redacted_image, mask = self.redact_image(image_array, phi_regions)
        # 6. Verify
        final_image, status = self.verify_image(redacted_image, mask, phi_regions)
        
        return final_image, status


# =========================================================================
# TEST RUNNER (Synthetic test image)
# =========================================================================
if __name__ == "__main__":
    print("Initializing test run...")
    # Create a synthetic 16-bit grayscale image representing an X-ray
    # Size 512x512, background value 500
    img = np.full((512, 512), 500, dtype=np.uint16)
    
    # Simulate a rib/bone block (value 3000)
    img[150:200, 100:400] = 3000
    
    # Write some simulated text using OpenCV
    # We will write patient info in the border (top)
    cv2.putText(img, "PATIENT NAME: RAJ KUMAR", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 4000, 2)
    # We will write a clinical marker "L" (should be kept)
    cv2.putText(img, "L", (450, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 4000, 3)

    # Instantiate anonymizer
    anonymizer = PixelAnonymizer(use_gpu=False)
    
    # Run anonymization
    print("Running anonymization...")
    clean_img, status = anonymizer.anonymize(img, modality="CR")
    
    print("\n" + "=" * 50)
    print(f"ANONYMIZATION STATUS: {status}")
    print("=" * 50)
    
    # Save test output images so the user can visually verify the difference
    os.makedirs("./test_outputs", exist_ok=True)
    
    # Normalize images for saving to PNG
    img_save = ((img - img.min()) / (img.max() - img.min()) * 255.0).astype(np.uint8)
    clean_save = ((clean_img - clean_img.min()) / (clean_img.max() - clean_img.min()) * 255.0).astype(np.uint8)
    
    cv2.imwrite("./test_outputs/original_synthetic.png", img_save)
    cv2.imwrite("./test_outputs/anonymized_synthetic.png", clean_save)
    print("Saved test comparison images to: ./test_outputs/")
