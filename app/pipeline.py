"""
pipeline.py — full 7-stage anonymization pipeline for a single DICOM file.
Wires together metadata sanitization, OCR-based PHI detection/classification,
tag cross-checking, pixel redaction, verification, and DICOM write-back.
"""

import os
import datetime

import numpy as np
import pydicom

from config import log, ENABLE_TAG_DEIDENTIFICATION, BBOX_IMAGE_NAME
from text_region_detect import detect_text_regions
from ocr_detect import detect_text_in_regions
from classify import merge_detections, classify_phi, expand_phi_blocks, _iou
from phi_tags import load_original_tag_values, match_against_stored_tags
from masking import redact_pixels
from verify import verify_redaction
from dicom_io import write_pixels_to_dicom
from de_identification.deidentify import deidentify_dataset
from bbox_visualize import save_bbox_image


def anonymize_dicom_file(input_path, before_output_path, output_path,
                          data_snapshot_path, paddle_ocr, easy_ocr, analyzer,
                          keystore):
    """
    Full 7-stage anonymization pipeline for a single DICOM file: burned-in
    pixel/OCR redaction (checkpointed to before_output_path, tags still
    original), then tag-level de-identification (hash/mask/suppress per
    de_identification/tag_mapping.py) as the last step, saved to output_path.

    `keystore` should be one shared KeyStore instance reused across every
    file in a batch, so hash/tokenise/encrypt values stay consistent across
    the whole set. `data_snapshot_path` is this file's own original-tag
    snapshot (per phi_tags.dump_original_tags), used for the Step 3
    burned-in-text cross-check.

    Returns an audit dict.
    """
    filename = os.path.basename(input_path)
    log.info(f"\n{'='*60}")
    log.info(f"Processing: {filename}")
    log.info(f"{'='*60}")

    audit = {
        "file": filename,
        "input_path":  input_path,
        "before_output_path": before_output_path,
        "output_path": output_path,
        "timestamp":   datetime.datetime.now().isoformat(),
        "modality":    "Unknown",
        "image_size":  "Unknown",
        "redacted_regions": [],
        "deidentified_tags": [],
        "verification_status": "NOT RUN",
        "error": None
    }

    try:
        # ── Read DICOM ────────────────────────────────────────────────────────
        ds = pydicom.dcmread(input_path, force=True)

        audit["modality"] = getattr(ds, "Modality", "Unknown")
        rows = getattr(ds, "Rows", "?")
        cols = getattr(ds, "Columns", "?")
        audit["image_size"] = f"{rows}x{cols}"

        # ── Extract pixels ────────────────────────────────────────────────────
        if not hasattr(ds, 'pixel_array'):
            log.warning(f"  No pixel data in {filename}. Saving metadata-only.")
            os.makedirs(os.path.dirname(before_output_path), exist_ok=True)
            ds.save_as(before_output_path, write_like_original=False)
            audit["deidentified_tags"] = deidentify_dataset(ds, keystore)
            ds.save_as(output_path, write_like_original=False)
            audit["verification_status"] = "SKIPPED (no pixels)"
            return audit

        pixels = ds.pixel_array.copy()
        original_dtype = pixels.dtype

        # Strip overlay planes (group 60xx) if present
        overlay_groups = [g for g in range(0x6000, 0x601F, 2)]
        for group in overlay_groups:
            for elem in list(ds):
                if elem.tag.group == group:
                    del ds[elem.tag]

        # Fix MONOCHROME1 inversion (dark=bright for OCR correctness)
        photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
        is_monochrome1 = (photometric == "MONOCHROME1")
        original_max = None
        if is_monochrome1:
            original_max = np.max(pixels)
            pixels = original_max - pixels
            log.info("  MONOCHROME1 detected — pixel values flipped.")

        # Normalize to 8-bit for OCR pipeline
        pix_min, pix_max = pixels.min(), pixels.max()
        if pix_max > pix_min:
            norm_8 = ((pixels - pix_min) / (pix_max - pix_min) * 255.0).astype(np.uint8)
        else:
            norm_8 = pixels.astype(np.uint8)

        # Handle multi-frame: process frame 0 for OCR detection, apply to all frames
        if pixels.ndim == 3:
            ocr_frame = norm_8[0]
            ocr_pixels = pixels[0]
        else:
            ocr_frame = norm_8
            ocr_pixels = pixels

        # ── Stage 2: Text region detection (shape-based, no OCR) ──────────────
        log.info("  [Stage 2] Detecting candidate text regions...")
        text_regions = detect_text_regions(ocr_frame)
        log.info(f"            Candidate regions: {len(text_regions)}")

        # ── Stage 3: OCR ──────────────────────────────────────────────────────
        log.info("  [Stage 3] Running OCR text detection on candidate regions...")
        raw_det = detect_text_in_regions(ocr_frame, text_regions, paddle_ocr, easy_ocr)
        log.info(f"            Raw detections: {len(raw_det)}")

        # ── Stage 4: Classify ─────────────────────────────────────────────────
        log.info("  [Stage 4] Classifying detections (PHI vs clinical)...")
        merged     = merge_detections(raw_det)
        phi_regions = classify_phi(merged, ocr_frame.shape, analyzer)
        phi_regions = expand_phi_blocks(merged, phi_regions, ocr_frame.shape)
        log.info(f"            PHI regions to redact: {len(phi_regions)}")

        # ── Step 3: match OCR text against the full original-tag backup ────────
        log.info("  [Step 3] Cross-checking OCR text against data.json (original tag values)...")
        stored_values = load_original_tag_values(data_snapshot_path)
        tag_matches = match_against_stored_tags(merged, stored_values, ocr_frame.shape)
        existing_bboxes = [r["bbox"] for r in phi_regions]
        added = 0
        for m in tag_matches:
            if not any(_iou(m["bbox"], b) > 0.3 for b in existing_bboxes):
                phi_regions.append(m)
                existing_bboxes.append(m["bbox"])
                added += 1
        log.info(f"            Additional regions matched to stored tags: {added}")

        audit["redacted_regions"] = [
            {"text": r["text"], "bbox": r["bbox"]} for r in phi_regions
        ]

        # ── Bbox visualization: save the frame with PHI regions boxed ─────────
        bbox_output_path = os.path.join(os.path.dirname(before_output_path), BBOX_IMAGE_NAME)
        save_bbox_image(ocr_frame, phi_regions, bbox_output_path)
        audit["bbox_image_path"] = bbox_output_path

        # ── Stage 5: Redact ───────────────────────────────────────────────────
        log.info("  [Stage 5] Redacting PHI pixels (Navier-Stokes inpainting)...")
        if pixels.ndim == 3:
            # Apply to every frame
            cleaned_frames = []
            for i in range(pixels.shape[0]):
                frame_pix = pixels[i]
                frame_phi = phi_regions  # same regions for all frames
                cleaned_frame, _ = redact_pixels(frame_pix, frame_phi)
                cleaned_frames.append(cleaned_frame)
            cleaned_pixels = np.stack(cleaned_frames, axis=0)
        else:
            cleaned_pixels, _ = redact_pixels(pixels, phi_regions)

        # ── Stage 6: Verify ───────────────────────────────────────────────────
        verify_frame = cleaned_pixels[0] if cleaned_pixels.ndim == 3 else cleaned_pixels
        cleaned_pixels_final = cleaned_pixels.copy()
        verify_clean, status = verify_redaction(
            verify_frame, phi_regions, paddle_ocr, easy_ocr, analyzer
        )
        if cleaned_pixels.ndim == 3:
            cleaned_pixels_final[0] = verify_clean
        else:
            cleaned_pixels_final = verify_clean
        audit["verification_status"] = status

        # ── Stage 7: Write back to DICOM ─────────────────────────────────────
        log.info("  [Stage 7] Writing anonymized pixels back into DICOM dataset...")
        if is_monochrome1 and original_max is not None:
            cleaned_pixels_final = original_max - cleaned_pixels_final
            log.info("  Flipped pixels back to MONOCHROME1 representation.")
        ds = write_pixels_to_dicom(ds, cleaned_pixels_final)

        # ── Checkpoint: pixels redacted, tags still original ─────────────────
        os.makedirs(os.path.dirname(before_output_path), exist_ok=True)
        ds.save_as(before_output_path, write_like_original=False)
        log.info(f"  [Checkpoint] Pixel-redacted DICOM saved (tags still original): {before_output_path}")

        # ── Last: tag de-identification (hash PatientID, mask dates, etc.) ────
        if ENABLE_TAG_DEIDENTIFICATION:
            audit["deidentified_tags"] = deidentify_dataset(ds, keystore)
        else:
            log.info("  Tag de-identification skipped (DICOM header tags preserved 100% untouched).")
            audit["deidentified_tags"] = []

        # ── Save as DICOM ONLY ────────────────────────────────────────────────
        ds.save_as(output_path, write_like_original=False)

        # Confirm output is .dcm
        assert output_path.lower().endswith(".dcm"), "BUG: output path must be .dcm!"
        log.info(f"  [DONE] Anonymized DICOM saved: {output_path}")
        log.info(f"         Status: {status}")

    except Exception as e:
        log.error(f"  [ERROR] {filename}: {e}", exc_info=True)
        audit["error"] = str(e)
        audit["verification_status"] = "ERROR"

    return audit
