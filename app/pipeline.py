"""
pipeline.py — full 7-stage anonymization pipeline for a single DICOM file.
Wires together metadata sanitization, OCR-based PHI detection/classification,
tag cross-checking, pixel redaction, verification, and DICOM write-back.

It runs as two stages so a batch can be parallelised safely:

  redact_file_pixels   OCR, classification, redaction and verification (all
                       the model work). Independent per file, so it runs in
                       worker processes. Writes before_deidentification.dcm.
  deidentify_file_tags Tag de-identification against the shared KeyStore. It
                       mints sequential tokens and per-column keys, so it runs
                       in the parent process, one file at a time, in input
                       order. The KeyStore then ends up exactly as a serial
                       run would leave it.

anonymize_dicom_file runs both stages back to back.

`job_config` (de_identification/job_config.py) selects the policy:
  None       fixed tag policy (tag_mapping.py), Navier-Stokes inpainting,
             verification escalates and reports PASSED (Escalated).
  JobConfig  the user's per-tag choices from the SPIDEr UI, a solid black
             fill, strict verification (can report FAILED), and an
             unparsable file fails instead of being read with force=True.
"""

import os
import datetime

import numpy as np
import pydicom

from config import (
    log,
    ENABLE_TAG_DEIDENTIFICATION,
    BBOX_IMAGE_NAME,
    SAVE_BBOX_PREVIEW,
)
from ocr_detect import detect_text_in_regions
from classify import merge_detections, classify_phi, expand_phi_blocks, _iou
from phi_tags import load_original_tag_values, match_against_stored_tags
from masking import redact_pixels
from verify import verify_redaction
from dicom_io import (
    decompress_pixels,
    mask_embedded_overlay_bits,
    strip_overlays,
    write_pixels_to_dicom,
)
from de_identification.deidentify import (
    deidentify_dataset,
    deidentify_dataset_from_config,
)
from bbox_visualize import save_bbox_image
import shared_ner
from file_errors import FileError


def _shared_ner_phi_texts(merged, job_config, ner=None):
    """The OCR texts the shared Anamika NER engine finds personal data in, or
    None when this run did not ask for it (dicom_deidentify.shared_ner
    .pixel_text) or the engine could not be asked -- classify_phi then decides
    exactly as it always has. Added 2026-11-23; see shared_ner.py."""
    if job_config is None or not job_config.shared_ner.pixel_text:
        return None
    texts = sorted({d["text"].strip() for d in merged if d["text"].strip()})
    if not texts:
        return None
    settings = job_config.shared_ner
    # `ner` is an injected client (single_file.Engines.ner); None keeps the
    # unix-socket client, exactly as before.
    found = shared_ner.resolve_client(ner, settings).spans(texts)
    if found is None:
        log.info("            Shared NER unavailable: own classifier only.")
        return None
    flagged = {t for t, spans in zip(texts, found) if spans}
    log.info(f"            Shared NER flagged {len(flagged)} of {len(texts)} text(s).")
    return flagged


def anonymize_dicom_file(
    input_path,
    before_output_path,
    output_path,
    data_snapshot_path,
    paddle_ocr,
    analyzer,
    keystore,
    deid_model=None,
    medical_ner=None,
    gliner_model=None,
    job_config=None,
    secrets=None,
    ner=None,
):
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
    audit = redact_file_pixels(
        input_path,
        before_output_path,
        output_path,
        data_snapshot_path,
        paddle_ocr,
        analyzer,
        deid_model=deid_model,
        medical_ner=medical_ner,
        gliner_model=gliner_model,
        job_config=job_config,
        ner=ner,
    )
    return deidentify_file_tags(audit, keystore, job_config, secrets, ner=ner)


def read_dicom(input_path, job_config=None):
    """Reads `input_path`. force=True lets a batch survive a file without the
    DICM preamble. Under a job config (fail_on_unparsable) it is off: force
    turns an unparsable file into a near-empty dataset that passes every
    stage and gets written out un-de-identified."""
    fail_on_unparsable = job_config.fail_on_unparsable if job_config else False
    ds = pydicom.dcmread(input_path, force=not fail_on_unparsable)
    if fail_on_unparsable and "SOPClassUID" not in ds:
        raise ValueError(
            "parsed no SOPClassUID: not a readable DICOM; refusing to emit it"
        )
    return ds


class PixelArtifacts:
    """Arrays the pixel stage computed that only the file-writing CLI wrapper
    uses (the opt-in bbox preview and after_preview.png)."""

    def __init__(self, ocr_frame, phi_regions, cleaned_pixels):
        self.ocr_frame = ocr_frame
        self.phi_regions = phi_regions
        self.cleaned_pixels = cleaned_pixels


def new_audit(filename, input_path=None, before_output_path=None, output_path=None):
    """The audit dict both entry points start from."""
    return {
        "file": filename,
        "input_path": input_path,
        "before_output_path": before_output_path,
        "output_path": output_path,
        "timestamp": datetime.datetime.now().isoformat(),
        "modality": "Unknown",
        "image_size": "Unknown",
        "redacted_regions": [],
        "deidentified_tags": [],
        "verification_status": "NOT RUN",
        "error": None,
        # Set once before_output_path is written; tells deidentify_file_tags
        # there is something to de-identify.
        "pixel_stage_complete": False,
    }


def redact_dataset_pixels(
    ds,
    filename,
    audit,
    paddle_ocr,
    analyzer,
    stored_values,
    deid_model=None,
    medical_ner=None,
    gliner_model=None,
    job_config=None,
    ner=None,
):
    """
    The pixel stage on an already-read dataset, with no file access: OCR,
    PHI classification, redaction and verification. Shared by the CLI
    (redact_file_pixels, which adds the checkpoint file) and by
    single_file.process_file (which adds nothing, so a Ray actor can call it
    per file). `stored_values` are the file's original tag values to
    cross-check burned-in text against (phi_tags.original_tag_values).

    Fills `audit`; raises on failure (FileError carries a stable code) so
    each caller decides how to report it. Returns (ds, PixelArtifacts); the
    artifacts are None for a file with no pixel data.
    """
    log.info(f"\n{'='*60}")
    log.info(f"Processing: {filename}")
    log.info(f"{'='*60}")

    pixel_method = job_config.pixel["method"] if job_config else "inpaint"
    strict_verify = job_config is not None

    audit["modality"] = getattr(ds, "Modality", "Unknown")
    rows = getattr(ds, "Rows", "?")
    cols = getattr(ds, "Columns", "?")
    audit["image_size"] = f"{rows}x{cols}"

    # ── Decode compressed pixel data, before any pixel is touched ────────
    # JPEG Lossless (PACS X-rays), JPEG baseline, JPEG-LS, JPEG 2000 and
    # RLE are decoded here and the file becomes uncompressed. Writing
    # redacted pixels into a file still marked as JPEG would corrupt it.
    source_ts = decompress_pixels(ds)
    if source_ts is not None:
        audit["source_transfer_syntax"] = str(source_ts)
        log.info(f"  Decoded {source_ts.name}; output will be uncompressed.")

    # ── Extract pixels ────────────────────────────────────────────────────
    if "PixelData" not in ds:
        # Rows/Columns say an image was meant to be here. pydicom drops a
        # Pixel Data element cut off mid-file (force=True), so this is a
        # truncated or corrupt file, not a pixel-less object (SR, KO, PR).
        if "Rows" in ds or "Columns" in ds:
            raise FileError(
                "NO_PIXEL_DATA",
                "image attributes present but no readable pixel data "
                "(truncated or corrupt file)",
            )
        log.warning(f"  No pixel data in {filename}. Saving metadata-only.")
        audit["verification_status"] = (
            "SKIPPED" if job_config else "SKIPPED (no pixels)"
        )
        return ds, None

    # Every stage below treats a 3-D array as (frames, rows, cols). A
    # colour image is (rows, cols, samples) and would be processed as
    # hundreds of one-row "frames": refuse it rather than write a file
    # whose burned-in text was never really looked at.
    samples = int(ds.get("SamplesPerPixel", 1))
    if samples != 1:
        raise FileError(
            "UNSUPPORTED_COLOUR",
            f"colour images are not supported (SamplesPerPixel={samples}, "
            f"PhotometricInterpretation={ds.get('PhotometricInterpretation')})",
        )

    pixels = ds.pixel_array.copy()
    original_dtype = pixels.dtype

    # Strip overlay planes (every 60xx group), and clear any plane that lives
    # in the unused high bits of the pixels themselves.
    bits_stored = int(ds.get("BitsStored", 0) or 0)
    overlays = strip_overlays(ds)
    if overlays["planes"]:
        audit["overlay_planes_removed"] = overlays["planes"]
        log.info(f"  Removed {overlays['planes']} overlay plane(s).")
    if overlays["embedded_bits"] and bits_stored:
        if any(b >= bits_stored for b in overlays["embedded_bits"]):
            pixels = mask_embedded_overlay_bits(pixels, bits_stored)
            audit["overlay_bits_cleared"] = True

    # Fix MONOCHROME1 inversion (dark=bright for OCR correctness)
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    is_monochrome1 = photometric == "MONOCHROME1"
    original_max = None
    if is_monochrome1:
        original_max = np.max(pixels)
        pixels = original_max - pixels
        log.info("  MONOCHROME1 detected — pixel values flipped.")

    # Robust contrast stretching (1st to 99th percentile) for 8-bit OCR pipeline
    # Prevents extreme outlier pixels (metal implants, air padding) from crushing text contrast
    p_min, p_max = np.percentile(pixels, (1.0, 99.0))
    if p_max > p_min:
        clipped = np.clip(pixels, p_min, p_max)
        norm_8 = ((clipped - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
    else:
        norm_8 = pixels.astype(np.uint8)

    # Handle multi-frame: process frame 0 for OCR detection, apply to all frames
    if pixels.ndim == 3:
        ocr_frame = norm_8[0]
        ocr_pixels = pixels[0]
    else:
        ocr_frame = norm_8
        ocr_pixels = pixels

    # ── Stage 3: OCR ──────────────────────────────────────────────────────
    # Full-image OCR. The shape-based candidate regions (Stage 2,
    # text_region_detect.py) are only consumed in run_on_regions mode,
    # so computing them here was wasted work. Full-image OCR is kept on
    # purpose: region-only OCR misses text the shape detector misses.
    log.info("  [Stage 3] Running OCR text detection on the full frame...")
    raw_det = detect_text_in_regions(ocr_frame, [], paddle_ocr=paddle_ocr)
    log.info(f"            Raw detections: {len(raw_det)}")

    # ── Stage 4: Classify ─────────────────────────────────────────────────
    log.info("  [Stage 4] Classifying detections (PHI vs clinical)...")
    merged = merge_detections(raw_det)
    phi_regions = classify_phi(
        merged,
        ocr_frame.shape,
        analyzer,
        gliner_model,
        medical_ner,
        deid_model,
        shared_ner_phi=_shared_ner_phi_texts(merged, job_config, ner),
    )
    phi_regions = expand_phi_blocks(merged, phi_regions, ocr_frame.shape)
    log.info(f"            PHI regions to redact: {len(phi_regions)}")

    # ── Step 3: match OCR text against the full original-tag backup ────────
    log.info(
        "  [Step 3] Cross-checking OCR text against data.json (original tag values)..."
    )
    tag_matches = match_against_stored_tags(merged, stored_values, ocr_frame.shape)
    existing_bboxes = [r["bbox"] for r in phi_regions]
    added = 0
    for m in tag_matches:
        if not any(_iou(m["bbox"], b) > 0.3 for b in existing_bboxes):
            phi_regions.append(m)
            existing_bboxes.append(m["bbox"])
            added += 1
    log.info(f"            Additional regions matched to stored tags: {added}")

    # The text is kept in memory for this run only; main.py strips it
    # before any audit is written to the output volume.
    audit["redacted_regions"] = [
        {"text": r["text"], "bbox": r["bbox"]} for r in phi_regions
    ]

    # ── Stage 5: Redact ───────────────────────────────────────────────────
    log.info("  [Stage 5] Redacting PHI pixels (Navier-Stokes inpainting)...")
    if pixels.ndim == 3:
        # Apply to every frame
        cleaned_frames = []
        for i in range(pixels.shape[0]):
            frame_pix = pixels[i]
            frame_phi = phi_regions  # same regions for all frames
            cleaned_frame, _ = redact_pixels(
                frame_pix, frame_phi, method=pixel_method
            )
            cleaned_frames.append(cleaned_frame)
        cleaned_pixels = np.stack(cleaned_frames, axis=0)
    else:
        cleaned_pixels, _ = redact_pixels(pixels, phi_regions, method=pixel_method)

    # ── Stage 6: Verify ───────────────────────────────────────────────────
    verify_frame = cleaned_pixels[0] if cleaned_pixels.ndim == 3 else cleaned_pixels
    cleaned_pixels_final = cleaned_pixels.copy()
    verify_clean, status = verify_redaction(
        verify_frame,
        phi_regions,
        paddle_ocr=paddle_ocr,
        analyzer=analyzer,
        method=pixel_method,
        strict=strict_verify,
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

    # What the file-writing wrapper needs for its opt-in extras (bbox preview,
    # after_preview.png). process_file ignores it: it writes nothing.
    return ds, PixelArtifacts(ocr_frame, phi_regions, cleaned_pixels_final)



def redact_file_pixels(
    input_path,
    before_output_path,
    output_path,
    data_snapshot_path,
    paddle_ocr,
    analyzer,
    deid_model=None,
    medical_ner=None,
    gliner_model=None,
    ds=None,
    job_config=None,
    ner=None,
):
    """
    Pixel stage (Stages 1-7 up to the checkpoint): OCR, PHI classification,
    redaction, verification, then before_output_path with the redacted pixels
    and the original tags. Touches no shared state, so it is safe to run in a
    worker process. Pass `ds` to reuse an already-read dataset (read it with
    read_dicom so fail_on_unparsable applies).

    A file-based wrapper around redact_dataset_pixels (which holds the logic
    and is what single_file.process_file calls). Returns an audit dict for
    deidentify_file_tags.
    """
    filename = os.path.basename(input_path)
    audit = new_audit(filename, input_path, before_output_path, output_path)

    try:
        if ds is None:
            ds = read_dicom(input_path, job_config)
        stored_values = load_original_tag_values(data_snapshot_path)
        ds, artifacts = redact_dataset_pixels(
            ds,
            filename,
            audit,
            paddle_ocr,
            analyzer,
            stored_values,
            deid_model=deid_model,
            medical_ner=medical_ner,
            gliner_model=gliner_model,
            job_config=job_config,
            ner=ner,
        )

        if artifacts is None:
            # No pixel data (SR, KO, PR): save the metadata-only checkpoint.
            os.makedirs(os.path.dirname(before_output_path), exist_ok=True)
            ds.save_as(before_output_path, write_like_original=False)
            audit["pixel_stage_complete"] = True
            return audit

        # ── Bbox visualization (opt-in: SKALD_SAVE_BBOX_PREVIEW=1) ───────────
        # The frame BEFORE redaction with boxes drawn on it: it shows the
        # burned-in PHI in the clear, so it goes beside the still-identified
        # checkpoint, never by default.
        if SAVE_BBOX_PREVIEW:
            bbox_output_path = os.path.join(
                os.path.dirname(before_output_path), BBOX_IMAGE_NAME
            )
            save_bbox_image(artifacts.ocr_frame, artifacts.phi_regions, bbox_output_path)
            audit["bbox_image_path"] = bbox_output_path

        # ── Checkpoint: pixels redacted, tags still original ─────────────────
        os.makedirs(os.path.dirname(before_output_path), exist_ok=True)
        ds.save_as(before_output_path, write_like_original=False)
        log.info(
            f"  [Checkpoint] Pixel-redacted DICOM saved (tags still original): {before_output_path}"
        )
        audit["pixel_stage_complete"] = True
        log.info(f"         Pixel status: {audit['verification_status']}")

        # Also save normalized 8-bit preview PNG for quick inspection
        try:
            cleaned_pixels_final = artifacts.cleaned_pixels
            vis_source = (
                cleaned_pixels_final[0]
                if cleaned_pixels_final.ndim == 3
                else cleaned_pixels_final
            )
            pmin, pmax = np.percentile(vis_source, (1.0, 99.0))
            if pmax > pmin:
                preview_8 = np.clip(
                    (vis_source - pmin) / (pmax - pmin) * 255.0, 0, 255
                ).astype(np.uint8)
            else:
                preview_8 = vis_source.astype(np.uint8)
            import cv2

            preview_p = os.path.join(os.path.dirname(output_path), "after_preview.png")
            cv2.imwrite(preview_p, preview_8, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        except Exception as pe:
            log.warning(f"  Could not save after_preview.png: {pe}")

    except Exception as e:
        log.error(f"  [ERROR] {filename}: {e}", exc_info=True)
        audit["error"] = str(e)
        audit["verification_status"] = "ERROR"

    return audit


def _deidentify_tags(ds, keystore, secrets, job_config, ner=None):
    """Runs the tag pass `job_config` selects on `ds`, in place; returns the
    audit trail."""
    if job_config is not None:
        # The user's per-tag choices from the SPIDEr UI.
        return deidentify_dataset_from_config(
            ds, keystore, secrets, job_config, ner=ner
        )
    if not ENABLE_TAG_DEIDENTIFICATION:
        log.info(
            "  Tag de-identification skipped (DICOM header tags preserved 100% untouched)."
        )
        return []
    return deidentify_dataset(ds, keystore)


def deidentify_file_tags(audit, keystore, job_config=None, secrets=None, ner=None):
    """
    Tag stage: reads the pixel-redacted checkpoint, de-identifies its tags
    with the shared `keystore` (and, under a job config, the run's `secrets`),
    and saves audit["output_path"]. Must run in the process that owns the
    KeyStore, one file at a time, in input order. Returns the same audit dict,
    updated.
    """
    if audit.get("error") or not audit.get("pixel_stage_complete"):
        return audit

    filename = audit["file"]
    output_path = audit["output_path"]
    try:
        ds = pydicom.dcmread(audit["before_output_path"], force=True)

        # ── Last: tag de-identification ───────────────────────────────────────
        audit["deidentified_tags"] = _deidentify_tags(
            ds, keystore, secrets, job_config, ner=ner
        )

        # ── Save as DICOM ONLY ────────────────────────────────────────────────
        ds.save_as(output_path, write_like_original=False)
        audit["output_transfer_syntax"] = str(ds.file_meta.TransferSyntaxUID)

        # Confirm output is .dcm
        assert output_path.lower().endswith(".dcm"), "BUG: output path must be .dcm!"
        log.info(f"  [DONE] Anonymized DICOM saved: {output_path}")
    except Exception as e:
        log.error(f"  [ERROR] {filename} (tag stage): {e}", exc_info=True)
        audit["error"] = str(e)
        audit["verification_status"] = "ERROR"

    return audit
