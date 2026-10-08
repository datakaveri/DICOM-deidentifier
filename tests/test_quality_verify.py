"""
test_quality_verify.py — Unit and integration tests for post-pipeline quality verification.
"""

import sys
import os
import json
import pytest
import numpy as np
import pydicom
from pydicom.dataset import Dataset, FileMetaDataset

# Ensure app is in path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))

from quality_verify import (
    compute_image_quality_metrics,
    compute_diagnostic_integrity,
    compute_metadata_compliance,
    compute_structural_validity,
    compute_quality_grade,
    run_quality_verification,
)


def _create_synthetic_dicom_pair(shape=(100, 100)):
    """Helper to create matching original and anonymized synthetic DICOM datasets."""
    # File meta
    file_meta = FileMetaDataset()
    file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7"

    # Original ds
    orig = Dataset()
    orig.file_meta = file_meta
    orig.is_little_endian = True
    orig.is_implicit_VR = False
    orig.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    orig.SOPInstanceUID = "1.2.3.4.5.6.7"
    orig.Modality = "CR"
    orig.PatientName = "DOE^JOHN"
    orig.PatientID = "12345"
    orig.PatientBirthDate = "19800101"
    orig.Rows, orig.Columns = shape
    orig.BitsAllocated = 16
    orig.BitsStored = 16
    orig.HighBit = 15
    orig.PixelRepresentation = 0
    orig.SamplesPerPixel = 1
    orig.PhotometricInterpretation = "MONOCHROME2"
    orig_arr = np.zeros(shape, dtype=np.uint16) + 1000
    orig.PixelData = orig_arr.tobytes()

    # Anonymized ds
    anon = Dataset()
    anon.file_meta = FileMetaDataset()
    anon.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian
    anon.is_little_endian = True
    anon.is_implicit_VR = False
    anon.SOPClassUID = "1.2.840.10008.5.1.4.1.1.1"
    anon.SOPInstanceUID = "9.9.9.9.9.9.9"  # Regenerated
    anon.Modality = "CR"
    # PatientName suppressed / removed
    anon.PatientID = "HASHED_999"
    anon.PatientBirthDate = "19800000"  # Masked
    anon.Rows, anon.Columns = shape
    anon.BitsAllocated = 16
    anon.BitsStored = 16
    anon.HighBit = 15
    anon.PixelRepresentation = 0
    anon.SamplesPerPixel = 1
    anon.PhotometricInterpretation = "MONOCHROME2"

    anon_arr = orig_arr.copy()
    # Modify a 10x10 region at [0, 0, 10, 10]
    anon_arr[0:10, 0:10] = 500
    anon.PixelData = anon_arr.tobytes()

    return orig, anon, orig_arr, anon_arr


def test_image_quality_metrics_identical():
    """Identical images must yield SSIM=1.0, 0 modified pixels."""
    arr = np.ones((50, 50), dtype=np.uint16) * 100
    metrics = compute_image_quality_metrics(arr, arr)
    assert metrics["ssim"] == 1.0
    assert metrics["total_pixels_modified"] == 0
    assert metrics["pixel_damage_ratio_pct"] == 0.0
    assert metrics["mse"] == 0.0


def test_image_quality_metrics_modified():
    """Test damage ratio and deviation when pixels are altered."""
    orig = np.zeros((100, 100), dtype=np.uint8)
    anon = orig.copy()
    anon[0:10, 0:10] = 200  # 100 pixels modified out of 10,000 = 1%
    metrics = compute_image_quality_metrics(orig, anon)
    assert metrics["total_pixels"] == 10000
    assert metrics["total_pixels_modified"] == 100
    assert metrics["pixel_damage_ratio_pct"] == 1.0
    assert metrics["max_pixel_deviation"] == 200.0


def test_diagnostic_integrity_perfect():
    """Modifications strictly inside redaction zones must yield PERFECT integrity."""
    orig = np.zeros((100, 100), dtype=np.uint16) + 500
    anon = orig.copy()
    # Redact at [10, 10, 20, 20]
    anon[10:20, 10:20] = 0
    redacted_regions = [{"bbox": [10, 10, 20, 20], "text": "TEST"}]

    res = compute_diagnostic_integrity(orig, anon, redacted_regions, orig.shape)
    assert res["integrity_status"] == "PERFECT"
    assert res["non_redacted_pixels_modified"] == 0
    assert res["modified_in_raw_bbox"] == 100


def test_diagnostic_integrity_leak():
    """Modifications outside in diagnostic anatomy must trigger INTEGRITY_FAILURE."""
    orig = np.zeros((100, 100), dtype=np.uint16) + 500
    anon = orig.copy()
    # Modify redaction box
    anon[10:20, 10:20] = 0
    # Leak outside treatment area: at [60, 60, 75, 75]
    anon[60:75, 60:75] = 999
    redacted_regions = [{"bbox": [10, 10, 20, 20], "text": "TEST"}]

    res = compute_diagnostic_integrity(orig, anon, redacted_regions, orig.shape)
    assert res["integrity_status"] == "INTEGRITY_FAILURE"
    assert res["non_redacted_pixels_modified"] > 10


def test_metadata_compliance():
    """Validates metadata PHI clearance, UID regeneration, and date masking."""
    orig, anon, _, _ = _create_synthetic_dicom_pair()
    res = compute_metadata_compliance(orig, anon)
    assert res["phi_tags_cleared"] is True
    assert res["uid_regenerated"] is True
    assert res["date_masking_valid"] is True
    assert res["compliance_status"] == "PASS"


def test_quality_grading():
    """Verifies that high quality with perfect integrity yields Grade A."""
    img_q = {"ssim": 0.98, "pixel_damage_ratio_pct": 2.0}
    diag_i = {"integrity_status": "PERFECT"}
    meta_c = {"compliance_status": "PASS"}
    struct = {"validity_status": "PASS"}

    grade, reason = compute_quality_grade(img_q, diag_i, meta_c, struct)
    assert grade == "A"


def test_pixel_damage_analysis_breakdown():
    """Tests damage composition and intensity breakdown of modified pixels."""
    orig = np.zeros((100, 100), dtype=np.uint8) + 100
    anon = orig.copy()
    # 50 pixels changed inside bbox
    anon[10:15, 10:20] = 200
    # 20 pixels changed outside bbox but inside padding (say at 10..15, 20..24)
    anon[10:15, 20:24] = 150
    redacted_regions = [{"bbox": [10, 10, 20, 15], "text": "NAME"}]

    from quality_verify import compute_pixel_damage_analysis
    res = compute_pixel_damage_analysis(orig, anon, redacted_regions, orig.shape)

    assert res["total_modified_pixels"] == 70
    comp = res["damage_composition_of_modified"]
    assert comp["intentional_phi_redacted_pixels"] == 50
    assert comp["intentional_phi_pct"] > 70.0
    assert comp["collateral_inpaint_buffer_pixels"] == 20
    assert comp["unintended_diagnostic_damage_pixels"] == 0
    intens = res["intensity_damage_on_modified"]
    assert intens["max_intensity_shift"] == 100.0
    assert intens["mean_intensity_shift"] > 0.0


def test_confidence_assessment():
    """Validates multi-dimensional confidence assessment scoring."""
    from quality_verify import compute_confidence_assessment
    img_q = {"total_pixels_modified": 50, "ssim": 0.99}
    pixel_dmg = {}
    diag_i = {"non_redacted_pixels_modified": 0}
    meta_c = {"phi_tags_cleared": True}
    redacted = [{"bbox": [0, 0, 10, 10], "reason": "cross_verify:metadata_match(NAME)"}]

    conf = compute_confidence_assessment(img_q, pixel_dmg, diag_i, meta_c, redacted)
    assert conf["overall_confidence_score"] >= 0.98
    assert conf["overall_confidence_level"] == "VERY_HIGH"
    assert conf["confidence_dimensions"]["pixel_modification_measurement"]["confidence_percentage"] == 100.0
    assert conf["confidence_dimensions"]["diagnostic_anatomy_preservation"]["confidence_percentage"] == 100.0


def test_run_quality_verification_json_structure(tmp_path):
    """Validates that run_quality_verification outputs all required JSON fields."""
    orig_path = str(tmp_path / "orig.dcm")
    anon_path = str(tmp_path / "anon.dcm")
    orig_ds, anon_ds, _, _ = _create_synthetic_dicom_pair(shape=(100, 100))
    orig_ds.save_as(orig_path)
    anon_ds.save_as(anon_path)

    redacted = [{"bbox": [0, 0, 10, 10], "text": "DOE^JOHN", "reason": "cross_verify:metadata_match(DOE^JOHN)"}]
    res = run_quality_verification(orig_path, anon_path, redacted)

    # 1. Total pixels & modified total in JSON
    assert "image_pixel_summary" in res
    pix_sum = res["image_pixel_summary"]
    assert pix_sum["total_pixels_in_image"] == 10000
    assert pix_sum["total_modified_pixels"] == 100
    assert pix_sum["total_modified_percentage"] == 1.0
    assert pix_sum["total_unmodified_pixels"] == 9900
    assert pix_sum["total_unmodified_percentage"] == 99.0

    # 2. How much is damaged in modified pixels
    assert "modified_pixel_damage_breakdown" in res
    dmg_bd = res["modified_pixel_damage_breakdown"]
    assert dmg_bd["total_modified_pixels"] == 100
    assert "how_much_is_damaged" in dmg_bd
    assert dmg_bd["how_much_is_damaged"]["intentional_phi_redacted_damage"]["pixel_count"] == 100
    assert dmg_bd["how_much_is_damaged"]["unintended_diagnostic_anatomy_damage"]["pixel_count"] == 0
    assert "damage_intensity_and_severity" in dmg_bd

    # 3. Metrics used
    assert "metrics_used" in res
    metrics = res["metrics_used"]
    assert "ssim" in metrics
    assert "psnr_db" in metrics
    assert "mse" in metrics
    assert "diagnostic_region_mse" in metrics
    assert "residual_stroke_check" in metrics

    # 4. Confidence assessment
    assert "confidence_assessment" in res
    conf = res["confidence_assessment"]
    assert conf["overall_confidence_percentage"] >= 95.0
    assert "confidence_dimensions" in conf
    assert "confidence_summary" in conf

    # 5. In-Line vs Post-Pipeline Harmony & Dynamic Range Scaling
    assert "inline_vs_post_pipeline_harmony" in res
    assert res["inline_vs_post_pipeline_harmony"]["harmony_status"] in ("HARMONIZED_ZERO_RESIDUAL", "RESIDUAL_STROKE_FLAGGED")
    sev_info = dmg_bd["damage_intensity_and_severity"]
    assert "actual_clinical_dynamic_range" in sev_info
    assert "theoretical_dynamic_range" in sev_info
    assert "normalized_shift_actual_range_pct" in sev_info
    assert "normalized_shift_theoretical_range_pct" in sev_info


