import os
import sys
import tempfile
import json
import csv
import pytest

app_dir = os.path.join(os.path.dirname(__file__), "..", "app")
sys.path.insert(0, app_dir)

from audit_reporter import generate_batch_redaction_summary


def test_generate_batch_redaction_summary():
    with tempfile.TemporaryDirectory() as tmp_dir:
        sample_results = [
            {
                "file": "file_b.dcm",
                "verification_status": "PASSED",
                "modality": "CR",
                "image_size": "2048x2048",
                "execution_time_seconds": 3.45,
                "redacted_regions": [
                    {"text": "PATIENT: RAMESHWAR DESHMUKH", "bbox": [10, 20, 200, 50], "reason": "phi_detected"},
                    {"text": "DOB: 12/05/1980", "bbox": [10, 55, 120, 75], "reason": "phi_detected"},
                ],
                "kept_regions": [
                    {"text": "CHEST PA VIEW", "bbox": [10, 90, 150, 110], "reason": "clinical_kept"}
                ],
                "deidentified_tags": [
                    {"tag": "(0010,0010)", "field": "PatientName", "technique": "pseudonymize", "action": "replaced"}
                ]
            },
            {
                "file": "file_a.dcm",
                "verification_status": "PASSED",
                "modality": "DX",
                "image_size": "1024x1024",
                "execution_time_seconds": 1.20,
                "redacted_regions": [],
                "kept_regions": [],
                "deidentified_tags": []
            },
        ]

        meta = generate_batch_redaction_summary(sample_results, tmp_dir)

        json_path = meta["json_path"]
        csv_path = meta["csv_path"]
        md_path = meta["markdown_path"]

        assert os.path.exists(json_path)
        assert os.path.exists(csv_path)
        assert os.path.exists(md_path)

        # 1. Verify JSON content
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            assert data["total_files_processed"] == 2
            assert data["total_pixel_regions_redacted"] == 2
            assert data["total_clinical_regions_preserved"] == 1
            assert data["total_metadata_tags_deidentified"] == 1
            assert len(data["files"]) == 2

        # 2. Verify CSV content
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = list(csv.reader(f))
            headers = reader[0]
            assert "File Name" in headers
            assert "Category" in headers
            assert "Action" in headers
            assert len(reader) >= 4

        # 3. Verify Markdown content
        with open(md_path, "r", encoding="utf-8") as f:
            md_content = f.read()
            assert "Batch DICOM De-Identification & Redaction Audit Report" in md_content
            assert "file_b.dcm" in md_content
            assert "RAMESHWAR DESHMUKH" in md_content
