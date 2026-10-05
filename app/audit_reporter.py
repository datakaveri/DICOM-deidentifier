"""
audit_reporter.py — Comprehensive Batch Redaction Summary & Audit Generator.

Generates unified reports across all processed DICOM files in a batch:
  1. redaction_summary.json  — Complete machine-readable master audit file.
  2. redaction_summary.csv   — Tabular report for Excel/Sheets (filter by REDACTED vs KEPT).
  3. redaction_summary_report.md — Human-readable markdown audit report for clinical review.
"""

from __future__ import annotations

import os
import json
import csv
import datetime
from typing import List, Dict, Any


def generate_batch_redaction_summary(results: List[Dict[str, Any]], output_dir: str) -> Dict[str, Any]:
    """
    Analyzes all pipeline audit results and writes out master summary files
    (JSON, CSV, Markdown) to output_dir so users can verify exactly what was
    redacted and what was preserved across the whole dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    total_files = len(results)
    total_pixel_redacted = sum(len(a.get("redacted_regions", [])) for a in results)
    total_clinical_kept = sum(len(a.get("kept_regions", [])) for a in results)
    total_tags_deid = sum(len(a.get("deidentified_tags", [])) for a in results)

    # ── 1. Build JSON Master Audit Data ───────────────────────────────────────
    master_summary = {
        "report_generated_at": timestamp,
        "total_files_processed": total_files,
        "total_pixel_regions_redacted": total_pixel_redacted,
        "total_clinical_regions_preserved": total_clinical_kept,
        "total_metadata_tags_deidentified": total_tags_deid,
        "files": []
    }

    for audit in results:
        fname = audit.get("file", "unknown")
        status = audit.get("verification_status", "UNKNOWN")
        exec_time = audit.get("execution_time_seconds", 0.0)

        redactions = audit.get("redacted_regions", [])
        kept = audit.get("kept_regions", [])
        tags = audit.get("deidentified_tags", [])

        file_entry = {
            "filename": fname,
            "status": status,
            "execution_time_seconds": exec_time,
            "modality": audit.get("modality", "Unknown"),
            "image_size": audit.get("image_size", "Unknown"),
            "pixel_redactions_count": len(redactions),
            "pixel_redactions": [
                {
                    "text": r.get("text", ""),
                    "bbox": r.get("bbox", []),
                    "reason": r.get("reason", "unspecified")
                }
                for r in redactions
            ],
            "clinical_preserved_count": len(kept),
            "clinical_preserved": [
                {
                    "text": k.get("text", ""),
                    "bbox": k.get("bbox", []),
                    "reason": k.get("reason", "clinical_or_non_pii_kept")
                }
                for k in kept
            ],
            "metadata_deidentified_count": len(tags),
            "metadata_deidentified": [
                {
                    "tag": t.get("tag", ""),
                    "field": t.get("field", ""),
                    "technique": t.get("technique", ""),
                    "action": t.get("action", "")
                }
                for t in tags
            ]
        }
        master_summary["files"].append(file_entry)

    json_path = os.path.join(output_dir, "redaction_summary.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(master_summary, f, indent=2)

    # ── 2. Build Tabular CSV Report ───────────────────────────────────────────
    csv_path = os.path.join(output_dir, "redaction_summary.csv")
    csv_headers = [
        "File Name",
        "Category",
        "Action",
        "Item Content / Field",
        "Coordinates / Technique",
        "Reason / Rule Triggered",
        "Verification Status"
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

        for audit in results:
            fname = audit.get("file", "unknown")
            status = audit.get("verification_status", "UNKNOWN")

            # Log Redacted Burned-in Pixels
            for r in audit.get("redacted_regions", []):
                writer.writerow([
                    fname,
                    "Burned-in Pixel",
                    "REDACTED",
                    r.get("text", ""),
                    str(r.get("bbox", [])),
                    r.get("reason", "phi_detected"),
                    status
                ])

            # Log Preserved Clinical Words
            for k in audit.get("kept_regions", []):
                writer.writerow([
                    fname,
                    "Burned-in Pixel",
                    "KEPT (Clinical)",
                    k.get("text", ""),
                    str(k.get("bbox", [])),
                    k.get("reason", "clinical_kept"),
                    status
                ])

            # Log De-identified Metadata Tags
            for t in audit.get("deidentified_tags", []):
                field_str = f"{t.get('tag', '')} {t.get('field', '')}".strip()
                writer.writerow([
                    fname,
                    "Metadata Header",
                    "DE-IDENTIFIED",
                    field_str,
                    t.get("technique", "suppress"),
                    t.get("action", "modified"),
                    status
                ])

    # ── 3. Build Markdown Report ──────────────────────────────────────────────
    md_path = os.path.join(output_dir, "redaction_summary_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 📋 Batch DICOM De-Identification & Redaction Audit Report\n\n")
        f.write(f"**Generated:** {timestamp}  \n")
        f.write(f"**Total Files Processed:** {total_files}  \n")
        f.write(f"**Total Burned-In PHI Regions Redacted:** {total_pixel_redacted}  \n")
        f.write(f"**Total Diagnostic / Clinical Texts Kept:** {total_clinical_kept}  \n")
        f.write(f"**Total Metadata Header Tags De-Identified:** {total_tags_deid}  \n\n")

        f.write("## 1. Batch Executive Summary\n\n")
        f.write("| File Name | Status | Burned-in Redacted | Clinical Kept | Metadata Tags Scrubbed | Execution Time |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for audit in results:
            fname = audit.get("file", "unknown")
            status = audit.get("verification_status", "UNKNOWN")
            r_cnt = len(audit.get("redacted_regions", []))
            k_cnt = len(audit.get("kept_regions", []))
            t_cnt = len(audit.get("deidentified_tags", []))
            sec = audit.get("execution_time_seconds", 0.0)
            f.write(f"| `{fname}` | **{status}** | {r_cnt} | {k_cnt} | {t_cnt} | {sec:.2f}s |\n")

        f.write("\n---\n\n")
        f.write("## 2. Detailed Per-File Audit Trail\n\n")
        for audit in results:
            fname = audit.get("file", "unknown")
            f.write(f"### File: `{fname}`\n\n")

            redactions = audit.get("redacted_regions", [])
            if redactions:
                f.write("#### 🔴 Redacted Burned-In PHI Pixels\n")
                f.write("| Text Found & Redacted | Bounding Box `[x1, y1, x2, y2]` | Detection Rule / Reason |\n")
                f.write("| :--- | :--- | :--- |\n")
                for r in redactions:
                    clean_txt = r.get("text", "").replace("|", "\\|")
                    f.write(f"| **{clean_txt}** | `{r.get('bbox', [])}` | `{r.get('reason', 'phi_detected')}` |\n")
                f.write("\n")
            else:
                f.write("*(No burned-in PHI text detected in this image)*\n\n")

            kept = audit.get("kept_regions", [])
            if kept:
                f.write("#### 🟢 Preserved Clinical & Diagnostic Text (Not Redacted)\n")
                f.write("| Clinical Text Preserved | Bounding Box | Reason Preserved |\n")
                f.write("| :--- | :--- | :--- |\n")
                for k in kept:
                    clean_txt = k.get("text", "").replace("|", "\\|")
                    f.write(f"| `{clean_txt}` | `{k.get('bbox', [])}` | `{k.get('reason', 'clinical_kept')}` |\n")
                f.write("\n")

            tags = audit.get("deidentified_tags", [])
            if tags:
                f.write("#### 🔒 De-Identified DICOM Header Tags (Sample)\n")
                f.write("| Tag ID | Field Name | Technique | Action Taken |\n")
                f.write("| :--- | :--- | :--- | :--- |\n")
                for t in tags[:15]:  # show up to first 15 tags in MD
                    f.write(f"| `{t.get('tag')}` | `{t.get('field')}` | `{t.get('technique')}` | `{t.get('action')}` |\n")
                if len(tags) > 15:
                    f.write(f"| *... and {len(tags) - 15} more tags* | | | *(see redaction_summary.json)* |\n")
                f.write("\n")

            f.write("---\n\n")

    return {
        "json_path": json_path,
        "csv_path": csv_path,
        "markdown_path": md_path,
        "total_files": total_files,
        "total_pixel_redacted": total_pixel_redacted,
        "total_clinical_kept": total_clinical_kept,
        "total_tags_deid": total_tags_deid
    }
