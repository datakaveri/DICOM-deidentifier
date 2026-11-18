#!/usr/bin/env python3
"""Turns a batch run's reports into one Excel workbook.

  Summary         one row per folder: files, failures, wall time, throughput,
                  model load, peak memory, p50/p90/p99 seconds per file
  <folder>        one row per processed file from benchmark_files.jsonl:
                  status, regions, OCR detections, image size, seconds per
                  stage, worker memory
  <folder> CPU    the monitor.csv timeline (container CPU and memory)

No tag values are read: the workbook holds timings, counts and paths only.

usage: results_to_excel.py REPORT_ROOT OUT.xlsx
"""

import csv
import json
import os
import re
import sys

from openpyxl import Workbook

STAGES = [
    "read",
    "decode",
    "ocr",
    "tag_match",
    "classify",
    "redact",
    "verify",
    "write_checkpoint",
    "preview",
    "pixel_stage",
    "tag_stage",
    "total",
]


def title(name, used, suffix=""):
    base = re.sub(r"[\[\]:*?/\\]", "_", name)[: 31 - len(suffix)] + suffix
    t, n = base, 2
    while t in used:
        t = f"{base[:28]}_{n}"
        n += 1
    used.add(t)
    return t


def main(report_root, out):
    wb = Workbook()
    summary = wb.active
    summary.title = "Summary"
    summary.append(
        [
            "Folder",
            "Files",
            "Failed",
            "Statuses",
            "Batch wall (s)",
            "Container (s)",
            "Files/min",
            "Model load (s)",
            "Workers x threads",
            "Container peak MB",
            "Worker peak MB",
            "File p50 (s)",
            "File p90 (s)",
            "File p99 (s)",
            "OCR p50 (s)",
            "Regions redacted",
            "Tag-matched regions",
            "Input GB",
        ]
    )
    run_summary = {}
    rs = os.path.join(report_root, "run_summary.json")
    if os.path.exists(rs):
        run_summary = json.load(open(rs))
    used = {"Summary"}
    for folder in sorted(os.listdir(report_root)):
        d = os.path.join(report_root, folder)
        bench_f = os.path.join(d, "benchmark_files.jsonl")
        if not os.path.isfile(bench_f):
            continue
        b = json.load(open(os.path.join(d, "benchmark_summary.json")))
        pfs = b["per_file_seconds"]
        summary.append(
            [
                folder,
                b["files"]["found"],
                b["files"]["failed"],
                json.dumps(b["files"]["by_status"]),
                b["seconds"]["batch_wall"],
                (run_summary.get(folder) or {}).get("container_seconds"),
                b["throughput"]["files_per_minute"],
                b["seconds"]["model_load"],
                f'{b["plan"]["workers"]} x {b["plan"]["threads_per_worker"]}',
                b["memory_mb"]["container_peak"],
                b["memory_mb"]["worker_peak_rss_max"],
                (pfs.get("file_total") or {}).get("p50"),
                (pfs.get("file_total") or {}).get("p90"),
                (pfs.get("file_total") or {}).get("p99"),
                (pfs.get("ocr") or {}).get("p50"),
                b["files"]["redacted_regions_total"],
                b["files"]["tag_matched_regions_total"],
                round(b["files"]["input_bytes_total"] / 1e9, 2),
            ]
        )
        ws = wb.create_sheet(title(folder, used))
        ws.append(
            [
                "File",
                "Status",
                "Error",
                "Regions",
                "Tag-matched",
                "OCR detections",
                "Image size",
                "Bits",
                "Source transfer syntax",
                "Size MB",
                "Tags changed",
            ]
            + [f"{s} (s)" for s in STAGES]
            + ["Worker pid", "Worker RSS MB", "Worker peak MB", "Finished at"]
        )
        for line in open(bench_f):
            r = json.loads(line)
            sec, w = r.get("seconds") or {}, r.get("worker") or {}
            ws.append(
                [
                    r["file"],
                    r["status"],
                    r.get("error"),
                    r["redacted_regions"],
                    r.get("tag_matched_regions"),
                    r.get("ocr_detections"),
                    r.get("image_size"),
                    r.get("bits_allocated"),
                    r.get("source_transfer_syntax"),
                    round((r.get("size_bytes") or 0) / 1e6, 2),
                    r.get("tags_changed"),
                ]
                + [sec.get(s) for s in STAGES]
                + [
                    w.get("pid"),
                    w.get("rss_mb"),
                    w.get("peak_rss_mb"),
                    r.get("finished_at"),
                ]
            )
        mon = os.path.join(d, "monitor.csv")
        if os.path.isfile(mon):
            wm = wb.create_sheet(title(folder, used, " CPU"))
            for row in csv.reader(open(mon)):
                wm.append(row)
    wb.save(out)
    print("wrote", out)


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
