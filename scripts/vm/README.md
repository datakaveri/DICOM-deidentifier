# Batch runs on a VM

Scripts for de-identifying a large set of DICOMs on a VM rather than in the
enclave. Run them in this order:

| Step | Where | Script |
|---|---|---|
| 1. Inventory every DICOM to Excel; select the chest images and their JPGs | where the data is | `scripts/batch/inventory.py --out RUN_DIR FOLDER...` |
| 2. One-time VM setup: data disk at `/data`, Docker, repo clone | VM, via `ssh vm 'bash -s' < scripts/vm/setup_vm.sh` | `setup_vm.sh` |
| 3. Upload only the selected files | where the data is | `upload_selection.sh user@vm RUN_DIR/selection FOLDER...` |
| 4. Build the image | VM | `docker build -t skald-dicom:batch /data/DICOM-deidentifier` |
| 5. Run every folder, with monitoring | VM | `nohup scripts/vm/run_batch.sh > /data/reports/batch.log 2>&1 &` |

`run_batch.sh` mirrors the input layout: `output/<folder>/<relative path>`
keeps each file's name. It also leaves `StudyInstanceUID`,
`SeriesInstanceUID` and `SOPInstanceUID` unchanged (`PRESERVE_TAGS`), copies
JPGs through unprocessed, and writes the benchmark logs under
`reports/<folder>/`:

- `run.log`: the container's full log, including a `PROGRESS` line per file.
- `monitor.csv`: container CPU and memory, plus host memory and load, every 5 s.
- `benchmark_files.jsonl`: one record per file, with seconds per stage (read,
  decode, OCR, tag match, classify, redact, verify, write, tag stage), worker
  memory, image size, regions and status.
- `benchmark_summary.json`: model load time, wall time, throughput,
  per-stage p50/p90/p99, and the peak memory of the container and processes.
- `manifest.json` and per-file `pipeline_audit.json` / `tag_audit.json`.

`reports/run_summary.json` combines all folders.

A rerun skips files that already finished (`SKALD_RESUME=1`). It keeps
pseudonyms consistent because the same `/data/keys` is reused. Never copy
`/data/keys` off the VM together with the output.
