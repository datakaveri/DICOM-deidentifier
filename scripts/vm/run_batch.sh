#!/usr/bin/env bash
# Runs skald-dicom over every folder under INPUT_ROOT on this VM, one container
# per folder, with detailed benchmark logs. Re-runnable: SKALD_RESUME skips
# files an earlier run already finished.
#
#   INPUT_ROOT/<folder>/<rel>.dcm  ->  OUTPUT_ROOT/<folder>/<rel>.dcm  (same name)
#   INPUT_ROOT/<folder>/<rel>.jpg  ->  OUTPUT_ROOT/<folder>/<rel>.jpg  (copied as is)
#   REPORT_ROOT/<folder>/          run.log, monitor.csv, manifest.json,
#                                  benchmark_summary.json, benchmark_files.jsonl,
#                                  per-file audits and previews
#   REPORT_ROOT/run_summary.json   both folders, with timings and counts
#
# Only DICOMs and JPGs that are in INPUT_ROOT are touched; select them before
# upload (scripts/batch/inventory.py). Tags in PRESERVE_TAGS are not changed.
# SKALD_WORKERS / SKALD_THREADS_PER_WORKER, if set, override the auto plan.
set -uo pipefail
INPUT_ROOT=${INPUT_ROOT:-/data/input}
OUTPUT_ROOT=${OUTPUT_ROOT:-/data/output}
REPORT_ROOT=${REPORT_ROOT:-/data/reports}
KEYS_DIR=${KEYS_DIR:-/data/keys}
IMAGE=${IMAGE:-skald-dicom:batch}
PRESERVE_TAGS=${PRESERVE_TAGS:-StudyInstanceUID,SeriesInstanceUID,SOPInstanceUID}
HERE=$(cd "$(dirname "$0")" && pwd)

mkdir -p "$OUTPUT_ROOT" "$REPORT_ROOT" "$KEYS_DIR"
chmod 700 "$KEYS_DIR"
summary="$REPORT_ROOT/run_summary.json"
echo "{}" > "$summary"
run_start=$(date +%s)
echo "[$(date -Is)] host: $(nproc) vCPU, $(free -g | awk '/^Mem:/{print $2}') GB RAM, image $IMAGE ($(docker image inspect -f '{{.Id}}' "$IMAGE" | cut -c1-19))"

i=0
for in_dir in "$INPUT_ROOT"/*/; do
  folder=$(basename "$in_dir")
  i=$((i + 1))
  out="$OUTPUT_ROOT/$folder"; rep="$REPORT_ROOT/$folder"; name="skald-batch-$i"
  mkdir -p "$out" "$rep"
  n_dcm=$(find "$in_dir" -name '*.dcm' | wc -l)
  echo "[$(date -Is)] === $folder: $n_dcm DICOM(s)"
  docker rm -f "$name" >/dev/null 2>&1
  start=$(date +%s)
  docker run -d --name "$name" --network none --user "$(id -u):$(id -g)" \
    -e SKALD_OUTPUT_LAYOUT=mirror -e SKALD_REPORT_DIR=/app/reports \
    -e SKALD_PRESERVE_TAGS="$PRESERVE_TAGS" -e SKALD_DISCARD_INTERMEDIATES=1 \
    -e SKALD_RESUME=1 -e SKALD_KEYSTORE_SAVE_EVERY=25 \
    ${SKALD_WORKERS:+-e SKALD_WORKERS=$SKALD_WORKERS} \
    ${SKALD_THREADS_PER_WORKER:+-e SKALD_THREADS_PER_WORKER=$SKALD_THREADS_PER_WORKER} \
    -v "$in_dir":/app/data:ro -v "$out":/app/output -v "$rep":/app/reports \
    -v "$KEYS_DIR":/app/keys "$IMAGE" >/dev/null
  "$HERE/monitor.sh" "$name" "$rep/monitor.csv" 5 &
  mon=$!
  docker logs -f "$name" > "$rep/run.log" 2>&1
  code=$(docker wait "$name")
  end=$(date +%s)
  wait "$mon" 2>/dev/null
  docker rm "$name" >/dev/null

  # Chest JPGs: copied as they are, never processed.
  rsync -a --include='*/' --include='*.jpg' --exclude='*' --prune-empty-dirs "$in_dir" "$out/"
  n_out_dcm=$(find "$out" -name '*.dcm' | wc -l)
  n_out_jpg=$(find "$out" -name '*.jpg' | wc -l)
  peak_cpu=$(awk -F, 'NR>1 && $2+0>m{m=$2+0} END{print m+0}' "$rep/monitor.csv")
  peak_mem=$(awk -F, 'NR>1{print $3}' "$rep/monitor.csv" | sort -h | tail -1)
  echo "[$(date -Is)] === $folder: exit $code, $((end - start))s, $n_out_dcm DICOM(s) + $n_out_jpg JPG(s) in output, peak container CPU ${peak_cpu}% mem ${peak_mem}"
  jq --arg f "$folder" --argjson code "$code" --argjson secs "$((end - start))" \
     --argjson in "$n_dcm" --argjson od "$n_out_dcm" --argjson oj "$n_out_jpg" \
     --arg cpu "$peak_cpu" --arg mem "$peak_mem" \
     --slurpfile bench <(cat "$rep/benchmark_summary.json" 2>/dev/null || echo null) \
     '.[$f] = {exit_code: $code, container_seconds: $secs, input_dicoms: $in,
               output_dicoms: $od, output_jpgs: $oj, monitor_peak_cpu_pct: $cpu,
               monitor_peak_mem: $mem, benchmark: $bench[0]}' "$summary" > "$summary.tmp" \
    && mv "$summary.tmp" "$summary"
done
jq --argjson t "$(( $(date +%s) - run_start ))" '. + {total_seconds: $t}' "$summary" > "$summary.tmp" && mv "$summary.tmp" "$summary"
echo "[$(date -Is)] all done in $(( $(date +%s) - run_start ))s -> $summary"
