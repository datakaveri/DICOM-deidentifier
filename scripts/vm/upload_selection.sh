#!/usr/bin/env bash
# Uploads only the selected files (chest DICOMs and their JPGs) of each source
# folder to the VM, keeping folder names and relative paths:
#   <src>/<rel>  ->  HOST:/data/input/<folder name>/<rel>
# usage: upload_selection.sh user@host SELECTION_DIR SRC_FOLDER [SRC_FOLDER...]
# SELECTION_DIR holds <folder>.dcm.txt / <folder>.jpg.txt from inventory.py.
set -euo pipefail
HOST=$1; SEL=$2; shift 2
for src in "$@"; do
  name=$(basename "$src")
  list=$(mktemp)
  cat "$SEL/$name.dcm.txt" "$SEL/$name.jpg.txt" > "$list"
  echo "[$(date -Is)] uploading $(wc -l < "$list") files of '$name'"
  ssh "$HOST" "mkdir -p '/data/input/$name'"
  # -s: paths with spaces; no -z: DICOM pixel data is already compressed.
  rsync -a -s --files-from="$list" --info=progress2,stats1 "$src/" "$HOST:/data/input/$name/"
  rm -f "$list"
  echo "[$(date -Is)] done '$name'"
done
