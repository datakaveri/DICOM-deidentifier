#!/usr/bin/env bash
# One-time setup of an Ubuntu 24.04 batch VM for skald-dicom (run as the admin
# user over SSH; uses sudo). Idempotent: safe to re-run.
#   - formats and mounts the attached empty data disk at /data
#   - installs Docker with its data root on /data, plus monitoring tools
#   - clones datakaveri/DICOM-deidentifier to /data/DICOM-deidentifier
set -euo pipefail
REPO_URL=${REPO_URL:-https://github.com/datakaveri/DICOM-deidentifier.git}
BRANCH=${BRANCH:-main}

if ! mountpoint -q /data; then
  # The data disk is the only unpartitioned, unmounted disk that is not the
  # OS disk or the Azure temp disk.
  DISK=$(lsblk -dpno NAME,TYPE,SIZE | awk '$2=="disk"{print $1}' | while read -r d; do
    if [ -z "$(lsblk -no MOUNTPOINT,FSTYPE "$d" | tr -d ' \n')" ]; then echo "$d"; fi
  done | head -1)
  [ -n "$DISK" ] || { echo "no empty data disk found"; lsblk; exit 1; }
  echo "formatting $DISK as /data"
  sudo mkfs.ext4 -q -L data "$DISK"
  sudo mkdir -p /data
  echo "LABEL=data /data ext4 defaults,nofail,discard 0 2" | sudo tee -a /etc/fstab >/dev/null
  sudo mount /data
fi
sudo chown "$(id -u):$(id -g)" /data

if ! command -v docker >/dev/null; then
  sudo apt-get update -qq
  sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq docker.io docker-buildx git rsync sysstat jq python3-venv >/dev/null
  sudo mkdir -p /etc/docker /data/docker
  echo '{"data-root": "/data/docker"}' | sudo tee /etc/docker/daemon.json >/dev/null
  sudo systemctl restart docker
  sudo usermod -aG docker "$USER"
fi

if [ ! -d /data/DICOM-deidentifier/.git ]; then
  git clone -q "$REPO_URL" /data/DICOM-deidentifier
fi
git -C /data/DICOM-deidentifier fetch -q origin
git -C /data/DICOM-deidentifier checkout -q "$BRANCH"
git -C /data/DICOM-deidentifier pull -q --ff-only || true

echo "--- setup done"
nproc; free -g | head -2; df -h /data | tail -1; sudo docker info --format 'docker {{.ServerVersion}} root={{.DockerRootDir}}'
git -C /data/DICOM-deidentifier log --oneline -1
