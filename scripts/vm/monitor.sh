#!/usr/bin/env bash
# Samples a running container and its host every INTERVAL seconds into a CSV,
# until the container exits: container CPU% and memory, host used/available
# memory, load average.  usage: monitor.sh CONTAINER CSV [INTERVAL]
set -u
C=$1; CSV=$2; INTERVAL=${3:-5}
echo "timestamp,container_cpu_pct,container_mem_used,container_mem_pct,host_mem_used_mb,host_mem_available_mb,load1,load5" > "$CSV"
while docker ps -q --filter "name=^${C}$" | grep -q .; do
  stats=$(docker stats --no-stream --format '{{.CPUPerc}},{{.MemUsage}},{{.MemPerc}}' "$C" 2>/dev/null | sed 's/%//g; s| / [^,]*||')
  read -r used avail < <(free -m | awk '/^Mem:/{print $3, $7}')
  read -r l1 l5 _ < /proc/loadavg
  [ -n "$stats" ] && echo "$(date -Is),$stats,$used,$avail,$l1,$l5" >> "$CSV"
  sleep "$INTERVAL"
done
