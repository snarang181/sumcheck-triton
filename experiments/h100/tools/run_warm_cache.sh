#!/usr/bin/env bash
# Compile every 128/256-bit eval kernel of the sweep in parallel (one kernel per process).
cd "$(dirname "$0")/../../.."
source experiments/h100/env_table5.sh
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton-cache-h100-table5}"
xargs -P "${JOBS:-12}" -L 1 sh -c 'timeout 10800 python experiments/h100/tools/warm_cache.py "$0" "$1" "$2" || echo "FAILED $0 $1 $2 rc=$?"' \
  < experiments/h100/logs/warm_tasks.txt
echo WARM_DONE
