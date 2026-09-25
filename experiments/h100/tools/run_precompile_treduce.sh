#!/usr/bin/env bash
# Compile every eval kernel of the tl.reduce ablation (25 zkPHIRE workloads x 4 widths,
# minus jellyfish zerocheck at 256 bits) into .triton-cache-h100-treduce, one kernel per
# process. Safe to rerun: already-compiled kernels return in about a second.
cd "$(dirname "$0")/../../.."
source experiments/h100/env_table5.sh
export TRITON_CACHE_DIR="$PWD/.triton-cache-h100-treduce"
xargs -P "${JOBS:-14}" -L 1 sh -c 'timeout 14400 python experiments/h100/tools/precompile_treduce.py "$0" "$1" "$2" || echo "FAILED $0 $1 $2 rc=$?"' \
  < experiments/h100/logs/treduce_tasks.txt
echo PRECOMPILE_TREDUCE_DONE
