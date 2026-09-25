#!/usr/bin/env bash
# Compile the 8 eval kernels of zk_jellyfish_zerocheck_hp (22 tables, degree 7) at 128 and
# 256 bits, one kernel per process, each capped at 4 h. Logs per-kernel compile time.
cd "$(dirname "$0")/../../.."
source experiments/h100/env_table5.sh
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton-cache-h100-table5}"
for bw in 128 256; do
  for p in 0 1 2 3 4 5 6 7; do
    ( start=$(date +%s)
      timeout 14400 python experiments/h100/tools/warm_cache.py zk_jellyfish_zerocheck_hp "$bw" "$p"
      echo "jellyfish_zerocheck ${bw}-bit point $p exit=$? wall_s=$(( $(date +%s) - start ))" ) &
  done
done
wait
echo JELLYFISH_WARM_DONE
