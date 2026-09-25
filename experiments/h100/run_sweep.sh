#!/usr/bin/env bash
# H100 rerun of the paper's 800-configuration zkPHIRE sweep: 25 workloads x 4 widths x
# rounds 14..28. Settings are the A100 run's as recorded in its CSV (2 warmups, 10 repeats,
# specialized points; whole-run timing, fixed challenge and element layout by default),
# plus --per-case-timeout 1800, which the A100 run did not use.
#
# Run detached:  setsid nohup experiments/h100/run_sweep.sh > experiments/h100/logs/sweep.log 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/h100/env_table5.sh
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton-cache-h100-table5}"
H=experiments/h100
mkdir -p "$H/sweep" "$H/logs"

python experiments/h100/tools/capture_env.py > "$H/logs/capture_env.log" 2>&1
nvidia-smi dmon -s pucmt -d 10 -o TD > "$H/logs/gpu_dmon.log" 2>&1 &
DMON=$!
trap 'kill $DMON 2>/dev/null || true' EXIT

CMD=(python -u benchmarks/field_sweep/resilient_compare_triton_cuda.py
  --zk-only --bit-widths 32 64 128 256
  --rounds 14 16 18 20 22 24 26 28
  --warmups 2 --repeats 10
  --point-mode specialized
  --per-case-timeout 1800
  --skip-build --cuda-binary build/field_sumcheck_cuda
  --python "$ZKDUEL_VENV/bin/python"
  --tmp-dir "$H/sweep/cases"
  --out "$H/sweep/triton_cuda_field_resilient_r14_28_h100.csv")

printf '%q ' "${CMD[@]}" > "$H/sweep/command.txt"
echo >> "$H/sweep/command.txt"
echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" >> "$H/sweep/command.txt"
date -u +%FT%TZ > "$H/sweep/started_at.txt"
"${CMD[@]}"
date -u +%FT%TZ > "$H/sweep/finished_at.txt"
