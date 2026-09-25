#!/usr/bin/env bash
# Synthetic-kernel scaling of TritonGPUCoalesce under Triton 3.7.0 and 3.8.0 (compile only).
# Optional overrides (defaults are the original run's): ZKDUEL_DIAG_DIR (output directory),
# ZKDUEL_TRITON38_PY (python of a Triton 3.8.0 venv), ZKDUEL_TRITON_VERSIONS
# (default "3.7.0 3.8.0").
cd "$(dirname "$0")/../../.."
source experiments/h100/env_table5.sh
OUT=${ZKDUEL_DIAG_DIR:-experiments/h100/diag}/coalesce_scaling.jsonl
mkdir -p "$(dirname "$OUT")"
declare -A PYS=([3.7.0]="$ZKDUEL_VENV/bin/python" [3.8.0]=${ZKDUEL_TRITON38_PY:-/home/ubuntu/venvs/zkduel-triton38/bin/python})
for ver in ${ZKDUEL_TRITON_VERSIONS:-3.7.0 3.8.0}; do
  nice -n 10 ${PYS[$ver]} experiments/h100/tools/coalesce_scaling.py --out $OUT
done
echo COALESCE_SCALING_DONE
