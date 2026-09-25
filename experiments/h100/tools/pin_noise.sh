#!/usr/bin/env bash
# Run-to-run noise of the wall-clock medians: is Triton's host time tied to the vCPU a process
# lands on? One configuration (zk_spartan_2, 32-bit, r=24, the sweep's settings), each backend
# pinned to each vCPU in turn (taskset), then UNPINNED unpinned runs. Needs a quiet GPU.
# Output: diag/pin_noise.jsonl, one line per run.
# Optional overrides (defaults are the original run's): ZKDUEL_DIAG_DIR (output directory),
# ZKDUEL_SWEEP_CACHE (Triton cache), ZKDUEL_CUDA_BIN, PIN_CPUS (vCPUs to pin to; default all),
# W, BW, R (configuration), UNPINNED (number of unpinned runs).
cd "$(dirname "$0")/../../.."
source experiments/h100/env_table5.sh
export TRITON_CACHE_DIR="${ZKDUEL_SWEEP_CACHE:-$PWD/.triton-cache-h100-table5}"
OUT=${ZKDUEL_DIAG_DIR:-experiments/h100/diag}/pin_noise.jsonl
CUDA_BIN=${ZKDUEL_CUDA_BIN:-build/field_sumcheck_cuda}
mkdir -p "$(dirname "$OUT")"
TMP=$(mktemp -d)
ARGS="--workloads ${W:-zk_spartan_2} --bit-widths ${BW:-32} --rounds ${R:-24} --warmups 2 --repeats 10 --point-mode specialized --layout element"
run() {  # run LABEL CPU BACKEND
  local pin=() csv="$TMP/$1_$2_$3.csv"
  [ "$2" != none ] && pin=(taskset -c "$2")
  if [ "$3" = triton ]; then
    "${pin[@]}" python benchmarks/field_sweep/triton_sweep.py $ARGS --out "$csv" > "$csv.log" 2>&1
  else
    "${pin[@]}" "$CUDA_BIN" $ARGS --out "$csv" > "$csv.log" 2>&1
  fi
  python3 - "$csv" "$1" "$2" "$3" >> $OUT <<'PY'
import csv, json, sys
p, label, cpu, backend = sys.argv[1:]
row = next(csv.DictReader(open(p)))
print(json.dumps({"label": label, "cpu": cpu, "backend": backend, "median_ms": float(row["median_ms"]),
                  "workload": row["workload"], "bit_width": int(row["bit_width"]), "rounds": int(row["rounds"])}))
PY
}
for cpu in ${PIN_CPUS:-$(seq 0 $(($(nproc --all) - 1)))}; do
  for b in triton cuda; do run pinned "$cpu" "$b"; done
done
for i in $(seq 1 "${UNPINNED:-10}"); do
  for b in triton cuda; do run unpinned none "$b"; done
done
rm -rf "$TMP"
echo PIN_NOISE_DONE
