#!/usr/bin/env bash
# Independent-reference validation of the Triton field-sweep kernels under the Table 5
# stack, in the sweep's configuration (element layout, specialized points). Uses the
# sweep's TRITON_CACHE_DIR, so it also compiles every kernel the sweep will launch.
set -uo pipefail
cd "$(dirname "$0")/../../.."
source experiments/h100/env_table5.sh
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/.triton-cache-h100-table5}"
OUT=experiments/h100/validation
mkdir -p "$OUT"
PY="$ZKDUEL_VENV/bin/python"
TOOL=experiments/h100/tools/validate_vs_registry.py
mapfile -t ALL < <("$PY" -c "import sys; sys.path.insert(0, 'benchmarks/field_sweep'); from workload_specs import WORKLOAD_NAMES; print('\n'.join(WORKLOAD_NAMES))")

launch() {  # bit_width n_procs [exclude...]
  local bw=$1 nproc=$2; shift 2
  local ws=()
  for w in "${ALL[@]}"; do [[ " $* " == *" $w "* ]] || ws+=("$w"); done
  for ((i = 0; i < nproc; i++)); do
    local part=()
    for ((j = i; j < ${#ws[@]}; j += nproc)); do part+=("${ws[j]}"); done
    timeout 7200 "$PY" -u "$TOOL" --bit-width "$bw" --layout element --point-mode specialized \
      --workloads "${part[@]}" --out "$OUT/bw${bw}_element_specialized_p$i.jsonl" \
      > "$OUT/bw${bw}_element_specialized_p$i.log" 2>&1 &
  done
}

launch 32 2
launch 64 4
launch 128 6 zk_jellyfish_zerocheck_hp
launch 256 12 zk_jellyfish_zerocheck_hp
# The widest workload, which the paper reports as never finishing compilation at 256 bits:
# time its compile separately at 128 and 256 bits.
for bw in 128 256; do
  ( start=$(date +%s)
    timeout 7200 "$PY" -u "$TOOL" --bit-width "$bw" --layout element --point-mode specialized \
      --workloads zk_jellyfish_zerocheck_hp --rounds 3 --sha3-rounds \
      --out "$OUT/bw${bw}_jellyfish_zerocheck.jsonl"
    echo "bw$bw jellyfish_zerocheck exit=$? seconds=$(( $(date +%s) - start ))"
  ) > "$OUT/bw${bw}_jellyfish_zerocheck_compile_probe.log" 2>&1 &
done
wait
echo VALIDATION_DONE
