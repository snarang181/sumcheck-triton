#!/usr/bin/env bash
# Hardware-counter anatomy of matched eval kernels (Nsight Compute): executed instructions,
# achieved occupancy, issue efficiency, warp stall reasons. One launch per kernel: round 0,
# point 2, rounds=20. Triton (original and tl.reduce) and CUDA, element layout,
# specialized points. ncu needs root here (RmProfilingAdminOnly=1), so root gets private
# copies of the Triton caches.
#   experiments/h100/tools/ncu_anatomy.sh [CASES...]   (CASE = workload:bits)
# Optional overrides (defaults are the original run's): ZKDUEL_DIAG_DIR (output goes to
# $ZKDUEL_DIAG_DIR/ncu), ZKDUEL_SWEEP_CACHE, ZKDUEL_TREDUCE_CACHE (Triton caches to copy),
# ZKDUEL_CUDA_BIN, NCU (ncu executable), NCU_TMP (prefix of root's scratch copies).
cd "$(dirname "$0")/../../.."
source experiments/h100/env_table5.sh
OUT=${ZKDUEL_DIAG_DIR:-experiments/h100/diag}/ncu; mkdir -p $OUT
NCU=${NCU:-/usr/local/bin/ncu}
NCU_TMP=${NCU_TMP:-/tmp/zkduel-ncu}
CUDA_BIN=${ZKDUEL_CUDA_BIN:-build/field_sumcheck_cuda}
CASES=("${@:-zk_spartan_2:32 zk_spartan_2:256 zk_complete_add_3:32 zk_complete_add_3:256 zk_witness_id_point_1:256 zk_vanilla_permcheck_hp:256}")
[ $# -gt 0 ] || CASES=(zk_spartan_2:32 zk_spartan_2:256 zk_complete_add_3:32 zk_complete_add_3:256 zk_witness_id_point_1:256 zk_vanilla_permcheck_hp:256)
for v in base treduce; do
  if [ $v = base ]; then src=${ZKDUEL_SWEEP_CACHE:-$PWD/.triton-cache-h100-table5}; else src=${ZKDUEL_TREDUCE_CACHE:-$PWD/.triton-cache-h100-treduce}; fi
  [ -d $NCU_TMP-cache-$v ] || cp -r "$src" $NCU_TMP-cache-$v
done
SECTIONS="--section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section InstructionStats"
run() {  # tag kernel-regex cmd...
  local tag=$1 k=$2; shift 2
  sudo env LD_LIBRARY_PATH="$LD_LIBRARY_PATH" TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-}" PYTHONPATH=src \
    TRITON_CACHE_DIR="${CACHE:-}" HOME=$NCU_TMP-home \
    "$NCU" --target-processes all -k "regex:$k" --launch-skip 2 --launch-count 1 $SECTIONS \
    --csv --page details "$@" > $OUT/$tag.csv 2> $OUT/$tag.log
  echo "$tag rc=$? rows=$(grep -c '"' $OUT/$tag.csv)"
}
for c in "${CASES[@]}"; do
  w=${c%%:*}; bw=${c##*:}
  common=(--workloads $w --bit-widths $bw --rounds 20 --warmups 0 --repeats 1 --point-mode specialized)
  CACHE=$NCU_TMP-cache-base run ${w}_bw${bw}_triton "_round_eval_specialized_kernel" \
    $ZKDUEL_VENV/bin/python benchmarks/field_sweep/triton_sweep.py "${common[@]}" --out $NCU_TMP-t.csv
  CACHE=$NCU_TMP-cache-treduce run ${w}_bw${bw}_treduce "_round_eval_specialized_kernel" \
    $ZKDUEL_VENV/bin/python experiments/h100/tools/triton_sweep_treduce.py "${common[@]}" --out $NCU_TMP-r.csv
  run ${w}_bw${bw}_cuda "eval_kernel" "$CUDA_BIN" "${common[@]}" --out $NCU_TMP-c.csv
done
