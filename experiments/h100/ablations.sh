#!/usr/bin/env bash
# Ablations of the design choices: (a) the two optimizations the paper proposes, layout
# (element- vs limb-major) x interpolation (generic vs specialized points), on both
# backends; (b) CUDA thread-block size, to show the CUDA baseline is tuned. Same sweep
# settings (warmups 2, repeats 10, fixed challenge, whole-run), wall clock and nsys kernel time.
set -uo pipefail
cd "$(dirname "$0")/../.."
H=experiments/h100
source $H/env_table5.sh
CACHE="$PWD/.triton-cache-h100-table5"
W="zk_spartan_2 zk_witness_id_point_1 zk_complete_add_3 zk_vanilla_permcheck_hp zk_opencheck"
log() { echo "$(date -u +%FT%TZ) $*"; }

log "compile: Triton kernels for element/generic, limb/specialized, limb/generic"
python3 - "$W" > $H/logs/compile_tasks.txt <<'PY'
import sys
sys.path.insert(0, "benchmarks/field_sweep")
from workload_specs import WORKLOAD_BY_NAME as S
tasks = []
for layout, pm in (("element", "generic"), ("limb", "specialized"), ("limb", "generic")):
    for bw in (256, 32):
        for w in sys.argv[1].split():
            for p in range(S[w].degree + 1):
                tasks.append(f"{w} {bw} {p} {pm} {layout}")
print("\n".join(tasks))
PY
TRITON_CACHE_DIR="$CACHE" xargs -P 24 -L 1 sh -c 'timeout 14400 python experiments/h100/tools/warm_cache.py "$0" "$1" "$2" --point-mode "$3" --layout "$4" || echo "FAILED $0 $1 $2 $3 $4 rc=$?"' \
  < $H/logs/compile_tasks.txt > $H/logs/compile.log 2>&1
log "compile done: $(grep -c point $H/logs/compile.log) ok, $(grep -c FAILED $H/logs/compile.log) failed"

log "timing: layout x interpolation, both backends"
for layout in element limb; do
  for pm in specialized generic; do
    python $H/tools/diag_overhead.py --label abl_${layout}_${pm} --layout $layout --point-mode $pm \
      --triton-cache-dir "$CACHE" --workloads $W --bit-widths 32 256 --rounds 20 24 \
      --out $H/diag/layout_points.jsonl >> $H/logs/ablations_diag.log 2>&1
  done
done
log "timing: CUDA thread-block size"
for b in 64 128 256 512; do
  python $H/tools/diag_overhead.py --label cudablock_$b --skip-triton --cuda-args "--block $b" \
    --workloads $W --bit-widths 32 256 --rounds 20 24 --out $H/diag/cuda_block.jsonl >> $H/logs/ablations_diag.log 2>&1
done
log "done"
