#!/usr/bin/env bash
# A100 kernel-level follow-ups (GPU-kernel split, tl.reduce, occupancy). See experiments/a100/followups/README.md.
#
# Run from the repo root on the A100 host, in the
# paper's Table 5 environment, with the GPU idle:
#   experiments/a100/followups/run_a100_followups.sh             # full run
#   QUICK=1 experiments/a100/followups/run_a100_followups.sh     # smaller 256-bit subset
#   PHASES="0 1 2" ...                                           # skip the cold-compile repro
#   PUSH=1 ...                                                   # push to branch a100-followups after each phase
set -uo pipefail
cd "$(dirname "$0")/../../.."
O="${OUT_DIR:-experiments/a100/followups}"
T=experiments/h100/tools
mkdir -p $O/logs $O/diag $O/env
PY="${PY:-python}"
export PYTHONPATH=src PYTHONUNBUFFERED=1
JOBS="${JOBS:-$(nproc)}"
PHASES="${PHASES:-0 1 2 3}"
BASE_CACHE="${BASE_CACHE:-$PWD/.triton-cache-sumcheck-field}"   # triton_sweep.py's default dir
TRED_CACHE="${TRED_CACHE:-$PWD/.triton-cache-a100-treduce}"
CUDA_BIN="${CUDA_BIN:-build/field_sumcheck_cuda}"
OCC_BIN=build/field_sumcheck_cuda_occ_a100
ZK="${ZK_LIST:-$($PY -c "import sys; sys.path.insert(0, 'benchmarks/field_sweep'); from workload_specs import ZK_WORKLOAD_NAMES as z; print(' '.join(z))")}"
if [ -n "${SUB256_LIST:-}" ]; then
  SUB256="$SUB256_LIST"
elif [ -n "${QUICK:-}" ]; then
  SUB256="zk_spartan_2 zk_witness_id_point_1 zk_complete_add_3 zk_vanilla_permcheck_hp"
else
  SUB256="zk_verifiable_asics zk_spartan_2 zk_witness_id_point_1 zk_incomplete_add_2 zk_complete_add_3 zk_complete_add_7 zk_vanilla_permcheck_hp zk_opencheck"
fi
OCC_WORKLOADS="${OCC_LIST:-zk_spartan_2 zk_witness_id_point_1 zk_complete_add_3 zk_vanilla_permcheck_hp}"
R28_WORKLOADS="${R28_LIST:-zk_vanilla_zerocheck_hp zk_vanilla_permcheck_hp zk_jellyfish_permcheck_hp zk_opencheck}"

log() { echo "$(date -u +%FT%TZ) $*" | tee -a $O/logs/run.log; }
die() { log "ERROR: $*"; exit 1; }
has() { [[ " $PHASES " == *" $1 "* ]]; }
checkpoint() {
  git add $O && { git diff --cached --quiet || git commit -q -m "a100 follow-ups: $1"; }
  if [ -n "${PUSH:-}" ]; then git push -q origin HEAD:refs/heads/a100-followups && log "pushed $(git rev-parse --short HEAD)"; fi
}
quiet_gpu() {
  local n; n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c . || true)
  [ "$n" -eq 0 ] || die "GPU busy ($n compute processes); stop them first"
}

if has 0; then
  log "phase 0: preflight"
  command -v nsys > /dev/null || die "nsys not found; install Nsight Systems (README) and rerun"
  $PY -c "import torch, triton; assert torch.cuda.is_available(); print(torch.__version__, triton.__version__)" \
    > $O/logs/stack.txt 2>&1 || die "torch/triton/CUDA not usable with $PY (see logs/stack.txt)"
  quiet_gpu
  $PY $T/capture_env.py --out-dir $O/env --label a100 > $O/logs/capture_env.log 2>&1 || log "capture_env failed (non-fatal)"
  if [ -d "$BASE_CACHE" ]; then log "reusing Triton cache $BASE_CACHE ($(ls "$BASE_CACHE" | wc -l) entries)"; else log "no Triton cache at $BASE_CACHE; kernels will be compiled"; fi
  if [ -x "$CUDA_BIN" ] && "$CUDA_BIN" --workloads zk_opencheck --bit-widths 32 --rounds 4 --warmups 0 --repeats 1 \
       --point-mode specialized --out /tmp/zkduel_preflight.csv > /dev/null 2>&1; then
    log "using existing CUDA binary $CUDA_BIN (built $(date -u -r "$CUDA_BIN" +%F))"
  else
    log "building $CUDA_BIN from the current source (~1 h)"
    cuda/build/build_field_sumcheck.sh > $O/logs/cuda_build.log 2>&1 || die "CUDA build failed (logs/cuda_build.log)"
  fi
  { echo "CUDA_BIN=$CUDA_BIN ($(sha256sum "$CUDA_BIN" | cut -c1-16))"; echo "BASE_CACHE=$BASE_CACHE"; echo "TRED_CACHE=$TRED_CACHE";
    echo "SUB256=$SUB256"; echo "git=$(git rev-parse --short HEAD)"; } > $O/env/run_config.txt
  checkpoint "preflight and environment"
fi

if has 1; then
  log "phase 1: compile (CPU only): Triton kernels for both variants, occupancy binary"
  $PY - "$ZK" "$SUB256" "$R28_WORKLOADS" > $O/logs/compile_tasks.txt <<'PY'
import sys
sys.path.insert(0, "benchmarks/field_sweep")
from workload_specs import WORKLOAD_BY_NAME as W
zk, sub256, r28 = (s.split() for s in sys.argv[1:4])
tasks = [(w, 32) for w in zk] + [(w, 64) for w in r28] + [(w, 256) for w in sub256]
# heaviest first so the long 256-bit kernels start early
tasks.sort(key=lambda t: (-t[1], -W[t[0]].terms * sum(W[t[0]].term_lens)))
for variant in ("base", "treduce"):
    for w, bw in tasks:
        for p in range(W[w].degree + 1):
            print(variant, w, bw, p)
PY
  log "  $(wc -l < $O/logs/compile_tasks.txt) kernel compiles, $JOBS in parallel"
  ( tmp=$(mktemp -d); mkdir -p "$tmp/cuda/src"; cp cuda/src/field_sumcheck.cu "$tmp/cuda/src/"
    (cd "$tmp" && patch -s -p1 < "$OLDPWD/experiments/h100/patches/cuda_occupancy_knob.patch") &&
    $PY - "$tmp/cuda/src/field_sumcheck.cu" <<'PY'
import re, sys
path = sys.argv[1]; src = open(path).read()
start = src.index("#define WORKLOAD_LIST(X)"); end = src.index("\n\n", start)
table = dict(re.findall(r"X\((\d+),\s*(\d+)\)", src[start:end]))
ids = ["10", "15", "20", "29"]   # spartan_2, witness_id_point_1, complete_add_3, vanilla_permcheck
open(path, "w").write(src[:start] + "#define WORKLOAD_LIST(X) \\\n  " + " ".join(f"X({i}, {table[i]})" for i in ids) + src[end:])
PY
    nvcc -O3 -std=c++17 -arch="${CUDA_ARCH:-native}" "$tmp/cuda/src/field_sumcheck.cu" -o $OCC_BIN ${NVCC_EXTRA:-} ${NVCC_LIBS:--lcrypto}
    echo "occupancy binary exit=$?"; rm -rf "$tmp" ) > $O/logs/build_occ.log 2>&1 &
  OCC_PID=$!
  xargs -P "$JOBS" -L 1 sh -c '
    if [ "$0" = base ]; then export TRITON_CACHE_DIR="'"$BASE_CACHE"'"; tool='"$T"'/warm_cache.py
    else export TRITON_CACHE_DIR="'"$TRED_CACHE"'"; tool='"$T"'/precompile_treduce.py; fi
    timeout 21600 '"$PY"' "$tool" "$1" "$2" "$3" || echo "FAILED $0 $1 $2 $3 rc=$?"' \
    < $O/logs/compile_tasks.txt > $O/logs/compile.log 2>&1
  wait $OCC_PID
  log "  compiles done: $(grep -c point $O/logs/compile.log) ok, $(grep -c FAILED $O/logs/compile.log) failed; $(tail -1 $O/logs/build_occ.log)"
  checkpoint "phase 1 compile logs"
fi

if has 2; then
  log "phase 2: timing on a quiet GPU"
  quiet_gpu
  D="--diag-dir $O/diag"
  for v in base treduce; do
    if [ $v = base ]; then args=(--triton-cache-dir "$BASE_CACHE"); else args=(--triton-script $T/triton_sweep_treduce.py --triton-cache-dir "$TRED_CACHE"); fi
    log "  $v: 32-bit, all 25 workloads, r=20/24"
    $PY $T/diag_overhead.py $D --label ${v}_all "${args[@]}" --cuda-binary "$CUDA_BIN" --workloads $ZK \
      --bit-widths 32 --rounds 20 24 --out $O/diag/${v}_all.jsonl >> $O/logs/diag_$v.log 2>&1
    log "  $v: 256-bit subset, r=20/24"
    $PY $T/diag_overhead.py $D --label ${v}_all "${args[@]}" --cuda-binary "$CUDA_BIN" --workloads $SUB256 \
      --bit-widths 256 --rounds 20 24 --out $O/diag/${v}_all.jsonl >> $O/logs/diag_$v.log 2>&1
    checkpoint "phase 2: $v wall vs GPU-kernel split"
  done
  if grep -q "exit=0" $O/logs/build_occ.log 2>/dev/null; then
    log "  occupancy ablation (A100: 164 KB shared memory per SM)"
    for bw in 32 256; do
      base=$((128 * bw / 32 * 4))
      for total in 0 15000 19000 31000 39000 80000; do
        extra=0; [ "$total" -gt 0 ] && extra=$((total - base))
        $PY $T/diag_overhead.py $D --label occ_smem$total --skip-triton --cuda-binary $OCC_BIN \
          --cuda-env ZKDUEL_EVAL_EXTRA_SMEM=$extra --workloads $OCC_WORKLOADS --bit-widths $bw --rounds 20 24 \
          --out $O/diag/occupancy.jsonl >> $O/logs/diag_occupancy.log 2>&1
      done
    done
    checkpoint "phase 2: occupancy ablation"
  else
    log "  occupancy binary missing; skipping the occupancy ablation (logs/build_occ.log)"
  fi
  log "  r=28 cases with more than 2^31 table elements (int64-offset fix in triton_sweep.py)"
  $PY $T/diag_overhead.py $D --label r28_fixed --triton-cache-dir "$BASE_CACHE" --cuda-binary "$CUDA_BIN" \
    --workloads $R28_WORKLOADS --bit-widths 32 64 --rounds 28 --out $O/diag/r28_fixed.jsonl >> $O/logs/diag_r28.log 2>&1
  $PY $T/table3_registers.py --cuda-binary "$CUDA_BIN" --triton-cases --md $O/diag/table3_cuda_sm80.md > /dev/null 2>&1 || true
  A100_DIAG_DIR=$O/diag $PY experiments/a100/followups/compare_gpus.py > $O/COMPARISON.md 2>> $O/logs/run.log || log "compare_gpus failed (non-fatal)"
  checkpoint "phase 2: r=28 reruns, CUDA registers, A100 vs H100 comparison"
fi

if has 3; then
  log "phase 3: cold-compile reproduction of two A100 skips (fresh Triton cache, full logs)"
  COLD="$PWD/.triton-cache-a100-cold-$(date -u +%Y%m%d%H%M)"
  sudo -n dmesg -T > $O/logs/dmesg_before.txt 2>/dev/null || dmesg -T > $O/logs/dmesg_before.txt 2>/dev/null || true
  for w in zk_complete_add_12 zk_vanilla_permcheck_hp; do
    ( TRITON_CACHE_DIR="$COLD-$w" /usr/bin/time -v $PY -u benchmarks/field_sweep/resilient_compare_triton_cuda.py \
        --workloads $w --bit-widths 256 --rounds 14 --warmups 2 --repeats 10 --point-mode specialized \
        --per-case-timeout 14400 --skip-build --cuda-binary "$CUDA_BIN" --python "$PY" \
        --tmp-dir $O/cold/$w --out $O/cold/${w}_bw256_r14.csv ) > $O/logs/cold_$w.log 2>&1 &
  done
  wait
  sudo -n dmesg -T > $O/logs/dmesg_after.txt 2>/dev/null || dmesg -T > $O/logs/dmesg_after.txt 2>/dev/null || true
  log "  $(grep -h -E '^(ok|skipped):' $O/logs/cold_*.log | tr '\n' ' ')"
  checkpoint "phase 3: cold-compile reproduction"
fi
log "done"
