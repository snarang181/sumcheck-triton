#!/usr/bin/env bash
# Fixed-harness H100 sweeps with the int64-index fix in both backends:
#   0. build a full CUDA binary from the fixed source (build/field_sumcheck_cuda_fixed)
#   1. diagnostic tier (7 polynomials) for Figure 4
#   2. zkPHIRE tier, 25 constraints at 32/64/128 bits and 24 at 256 bits
#   3. wall-clock vs GPU-kernel split (diag_overhead.py) with the same binary
# CUDA whole-run timing: synchronized host timer (same as Triton); median = mean of middle two.
# Same settings as the main sweep: warmups 2, repeats 10, specialized points, element layout,
# whole-run timing, fixed challenges, 1800 s per-case timeout. zk_jellyfish_zerocheck_hp at
# 256 bits is not rerun: its eval kernels take hours to compile (T12), so it would only repeat
# the six 1800 s timeouts of the first H100 sweep.
# Phases run one after another, so no timing overlaps a compile. Each phase is committed and
# pushed to h100-experiments (token from ~/env.sh via a one-off credential helper).
set -uo pipefail
cd "$(dirname "$0")/../.."
source experiments/h100/env_table5.sh
export TRITON_CACHE_DIR="$PWD/.triton-cache-h100-table5"
H=experiments/h100
OUT=$H/sweep_fixed
BIN=build/field_sumcheck_cuda_fixed
mkdir -p $OUT $H/logs
log() { echo "$(date -u +%FT%TZ) $*"; }

push() {
  git add $OUT $H/logs/fixed_sweep.log $H/logs/cuda_build_fixed.log $0 2>/dev/null
  git diff --cached --quiet && return 0
  git commit -q -m "Fixed-harness H100 sweep (fixed harness): $1

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
  ( set -a; . ~/env.sh >/dev/null 2>&1; set +a
    git -c credential.helper='!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' \
      push -q origin h100-experiments ) > /dev/null 2>&1 && log "pushed: $1" || log "push failed: $1"
}

sweep() {  # sweep NAME ARGS...
  local name=$1; shift
  local cmd=(python -u benchmarks/field_sweep/resilient_compare_triton_cuda.py "$@"
    --rounds 14 16 18 20 22 24 26 28 --warmups 2 --repeats 10
    --point-mode specialized --per-case-timeout 1800
    --skip-build --cuda-binary $BIN --python "$ZKDUEL_VENV/bin/python"
    --tmp-dir $OUT/cases_$name --out $OUT/$name.csv)
  printf '%q ' "${cmd[@]}" > $OUT/$name.command.txt
  echo "TRITON_CACHE_DIR=$TRITON_CACHE_DIR" >> $OUT/$name.command.txt
  log "sweep $name: start"
  "${cmd[@]}" > $H/logs/fixed_$name.log 2>&1
  log "sweep $name: done rc=$? rows=$(($(wc -l < $OUT/$name.csv) - 1))"
  push "$name"
}

# Host-latency gate: this host's launch/sync latency drifted during the day (GPU kernel time
# unchanged). Probe one reference configuration (zk_spartan_2, 32-bit, r=24; CUDA median
# 2.90 ms at 12:40 UTC) every 10 minutes and start the timed phases once it is within 5%,
# or at the deadline, recording which.
gate() {
  local deadline=$(date -ud "22:30" +%s) ref=2.90 csv=$OUT/probe.csv
  while :; do
    $BIN --workloads zk_spartan_2 --bit-widths 32 --rounds 24 --warmups 2 --repeats 10 \
      --point-mode specialized --layout element --timer events --out $csv > /dev/null 2>&1
    local ms=$(python3 -c "import csv; print(next(csv.DictReader(open('$csv')))['median_ms'])")
    echo "$(date -u +%FT%TZ) probe cuda_ms=$ms" >> $OUT/probe.log
    if python3 -c "import sys; sys.exit(0 if float('$ms') <= $ref * 1.05 else 1)"; then
      log "gate: host latency back to baseline (cuda $ms ms)"; return 0; fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      log "gate: deadline reached with cuda $ms ms (baseline $ref); proceeding"; return 0; fi
    sleep 600
  done
}

log "phase 0: CUDA build from fixed source"
t0=$(date +%s)
nvcc -O3 -std=c++17 -arch=native \
  -I"${ZKDUEL_OPENSSL_INCLUDE}" -I"${ZKDUEL_OPENSSL_INCLUDE}/x86_64-linux-gnu" \
  cuda/src/field_sumcheck.cu -o $BIN \
  -L/usr/lib/x86_64-linux-gnu -l:libcrypto.so.3 > $H/logs/cuda_build_fixed.log 2>&1 \
  || { log "CUDA build failed"; exit 1; }
echo "BUILD_OK seconds=$(( $(date +%s) - t0 )) sha256=$(sha256sum $BIN | cut -c1-16) source=$(git rev-parse --short HEAD)" >> $H/logs/cuda_build_fixed.log
log "phase 0 done: $(tail -1 $H/logs/cuda_build_fixed.log)"
python $H/tools/capture_env.py --out-dir $OUT/env --label h100_fixed > /dev/null 2>&1 || true
gate
date -u +%FT%TZ > $OUT/started_at.txt

sweep diag_tier --workloads poly_a poly_ab poly_ab_plus_c poly_abc poly_aabbc poly_abc_plus_de poly_abcg_plus_deg \
  --bit-widths 32 64 128 256
sweep zk_32_128 --zk-only --bit-widths 32 64 128
ZK256=$(PYTHONPATH=benchmarks/field_sweep:src python -c "from workload_specs import ZK_WORKLOAD_NAMES as Z; print(' '.join(w for w in Z if w != 'zk_jellyfish_zerocheck_hp'))")
sweep zk_256 --workloads $ZK256 --bit-widths 256

log "diagnostics: wall-clock vs GPU-kernel split (Table split) with the fixed binary"
python $H/tools/diag_overhead.py --label baseline_fixed --cuda-binary $BIN \
  --triton-cache-dir "$TRITON_CACHE_DIR" --bit-widths 32 256 --rounds 20 24 \
  --out $OUT/baseline_fixed.jsonl > $H/logs/fixed_diag.log 2>&1
log "diagnostics done: $(wc -l < $OUT/baseline_fixed.jsonl) records"
date -u +%FT%TZ > $OUT/finished_at.txt
push "finished"
$H/notify_slack.sh "Fixed-harness H100 sweep (fixed harness) finished: $(for f in $OUT/*.csv; do printf '%s %s rows; ' $(basename $f .csv) $(($(wc -l < $f) - 1)); done) Pushed to h100-experiments." || true
log "all done"
