#!/usr/bin/env bash
# Follow-up to run_fixed_sweep.sh: the §5.3 wall-clock vs GPU-kernel split for all 25
# zkPHIRE constraints (the pipeline's own diagnostics step omitted --workloads and so covers only
# diag_overhead.py's default three). Waits for the pipeline to finish, probes host latency before
# and after (same reference configuration as the pipeline's gate), runs the split with the fixed
# CUDA binary, then commits and pushes the result.
cd "$(dirname "$0")/../.."
source experiments/h100/env_table5.sh
export TRITON_CACHE_DIR="$PWD/.triton-cache-h100-table5"
H=experiments/h100
OUT=$H/sweep_fixed
BIN=build/field_sumcheck_cuda_fixed
log() { echo "$(date -u +%FT%TZ) $*"; }
probe() {
  $BIN --workloads zk_spartan_2 --bit-widths 32 --rounds 24 --warmups 2 --repeats 10 \
    --point-mode specialized --layout element --timer events --out $OUT/probe.csv > /dev/null 2>&1
  echo "$(date -u +%FT%TZ) probe($1) cuda_ms=$(python3 -c "import csv; print(next(csv.DictReader(open('$OUT/probe.csv')))['median_ms'])")" >> $OUT/probe.log
}

: # sweep already finished

# Latency gate, as in the sweep: the host's launch latency drifts (e.g. while the orbit/osquery
# agent restarts). Probe every 5 minutes until within 5% of the 2.90 ms baseline, or deadline.
deadline=$(date -ud "23:30" +%s)
while :; do
  probe split32-gate
  ms=$(tail -1 $OUT/probe.log | sed 's/.*cuda_ms=//')
  python3 -c "import sys; sys.exit(0 if float('$ms') <= 2.90 * 1.05 else 1)" && break
  [ "$(date +%s)" -ge "$deadline" ] && { log "gate deadline reached ($ms ms); proceeding"; break; }
  sleep 300
done
log "gate passed ($ms ms)"
ZK=$(PYTHONPATH=benchmarks/field_sweep:src python -c "from workload_specs import ZK_WORKLOAD_NAMES as Z; print(' '.join(Z))")
log "split32: all 25 constraints, 32 bits, r=20/24 (quiet host)"
python $H/tools/diag_overhead.py --label baseline_fixed_32 --cuda-binary $BIN \
  --triton-cache-dir "$TRITON_CACHE_DIR" --workloads $ZK --bit-widths 32 --rounds 20 24 \
  --out $OUT/baseline_fixed_32.jsonl > $H/logs/fixed_split32.log 2>&1
probe split32-after
log "split done: $(wc -l < $OUT/baseline_fixed_32.jsonl) records"
git add $OUT/baseline_fixed_32.jsonl $OUT/probe.log $H/logs/fixed_split32.log $H/diag/raw/baseline_fixed_32 $0
git commit -q -m "Fixed-harness H100: 32-bit split rerun on a quiet host

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
( set -a; . ~/env.sh >/dev/null 2>&1; set +a
  git -c credential.helper='!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' \
    push -q origin h100-experiments ) > /dev/null 2>&1 && log "pushed" || log "push failed"
$H/notify_slack.sh "Fixed-harness H100 32-bit split rerun done." || true
log "all done"
