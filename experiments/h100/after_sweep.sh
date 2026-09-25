#!/usr/bin/env bash
# Follow-up experiments, run automatically once the sweep has finished. Timing-sensitive
# steps run alone; compiles overlap only with CUDA-only GPU-kernel-time measurements.
# Commits, pushes and pings Slack after each step (push_checkpoints.sh stops at sweep end).
set -uo pipefail
cd "$(dirname "$0")/../.."
H=experiments/h100
source $H/env_table5.sh
SWEEP_CACHE="$PWD/.triton-cache-h100-table5"
TREDUCE_CACHE="$PWD/.triton-cache-h100-treduce"
ZK=$(python3 -c "import sys; sys.path.insert(0, 'benchmarks/field_sweep'); from workload_specs import ZK_WORKLOAD_NAMES as z; print(' '.join(z))")

log() { echo "$(date -u +%FT%TZ) $*"; }
checkpoint() {  # message
  git add $H
  git diff --cached --quiet || git commit -q -m "h100 follow-ups: $1" -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
  ( source ~/env.sh
    GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
      -c 'credential.helper=!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' \
      push -q https://github.com/snarang181/zk-duel.git HEAD:refs/heads/h100-experiments ) \
    && log "pushed $(git rev-parse --short HEAD)" || log "push failed"
  $H/notify_slack.sh "zkDuel H100 follow-ups: $1 (pushed $(git rev-parse --short HEAD))."
}

log "waiting for the sweep to finish"
until [ -f $H/sweep/finished_at.txt ]; do sleep 60; done
sleep 120   # let push_checkpoints.sh make its final push first
python3 $H/tools/summarize_sweep.py $H/sweep/triton_cuda_field_resilient_r14_28_h100.csv --label "H100" --md $H/sweep/summary_h100.md > /dev/null
python3 $H/tools/summarize_sweep.py experiments/a100/triton_cuda_field_resilient_r14_28_new.csv --label "A100" --md $H/sweep/summary_a100.md > /dev/null
python3 $H/tools/classify_skips.py $H/sweep/triton_cuda_field_resilient_r14_28_h100.csv --md $H/sweep/skips_h100.md > /dev/null
checkpoint "sweep summaries (Tables 2/4, skip classes)"

log "step 1: r=28 cases with more than 2^31 table elements, int64-offset fix"
python $H/tools/diag_overhead.py --label r28_int64_fix --triton-script $H/tools/triton_sweep_fixed.py \
  --triton-cache-dir "$SWEEP_CACHE" --cuda-binary build/field_sumcheck_cuda_int64_r28 \
  --workloads zk_vanilla_zerocheck_hp zk_vanilla_permcheck_hp zk_jellyfish_zerocheck_hp zk_jellyfish_permcheck_hp zk_opencheck \
  --bit-widths 32 64 --rounds 28 --out $H/diag/r28_int64_fix.jsonl > $H/logs/diag_r28.log 2>&1
checkpoint "r=28 reruns with the int64-offset fix"

log "step 2: wall vs GPU-kernel time, all 25 workloads, 32/256-bit, r=20/24"
python $H/tools/diag_overhead.py --label baseline_all --triton-cache-dir "$SWEEP_CACHE" \
  --workloads $ZK --bit-widths 32 256 --rounds 20 24 --out $H/diag/baseline_all.jsonl > $H/logs/diag_baseline_all.log 2>&1
checkpoint "wall vs GPU-kernel split for all 25 workloads"

log "step 3: finish tl.reduce precompile; meanwhile the CUDA occupancy ablation (GPU-time metric)"
JOBS=22 $H/tools/run_precompile_treduce.sh > $H/logs/precompile_treduce_after.log 2>&1 &
PRE=$!
for bw in 32 256; do
  base=$((128 * bw / 32 * 4))
  for total in 0 21000 44000 56000 110000; do
    extra=0; [ "$total" -gt 0 ] && extra=$((total - base))
    python $H/tools/diag_overhead.py --label occ_smem${total} --skip-triton --cuda-binary build/field_sumcheck_cuda_occ \
      --cuda-env ZKDUEL_EVAL_EXTRA_SMEM=$extra --workloads poly_abc zk_spartan_2 zk_witness_id_point_1 zk_complete_add_3 zk_vanilla_permcheck_hp \
      --bit-widths $bw --rounds 20 24 --out $H/diag/occupancy.jsonl >> $H/logs/diag_occupancy.log 2>&1
  done
done
checkpoint "CUDA occupancy ablation (shared-memory padding)"
wait $PRE
checkpoint "tl.reduce ablation kernels compiled"

log "step 4: tl.reduce ablation, all 25 workloads, 32/256-bit, r=20/24"
python $H/tools/diag_overhead.py --label treduce_all --triton-script $H/tools/triton_sweep_treduce.py \
  --triton-cache-dir "$TREDUCE_CACHE" --workloads $ZK --bit-widths 32 256 --rounds 20 24 \
  --out $H/diag/treduce_all.jsonl > $H/logs/diag_treduce_all.log 2>&1
checkpoint "tl.reduce ablation for all 25 workloads"
log "all follow-ups done"
