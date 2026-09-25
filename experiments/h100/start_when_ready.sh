#!/usr/bin/env bash
# Wait until no Triton compile job is running (cache warmer, validation, jellyfish probes),
# validate the 128/256-bit kernels against the registry with the warm cache, then run the
# sweep. Keeps the sweep's timings free of concurrent compile load.
set -uo pipefail
cd "$(dirname "$0")/../.."
H=experiments/h100
busy() { ps -eo args | grep -v -e grep -e shell-snapshots | grep -c -E "validate_vs_registry[.]py|warm_cache[.]py|run_validation[.]sh|run_warm_cache[.]sh"; }
echo "$(date -u +%FT%TZ) waiting for compile jobs"
while [ "$(busy)" -gt 0 ]; do sleep 60; done
echo "$(date -u +%FT%TZ) compile jobs finished; post-warm validation"
experiments/h100/notify_slack.sh "zkDuel H100: Triton cache warm; running post-warm validation, then the 800-config sweep starts."
source $H/env_table5.sh
export TRITON_CACHE_DIR="$PWD/.triton-cache-h100-table5"
for bw in 128 256; do
  timeout 3600 python -u $H/tools/validate_vs_registry.py --bit-width $bw --layout element \
    --point-mode specialized --exclude zk_jellyfish_zerocheck_hp \
    --out $H/validation/post_warm_bw${bw}.jsonl > $H/validation/post_warm_bw${bw}.log 2>&1 &
done
wait
bad=$(cat $H/validation/post_warm_bw*.jsonl | grep -c -v '"match": true')
echo "$(date -u +%FT%TZ) post-warm validation done; non-matching rows: $bad"
[ "$bad" -eq 0 ] || experiments/h100/notify_slack.sh "zkDuel H100: post-warm validation has $bad non-matching rows; starting the sweep anyway (see validation/post_warm_*.jsonl)."
echo "$(date -u +%FT%TZ) starting sweep"
$H/run_sweep.sh > $H/logs/sweep.log 2>&1
echo "$(date -u +%FT%TZ) sweep exited rc=$?"
