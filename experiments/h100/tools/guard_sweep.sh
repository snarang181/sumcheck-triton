#!/usr/bin/env bash
# Keep ablation compiles off the machine while the sweep runs: when start_when_ready.sh
# reports that the warm-up is over, stop every precompile_treduce job (compiled kernels
# stay cached; rerun run_precompile_treduce.sh after the sweep to finish the rest).
cd "$(dirname "$0")/../../.."
until grep -q "compile jobs finished" experiments/h100/logs/start_when_ready.log; do sleep 15; done
pkill -f "run_precompile_treduce[.]sh"
pkill -f "xargs -P [0-9]* -L 1 sh -c timeout 14400 python experiments/h100/tools/precompile_treduce"
pkill -f "python experiments/h100/tools/precompile_treduce[.]py"
sleep 5
echo "$(date -u +%FT%TZ) sweep starting: stopped precompile_treduce jobs; left running: $(pgrep -fc 'precompile_treduce[.]py')"
