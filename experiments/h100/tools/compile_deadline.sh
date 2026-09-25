#!/usr/bin/env bash
# Hard deadline for the Triton warm-up: at DEADLINE (UTC), stop any compile still running
# so start_when_ready.sh can start the sweep within the GPU time budget. Kernels that miss
# the deadline compile inside their sweep case instead (subject to --per-case-timeout).
DEADLINE="${DEADLINE:-2026-09-25T00:30:00Z}"
now=$(date -u +%s); end=$(date -u -d "$DEADLINE" +%s)
[ "$end" -gt "$now" ] && sleep $((end - now))
left=$(ps -eo pid,args | grep -v -e grep -e shell-snapshots | grep "python experiments/h100/tools/warm_cache[.]py" | awk '{print $1}')
echo "$(date -u +%FT%TZ) deadline reached; stopping $(echo $left | wc -w) compile(s)"
ps -eo args | grep -v -e grep -e shell-snapshots | grep "python experiments/h100/tools/warm_cache[.]py" | sed 's/^/  unfinished: /'
[ -n "$left" ] && kill $left
