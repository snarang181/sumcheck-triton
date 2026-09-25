#!/usr/bin/env bash
# Hold ablations.sh (first argument) before its timing phase until every other PID given
# has exited, so no timing run overlaps a compile. Usage: hold_ablation_timing.sh ABL PID...
abl=$1; shift
any_alive() { for p in "$@"; do kill -0 "$p" 2>/dev/null && return 0; done; return 1; }
kill -STOP "$abl"
echo "$(date -u +%FT%TZ) holding $abl until $* exit"
while any_alive "$@"; do sleep 30; done
sleep 60
echo "$(date -u +%FT%TZ) resuming ablations.sh"
kill -CONT "$abl"
