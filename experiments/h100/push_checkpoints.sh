#!/usr/bin/env bash
# Commit experiments/h100 and push it to the h100-experiments branch every INTERVAL
# seconds, until the sweep writes experiments/h100/sweep/finished_at.txt. After each push,
# post progress to Slack via notify_slack.sh (a no-op until Slack credentials are in ~/env.sh).
# The GitHub token is read from ~/env.sh (GITHUB_TOKEN) by a one-off credential helper, so
# it is never written to git config, the remote URL, or the command line.
#
# Run detached:  setsid nohup experiments/h100/push_checkpoints.sh > experiments/h100/logs/push.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
INTERVAL="${INTERVAL:-1800}"
BRANCH=h100-experiments
REMOTE_URL=https://github.com/snarang181/zk-duel.git
CSV="${CSV:-experiments/h100/sweep/triton_cuda_field_resilient_r14_28_h100.csv}"
FINISHED="${FINISHED:-experiments/h100/sweep/finished_at.txt}"
STARTED="${STARTED:-experiments/h100/sweep/started_at.txt}"
NAME="${NAME:-zkDuel H100 sweep}"

progress() {
  python3 - "$CSV" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open())) if p.exists() else []
ok = sum(r["triton_status"] == "ok" and r["cuda_status"] == "ok" and r["checksum_match"] == "true" for r in rows)
skipped = sum(r["triton_status"] == "skipped" for r in rows)
mismatch = sum(r["checksum_match"] == "false" for r in rows)
last = f"; last: {rows[-1]['workload']} {rows[-1]['bit_width']}-bit r={rows[-1]['rounds']}" if rows else ""
print(f"{len(rows)}/800 cases ({ok} validated, {skipped} skipped, {mismatch} checksum mismatches){last}")
PY
}

push_once() {
  git add experiments/h100
  if ! git diff --cached --quiet; then
    local done_cases=0
    [ -f "$CSV" ] && done_cases=$(($(wc -l < "$CSV") - 1))
    git commit -q -m "h100: sweep checkpoint (${done_cases}/800 cases recorded)" \
      -m "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
  fi
  (
    # shellcheck disable=SC1090
    source ~/env.sh
    GIT_TERMINAL_PROMPT=0 git -c credential.helper= \
      -c 'credential.helper=!f() { echo username=x-access-token; echo "password=$GITHUB_TOKEN"; }; f' \
      push -q "$REMOTE_URL" "HEAD:refs/heads/$BRANCH"
  )
}

while true; do
  if push_once; then
    echo "$(date -u +%FT%TZ) pushed $(git rev-parse --short HEAD)"
    if [ -f "$STARTED" ]; then
      experiments/h100/notify_slack.sh "$NAME: $(progress). Pushed $(git rev-parse --short HEAD) to $BRANCH."
    fi
  else
    echo "$(date -u +%FT%TZ) push failed; will retry"
    experiments/h100/notify_slack.sh "$NAME: GitHub push failed at $(date -u +%FT%TZ); will retry."
  fi
  if [ -f "$FINISHED" ]; then
    push_once && echo "$(date -u +%FT%TZ) final push $(git rev-parse --short HEAD)"
    experiments/h100/notify_slack.sh "$NAME FINISHED: $(progress). Final push $(git rev-parse --short HEAD) to $BRANCH."
    break
  fi
  sleep "$INTERVAL"
done
