#!/usr/bin/env bash
# Post a message to Slack: experiments/h100/notify_slack.sh "text"
# Credentials are read from ~/env.sh at call time, either
#   SLACK_WEBHOOK_URL                      (incoming webhook), or
#   SLACK_BOT_TOKEN + SLACK_CHANNEL        (chat.postMessage; channel ID or member ID for a DM),
#   also accepted as OAUTH_SLACK_TOKEN + SLACK_MEMBER_ID.
# Does nothing (exit 0) when neither is set. Never prints the credentials.
set -uo pipefail
text="${1:-}"
[ -n "$text" ] || exit 0
# shellcheck disable=SC1090
[ -f ~/env.sh ] && source ~/env.sh
SLACK_BOT_TOKEN="${SLACK_BOT_TOKEN:-${OAUTH_SLACK_TOKEN:-}}"
SLACK_CHANNEL="${SLACK_CHANNEL:-${SLACK_MEMBER_ID:-}}"
payload=$(TEXT="$text" CHANNEL="${SLACK_CHANNEL:-}" python3 -c \
  'import json, os; d = {"text": os.environ["TEXT"]}; c = os.environ["CHANNEL"]; d.update({"channel": c} if c else {}); print(json.dumps(d))')
if [ -n "${SLACK_WEBHOOK_URL:-}" ]; then
  curl -sS -m 20 -o /dev/null -w "slack webhook http=%{http_code}\n" -X POST \
    -H 'Content-type: application/json' --data "$payload" "$SLACK_WEBHOOK_URL"
elif [ -n "${SLACK_BOT_TOKEN:-}" ] && [ -n "${SLACK_CHANNEL:-}" ]; then
  curl -sS -m 20 -X POST -H "Authorization: Bearer $SLACK_BOT_TOKEN" \
    -H 'Content-type: application/json; charset=utf-8' --data "$payload" \
    https://slack.com/api/chat.postMessage | python3 -c \
    'import json, sys; r = json.load(sys.stdin); print("slack api ok" if r.get("ok") else "slack api error: " + str(r.get("error")))'
fi
exit 0
