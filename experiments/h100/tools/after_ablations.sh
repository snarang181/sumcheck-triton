#!/usr/bin/env bash
# Start the pinning-noise test once ablations.sh has finished (quiet GPU).
cd "$(dirname "$0")/../../.."
until grep -q " done$" experiments/h100/logs/ablations.log; do sleep 30; done
sleep 30
echo "$(date -u +%FT%TZ) starting pin_noise.sh"
experiments/h100/tools/pin_noise.sh
