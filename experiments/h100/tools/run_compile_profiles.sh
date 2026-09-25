#!/usr/bin/env bash
# Compile-cost diagnosis for the widest workload: profile the Triton compile of
# zk_jellyfish_zerocheck_hp's heaviest eval kernel (point 7) at 32/64/128 bits under
# Triton 3.7.0 (the paper's) and 3.8.0, each from a fresh cache, one kernel per process.
# Optional overrides (defaults are the original run's): ZKDUEL_DIAG_DIR, ZKDUEL_LOG_DIR (output
# directories), ZKDUEL_TRITON38_PY (python of a Triton 3.8.0 venv), ZKDUEL_TRITON_VERSIONS
# (default "3.7.0 3.8.0").
cd "$(dirname "$0")/../../.."
source experiments/h100/env_table5.sh
OUT=${ZKDUEL_DIAG_DIR:-experiments/h100/diag}/compile_profile.jsonl
LOGS=${ZKDUEL_LOG_DIR:-experiments/h100/logs}
mkdir -p "$(dirname "$OUT")" "$LOGS"
declare -A PYS=([3.7.0]="$ZKDUEL_VENV/bin/python" [3.8.0]=${ZKDUEL_TRITON38_PY:-/home/ubuntu/venvs/zkduel-triton38/bin/python})
for ver in ${ZKDUEL_TRITON_VERSIONS:-3.7.0 3.8.0}; do
  for bw in 32 64 128; do
    ( cache=$(mktemp -d); TRITON_CACHE_DIR=$cache nice -n 10 ${PYS[$ver]} experiments/h100/tools/compile_profile.py \
        zk_jellyfish_zerocheck_hp $bw 7 >> $OUT 2> $LOGS/compile_profile_${ver}_bw$bw.log
      rm -rf "$cache" ) &
  done
done
wait
echo COMPILE_PROFILES_DONE
