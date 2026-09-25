#!/usr/bin/env bash
# Reproduce the experiments behind the zkDuel paper's reported numbers (HASP '26) on one GPU.
#
#   scripts/run_paper_experiments.sh [--dry-run] [--phases "LIST"] [--out DIR] [--skip-gate]
#                                    [--jobs N] [--force] [--include-jellyfish256]
#                                    [--cache-dir DIR] [--gate-baseline MS] [--gate-max-wait S]
#
# The primary platform is one NVIDIA H100 (the paper's A100 results are an existing dataset,
# experiments/a100/, and are not rerun). Set up the software stack first with
# scripts/setup_env.sh. A full run takes about half a day; run it under tmux or nohup and follow
# OUT/logs/run_paper_experiments.log. Nothing is pushed or posted anywhere.
#
# Phases (default: all, always in this order):
#   env         record the environment (Table 5 layout)                     -> OUT/env/
#   build       CUDA binary from the current source, and the occupancy-ablation variant -> OUT/bin/
#   validate    compile the sweep's Triton kernels; CPU-reference validation of both backends
#               (r=3 and r=10, fixed and SHA3 challenges)                    -> OUT/validation/
#   precompile  compile the remaining Triton kernels (tl.reduce and ablation variants), in
#               parallel, so no compile overlaps a timed phase               -> OUT/triton-cache/
#   sweep       diagnostic tier (Figure 4) and zkPHIRE tier                  -> OUT/sweep/
#   split       Nsight Systems wall-clock vs GPU-kernel split, all 25 constraints, 32/256 bits,
#               r=20/24; the tl.reduce variant; the r=28 overflow cases     -> OUT/diag/
#   occupancy   CUDA occupancy ablation (shared-memory padding binary)       -> OUT/diag/
#   ablations   layout x interpolation ablation, CUDA block-size sweep       -> OUT/diag/
#   counters    register anatomy (SASS) and Nsight Compute counters (ncu needs sudo) -> OUT/diag/
#   compile     Triton compile-time profiling (3.7.0; 3.8.0 if configured), Coalesce scaling
#   noise       run-to-run noise / vCPU pinning                              -> OUT/diag/
#   tables      findings tables T1-T18 and Figure 4 from OUT              -> OUT/tables/, OUT/figures/
# sweep, split, occupancy, ablations and noise are timing-sensitive: each first waits for the
# host-latency gate (a reference CUDA probe must come within 5% of a baseline, or a deadline
# passes). Keep the GPU otherwise idle while they run.
#
# Every phase logs to OUT/logs/PHASE.log and is resumable: a phase (and each long step inside it)
# writes a marker under OUT/.done/ when it completes, and a later run skips it unless --force.
# Outputs of an interrupted step are moved to OUT/.stale/ before the step is retried. The tables
# phase reruns whenever any result is newer than its marker.
#
# Options:
#   --dry-run               print the commands of the selected phases without running anything
#   --phases "LIST"         phases to run (space- or comma-separated); default: all
#   --out DIR               output directory (default: results/paper_run; must be outside experiments/)
#   --skip-gate             do not wait for the host-latency gate
#   --jobs N                parallel Triton compiles (default: number of CPUs minus 2)
#   --force                 rerun selected phases even if their markers exist
#   --include-jellyfish256  also compile and sweep zk_jellyfish_zerocheck_hp at 256 bits (its
#                           Triton eval kernels take roughly 2-7 h each to compile)
#   --cache-dir DIR         Triton cache root (default: OUT/triton-cache)
#   --gate-baseline MS      gate reference, CUDA median ms of zk_spartan_2 32-bit r=24 on a quiet
#                           host (default 2.90, the paper's H100 PCIe host)
#   --gate-max-wait S       give up waiting for the gate after S seconds and proceed (default 7200)
# Environment: ZKDUEL_VENV, CUDA_HOME, ZKDUEL_CUDA_COMPAT, ZKDUEL_OPENSSL_INCLUDE (see
# experiments/h100/env_table5.sh and scripts/setup_env.sh), ZKDUEL_TRITON38_PY (Triton 3.8.0
# python for the compile phase), ZKDUEL_OCC_SMEM_TOTALS (occupancy ablation shared-memory totals,
# default for H100's 228 KB/SM), ZKDUEL_WARM_TIMEOUT (per-kernel compile timeout, s),
# ZKDUEL_GATE_INTERVAL (seconds between gate probes, default 600).
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
SELF="$ROOT/scripts/run_paper_experiments.sh"
SELF_REL=scripts/run_paper_experiments.sh
H=experiments/h100
T=$H/tools

# Internal: one Triton kernel compile, run by xargs from the warm step (see warm below).
if [ "${1:-}" = "--warm-task" ]; then
  shift
  [ $# -eq 6 ] || { echo "usage: $SELF_REL --warm-task GROUP WORKLOAD BITS POINT POINT_MODE LAYOUT" >&2; exit 2; }
  cd "$ROOT" || exit 1
  [ -n "${ZKDUEL_SWEEP_CACHE:-}" ] && [ -n "${ZKDUEL_TREDUCE_CACHE:-}" ] \
    || { echo "--warm-task: ZKDUEL_SWEEP_CACHE and ZKDUEL_TREDUCE_CACHE must be set" >&2; exit 2; }
  [ -n "${ZKDUEL_VENV:-}" ] || source "$H/env_table5.sh"
  g=$1 w=$2 bw=$3 p=$4 pm=$5 lay=$6
  if [ "$g" = treduce ]; then cache=$ZKDUEL_TREDUCE_CACHE tool=$T/precompile_treduce.py
  else cache=$ZKDUEL_SWEEP_CACHE tool=$T/warm_cache.py; fi
  to=${ZKDUEL_WARM_TIMEOUT:-14400}
  [ "$w $bw" = "zk_jellyfish_zerocheck_hp 256" ] && to=${ZKDUEL_WARM_TIMEOUT_JF256:-43200}
  TRITON_CACHE_DIR="$cache" timeout "$to" "$ZKDUEL_VENV/bin/python" "$tool" "$w" "$bw" "$p" \
    --point-mode "$pm" --layout "$lay" || echo "FAILED $g $w $bw $p $pm $lay rc=$?"
  exit 0
fi

ALL_PHASES="env build validate precompile sweep split occupancy ablations counters compile noise tables"
DRY_RUN=0 SKIP_GATE=0 FORCE=0 INCLUDE_JF256=0
PHASES_ARG="" OUT_ARG="" CACHE_ARG=""
NCPU=$(nproc 2>/dev/null || echo 4)
JOBS=$(( NCPU > 2 ? NCPU - 2 : 1 ))
GATE_BASELINE=2.90 GATE_MAX_WAIT=7200 GATE_INTERVAL=${ZKDUEL_GATE_INTERVAL:-600}

usage() { sed -n '2,/^set -uo pipefail/{/^set -uo/d;s/^# \{0,1\}//;p}' "$SELF"; }
die() { echo "error: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --phases) [ $# -ge 2 ] || die "--phases needs a list"; PHASES_ARG=$2; shift ;;
    --phases=*) PHASES_ARG=${1#*=} ;;
    --out) [ $# -ge 2 ] || die "--out needs a directory"; OUT_ARG=$2; shift ;;
    --out=*) OUT_ARG=${1#*=} ;;
    --skip-gate) SKIP_GATE=1 ;;
    --jobs) [ $# -ge 2 ] || die "--jobs needs a number"; JOBS=$2; shift ;;
    --jobs=*) JOBS=${1#*=} ;;
    --force) FORCE=1 ;;
    --include-jellyfish256) INCLUDE_JF256=1 ;;
    --cache-dir) [ $# -ge 2 ] || die "--cache-dir needs a directory"; CACHE_ARG=$2; shift ;;
    --cache-dir=*) CACHE_ARG=${1#*=} ;;
    --gate-baseline) [ $# -ge 2 ] || die "--gate-baseline needs a value in ms"; GATE_BASELINE=$2; shift ;;
    --gate-max-wait) [ $# -ge 2 ] || die "--gate-max-wait needs seconds"; GATE_MAX_WAIT=$2; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (see --help)" ;;
  esac
  shift
done
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
[[ "$GATE_MAX_WAIT" =~ ^[0-9]+$ ]] || die "--gate-max-wait must be a number of seconds"
[[ "$GATE_BASELINE" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "--gate-baseline must be a number (ms)"

SELECTED=""
if [ -z "$PHASES_ARG" ] || [ "$PHASES_ARG" = all ]; then
  SELECTED=$ALL_PHASES
else
  want=" ${PHASES_ARG//,/ } "
  for p in $want; do [[ " $ALL_PHASES " == *" $p "* ]] || die "unknown phase '$p' (phases: $ALL_PHASES)"; done
  for p in $ALL_PHASES; do [[ "$want" == *" $p "* ]] && SELECTED+="${SELECTED:+ }$p"; done
fi

# Output paths: --out is relative to the caller's directory (the default to the repository root);
# paths inside the repository are shown relative to its root, where every command runs.
OUT_ABS=$(realpath -m -- "${OUT_ARG:-$ROOT/results/paper_run}")
cd "$ROOT" || exit 1
case "$OUT_ABS/" in
  "$ROOT/experiments/"*)
    die "--out $OUT_ARG is inside experiments/ (the committed data); choose another directory" ;;
esac
[ "$OUT_ABS" = "$ROOT" ] && die "--out must not be the repository root"
OUT=${OUT_ABS#"$ROOT/"}
CACHE_ROOT=$(realpath -m -- "${CACHE_ARG:-$OUT_ABS/triton-cache}")
SWEEP_CACHE=$CACHE_ROOT/sweep
TREDUCE_CACHE=$CACHE_ROOT/treduce
BIN=$OUT/bin/field_sumcheck_cuda
OCC_BIN=$OUT/bin/field_sumcheck_cuda_occ
DIAG=$OUT/diag
LOGS=$OUT/logs
DONE=$OUT/.done
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
GIT_REV=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)
GIT_DIRTY=$( [ -n "$(git status --porcelain --untracked-files=no 2>/dev/null)" ] && echo " (with uncommitted changes)" )

# Software stack (experiments/h100/env_table5.sh reads .zkduel_env.sh from scripts/setup_env.sh).
source "$H/env_table5.sh"
unset TRITON_CACHE_DIR   # every command below names its cache explicitly
PY="$ZKDUEL_VENV/bin/python"
HELPER_PY=python3; [ -x "$PY" ] && HELPER_PY=$PY
TRITON38_PY=${ZKDUEL_TRITON38_PY:-}
NCU=$(command -v ncu 2>/dev/null || echo /usr/local/bin/ncu)
OCC_SMEM_TOTALS=${ZKDUEL_OCC_SMEM_TOTALS:-0 21000 44000 56000 110000}
WARM_TIMEOUT=${ZKDUEL_WARM_TIMEOUT:-14400}
export ZKDUEL_SWEEP_CACHE=$SWEEP_CACHE ZKDUEL_TREDUCE_CACHE=$TREDUCE_CACHE ZKDUEL_WARM_TIMEOUT=$WARM_TIMEOUT

specs() {  # print a workload list from benchmarks/field_sweep/workload_specs.py (pure Python)
  PYTHONPATH=benchmarks/field_sweep PYTHONDONTWRITEBYTECODE=1 "$HELPER_PY" -c "import workload_specs as s; $1"
}
ZK_ALL=$(specs 'print(" ".join(s.ZK_WORKLOAD_NAMES))') || die "cannot read benchmarks/field_sweep/workload_specs.py"
POLY=$(specs 'print(" ".join(w.name for w in s.POLY_WORKLOADS))')
ZK256=$ZK_ALL
[ $INCLUDE_JF256 = 1 ] || ZK256=$(specs 'print(" ".join(w for w in s.ZK_WORKLOAD_NAMES if w != "zk_jellyfish_zerocheck_hp"))')
R28_W="zk_vanilla_zerocheck_hp zk_vanilla_permcheck_hp zk_jellyfish_zerocheck_hp zk_jellyfish_permcheck_hp zk_opencheck"
OCC_W="poly_abc zk_spartan_2 zk_witness_id_point_1 zk_complete_add_3 zk_vanilla_permcheck_hp"
OCC_IDS=$(specs "print(' '.join(str(s.WORKLOAD_BY_NAME[w].ident) for w in '$OCC_W'.split()))")
ABL_W="zk_spartan_2 zk_witness_id_point_1 zk_complete_add_3 zk_vanilla_permcheck_hp zk_opencheck"

# ------------------------------------------------------------------------------------ helpers
PLOG=/dev/null PHASE=""
q() { local a s=""; for a in "$@"; do s+="$(printf '%q' "$a") "; done; printf '%s' "${s% }"; }
say() {
  if [ $DRY_RUN = 1 ]; then echo "$*"; return; fi
  local m; m="[$(date -u +%FT%TZ)] $*"; echo "$m"; echo "$m" >> "$PLOG"
}
show() { echo "  + $*"; [ $DRY_RUN = 1 ] || echo "+ $*" >> "$PLOG"; }
run() {  # CMD...: print, then run with stdout and stderr appended to the phase log
  show "$(q "$@")"
  [ $DRY_RUN = 1 ] && return 0
  "$@" >> "$PLOG" 2>&1
}
run_to() {  # FILE CMD...: stdout to FILE, stderr to the phase log
  local f=$1; shift
  show "$(q "$@") > $(q "$f")"
  [ $DRY_RUN = 1 ] && return 0
  "$@" > "$f" 2>> "$PLOG"
}
BG_PIDS=()
run_bg() {  # LOGFILE CMD...: run in the background with its own log; collect with wait_bg
  local f=$1; shift
  show "$(q "$@") > $(q "$f") 2>&1 &"
  [ $DRY_RUN = 1 ] && return 0
  "$@" > "$f" 2>&1 &
  BG_PIDS+=($!)
}
wait_bg() {
  local rc=0 p
  for p in ${BG_PIDS[@]+"${BG_PIDS[@]}"}; do wait "$p" || rc=1; done
  BG_PIDS=()
  [ $DRY_RUN = 1 ] && show "wait"
  return $rc
}
marker() { echo "$(date -u +%FT%TZ) git=$GIT_REV jf256=$INCLUDE_JF256"; }
is_done() { [ -f "$DONE/$1" ]; }
end_step() { [ $DRY_RUN = 1 ] && return 0; mkdir -p "$DONE"; marker > "$DONE/$1"; }
stale_aside() {  # NAME PATH...: move outputs left by an unfinished attempt out of the way
  local name=$1 f dest; shift
  local stale=()
  for f in "$@"; do [ -e "$f" ] && stale+=("$f"); done
  [ ${#stale[@]} -eq 0 ] && return 0
  dest=$OUT/.stale/$name.$RUN_ID
  say "  $name: moving outputs of an unfinished earlier attempt to $dest"
  run mkdir -p "$dest"
  for f in "${stale[@]}"; do run mv "$f" "$dest/"; done
}
begin_step() {  # NAME OUTPUT...: returns 1 if the step is already done, else clears stale outputs
  local name=$1; shift
  if is_done "$name"; then say "  $name: done earlier ($(cut -d' ' -f1 "$DONE/$name")); skipping"; return 1; fi
  stale_aside "$name" "$@"
  say "  $name"
  return 0
}
pending() { local s; for s in "$@"; do is_done "$s" || return 0; done; return 1; }

# Triton kernel warm-up: every eval kernel a later phase launches is compiled first, one kernel
# per process and --jobs at a time (Triton compiles one kernel at a time per process), heaviest
# first, so no timed run pays for a compile and no sweep case hits its timeout while compiling.
#   sweep    every eval kernel of the sweep and of the validation (element layout, specialized)
#   treduce  the tl.reduce variant's kernels for the split (25 constraints, 32/256 bits)
#   ablation the layout x interpolation variants other than the default (5 constraints, 32/256)
gen_tasks() {  # GROUP... -> "GROUP WORKLOAD BITS POINT POINT_MODE LAYOUT" per line
  PYTHONPATH=benchmarks/field_sweep PYTHONDONTWRITEBYTECODE=1 "$HELPER_PY" - "$INCLUDE_JF256" "$ABL_W" "$@" <<'PY'
import sys
from workload_specs import WORKLOAD_BY_NAME as S, WORKLOAD_NAMES, ZK_WORKLOAD_NAMES
jf256, abl, groups = sys.argv[1] == "1", sys.argv[2].split(), sys.argv[3:]
tasks = []
for g in groups:
    if g == "sweep":
        cfg = [(w, bw, "specialized", "element") for w in WORKLOAD_NAMES for bw in (32, 64, 128, 256)]
    elif g == "treduce":
        cfg = [(w, bw, "specialized", "element") for w in ZK_WORKLOAD_NAMES for bw in (32, 256)]
    elif g == "ablation":
        cfg = [(w, bw, pm, lay) for lay, pm in (("element", "generic"), ("limb", "specialized"),
               ("limb", "generic")) for bw in (256, 32) for w in abl]
    else:
        raise SystemExit(f"unknown kernel group {g}")
    for w, bw, pm, lay in cfg:
        # jellyfish zerocheck at 256 bits: hours per kernel (T12); only the sweep can include it
        if w == "zk_jellyfish_zerocheck_hp" and bw == 256 and not (jf256 and g == "sweep"):
            continue
        tasks += [(g, w, bw, p, pm, lay) for p in range(S[w].degree + 1)]
tasks.sort(key=lambda t: (-t[2], -S[t[1]].terms * sum(S[t[1]].term_lens)))
print("\n".join(" ".join(map(str, t)) for t in tasks))
PY
}
declare -A WARMED=() WARM_FAILED=()
warm_done() {  # GROUP: compiled earlier in this run, or by a completed validate/precompile
  local g=$1 f
  [ -n "${WARMED[$g]:-}" ] && return 0
  for f in "$DONE/precompile" $([ "$g" = sweep ] && echo "$DONE/validate.warm"); do
    [ -f "$f" ] || continue
    [ "$g" = sweep ] && [ $INCLUDE_JF256 = 1 ] && ! grep -q "jf256=1" "$f" && continue
    return 0
  done
  return 1
}
warm() {  # GROUP...: compile these groups' kernels; returns 1 only if the step could not run
  local tag list n tasks log failed=0 g
  tag=$(IFS=_; echo "$*")
  tasks=$OUT/precompile/tasks_$tag.txt log=$OUT/precompile/warm_$tag.log
  list=$(gen_tasks "$@") || { say "  could not generate the kernel list"; return 1; }
  n=$(printf '%s\n' "$list" | grep -c .)
  say "  compile $n Triton eval kernels ($*), $JOBS at a time, heaviest first; per-kernel timeout ${WARM_TIMEOUT}s"
  if [ $DRY_RUN = 1 ]; then
    show "(write the $n tasks, one 'GROUP WORKLOAD BITS POINT POINT_MODE LAYOUT' per line, to $tasks)"
  else
    mkdir -p "$OUT/precompile"; printf '%s\n' "$list" > "$tasks"
  fi
  show "env ZKDUEL_SWEEP_CACHE=$(q "$SWEEP_CACHE") ZKDUEL_TREDUCE_CACHE=$(q "$TREDUCE_CACHE") xargs -P $JOBS -L 1 $SELF_REL --warm-task < $(q "$tasks") > $(q "$log") 2>&1"
  if [ $DRY_RUN = 0 ]; then
    mkdir -p "$SWEEP_CACHE" "$TREDUCE_CACHE"
    xargs -P "$JOBS" -L 1 "$SELF" --warm-task < "$tasks" > "$log" 2>&1
    failed=$(grep -c '^FAILED' "$log")
    say "  compiled $(grep -c ' point [0-9]*: ' "$log") kernels, $failed failed (log: $log)"
    [ "$failed" -eq 0 ] || grep '^FAILED' "$log" | head -20 | sed 's/^/    /'
  fi
  for g in "$@"; do WARMED[$g]=1; WARM_FAILED[$g]=$failed; done
  return 0
}
ensure_warm() {  # GROUP...: warm the groups no earlier phase has compiled
  local need=() g
  for g in "$@"; do warm_done "$g" || need+=("$g"); done
  [ ${#need[@]} -eq 0 ] && return 0
  warm "${need[@]}"
}

# Host-latency gate (as in experiments/h100/run_fixed_sweep.sh): this host's kernel
# launch / sync latency drifts over the day while GPU kernel time does not. Probe one reference
# configuration with the CUDA binary until its median is within 5% of the baseline, or until
# the deadline, and record which.
gate() {
  if [ $SKIP_GATE = 1 ]; then say "  host-latency gate skipped (--skip-gate)"; return 0; fi
  local csv=$OUT/gate/probe.csv log=$OUT/gate/probe.log ms deadline first=1
  if [ $DRY_RUN = 0 ] && [ ! -x "$BIN" ]; then say "  gate: $BIN is missing; run the build phase first"; return 1; fi
  say "  host-latency gate: CUDA probe until within 5% of $GATE_BASELINE ms (every ${GATE_INTERVAL}s, at most ${GATE_MAX_WAIT}s)"
  deadline=$(( $(date +%s) + GATE_MAX_WAIT ))
  [ $DRY_RUN = 1 ] || mkdir -p "$OUT/gate"
  if [ $DRY_RUN = 0 ]; then
    local apps; apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
    [ -z "$apps" ] || say "  WARNING: other processes are using the GPU: $(echo "$apps" | tr '\n' ';')"
  fi
  while :; do
    run "$BIN" --workloads zk_spartan_2 --bit-widths 32 --rounds 24 --warmups 2 --repeats 10 \
      --point-mode specialized --layout element --timer events --out "$csv" \
      || { say "  gate: the CUDA probe failed (see $PLOG)"; return 1; }
    if [ $DRY_RUN = 1 ]; then
      show "(compare median_ms in $csv with 1.05 x $GATE_BASELINE ms; if above, probe again every ${GATE_INTERVAL}s until within or ${GATE_MAX_WAIT}s have passed)"
      return 0
    fi
    ms=$("$HELPER_PY" -c 'import csv, sys; print(next(csv.DictReader(open(sys.argv[1])))["median_ms"])' "$csv" 2>/dev/null || echo nan)
    echo "$(date -u +%FT%TZ) phase=$PHASE probe cuda_ms=$ms baseline_ms=$GATE_BASELINE" >> "$log"
    if [ $first = 1 ] && "$HELPER_PY" -c 'import sys; m, b = map(float, sys.argv[1:]); sys.exit(0 if 0.67 < m / b < 1.5 else 1)' "$ms" "$GATE_BASELINE" 2>/dev/null; then :
    elif [ $first = 1 ]; then
      say "  NOTE: probe $ms ms is far from the $GATE_BASELINE ms baseline (the paper's H100 PCIe host); on another GPU or host pass --gate-baseline with this host's quiet value, or --skip-gate"
    fi
    first=0
    if "$HELPER_PY" -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) * 1.05 else 1)' "$ms" "$GATE_BASELINE" 2>/dev/null; then
      say "  gate passed: CUDA probe $ms ms (baseline $GATE_BASELINE ms)"; return 0
    fi
    if [ "$(date +%s)" -ge "$deadline" ]; then
      say "  gate: deadline reached with CUDA probe $ms ms (baseline $GATE_BASELINE ms); proceeding"; return 0
    fi
    say "  gate: CUDA probe $ms ms is above 1.05 x $GATE_BASELINE ms; next probe in ${GATE_INTERVAL}s"
    sleep "$GATE_INTERVAL"
  done
}

need_bin() {  # FILE...: binaries a phase needs
  local b
  [ $DRY_RUN = 1 ] && return 0
  for b in "$@"; do [ -x "$b" ] || { say "  $b is missing; run the build phase first"; return 1; }; done
}

# Sweep settings (as in the paper's H100 sweep): warmups 2, repeats 10, specialized points,
# element layout, whole-run timing, fixed challenges, 1800 s per-case timeout.
SWEEP_ROUNDS="14 16 18 20 22 24 26 28"
diag() {  # diag_overhead.py ARGS...: wall-clock vs nsys GPU-kernel split, raw runs under OUT/diag/raw
  run "$PY" "$T/diag_overhead.py" --diag-dir "$DIAG" "$@"
}

# ------------------------------------------------------------------------------------- phases
phase_env() {
  run mkdir -p "$OUT/env"
  run "$PY" "$T/capture_env.py" --out-dir "$OUT/env" --label run || return 1
  show "(write $OUT/env/run_config.txt: git revision, options, paths)"
  [ $DRY_RUN = 1 ] && return 0
  {
    echo "date=$(date -u +%FT%TZ)"
    echo "git=$(git rev-parse HEAD 2>/dev/null)$GIT_DIRTY"
    echo "phases=$SELECTED"
    echo "jobs=$JOBS skip_gate=$SKIP_GATE gate_baseline_ms=$GATE_BASELINE gate_max_wait_s=$GATE_MAX_WAIT include_jellyfish256=$INCLUDE_JF256"
    echo "ZKDUEL_VENV=$ZKDUEL_VENV"
    echo "CUDA_HOME=$CUDA_HOME"
    echo "ZKDUEL_CUDA_COMPAT=$ZKDUEL_CUDA_COMPAT"
    echo "ZKDUEL_OPENSSL_INCLUDE=$ZKDUEL_OPENSSL_INCLUDE"
    echo "ZKDUEL_TRITON38_PY=$TRITON38_PY"
    echo "triton_caches=$SWEEP_CACHE $TREDUCE_CACHE"
    echo "nvcc=$(command -v nvcc) nsys=$(command -v nsys) ncu=$NCU"
  } > "$OUT/env/run_config.txt"
  git status --porcelain > "$OUT/env/git_status.txt" 2>/dev/null
}

phase_build() {
  command -v nvcc > /dev/null || [ $DRY_RUN = 1 ] || { say "  nvcc not found; run scripts/setup_env.sh"; return 1; }
  if begin_step build.main "$BIN"; then
    local t0; t0=$(date +%s)
    say "  CUDA binary from the current source (about 1 h: every workload's kernels are instantiated)"
    run "$H/build_field_sumcheck.sh" "$BIN" || return 1
    show "(record build seconds, sha256 and source revision in $BIN.build.txt)"
    [ $DRY_RUN = 1 ] || echo "BUILD_OK seconds=$(( $(date +%s) - t0 )) sha256=$(sha256sum "$BIN" | cut -c1-16) source=$GIT_REV$GIT_DIRTY" \
      | tee -a "$PLOG" > "$BIN.build.txt"
    end_step build.main
  fi
  if begin_step build.occ "$OCC_BIN"; then
    say "  occupancy-ablation variant: patches/cuda_occupancy_knob.patch, workloads $OCC_W only"
    run env BUILD_DIR="$OUT/bin" "$H/build_ablation_binary.sh" field_sumcheck_cuda_occ "$OCC_IDS" \
      "$H/patches/cuda_occupancy_knob.patch" || return 1
    end_step build.occ
  fi
}

phase_validate() {
  local v=$OUT/validation bw
  if begin_step validate.warm; then
    ensure_warm sweep || return 1
    [ "${WARM_FAILED[sweep]:-0}" -eq 0 ] && end_step validate.warm
  fi
  need_bin "$BIN" || return 1
  local rc=0
  if begin_step validate.triton "$v"/triton_bw*.jsonl "$v"/triton_bw*.log; then
    run mkdir -p "$v"
    for bw in 32 64 128 256; do
      local ex=()
      [ $bw = 256 ] && [ $INCLUDE_JF256 = 0 ] && ex=(--exclude zk_jellyfish_zerocheck_hp)
      run_bg "$v/triton_bw$bw.log" env TRITON_CACHE_DIR="$SWEEP_CACHE" timeout 14400 "$PY" -u \
        "$T/validate_vs_registry.py" --bit-width $bw --layout element --point-mode specialized \
        ${ex[@]+"${ex[@]}"} --out "$v/triton_bw$bw.jsonl"
    done
    local bg_ok=1
    wait_bg || { bg_ok=0; say "  WARNING: a Triton validation process exited non-zero (see $v/triton_bw*.log)"; }
    show "(check: every row of $v/triton_bw*.jsonl matches the CPU reference, all four widths present)"
    if [ $DRY_RUN = 0 ]; then
      if "$HELPER_PY" - "$v"/triton_bw*.jsonl <<'PY' 2>&1 | tee -a "$PLOG"; then [ $bg_ok = 1 ] && end_step validate.triton || rc=1; else rc=1; fi
import json, sys
rows = [json.loads(line) for f in sys.argv[1:] for line in open(f)]
bad = [r for r in rows if r.get("match") is not True]
widths = sorted({r["bit_width"] for r in rows})
print(f"  Triton vs CPU reference: {len(rows) - len(bad)}/{len(rows)} rows match, "
      f"{len({r['workload'] for r in rows})} workloads, widths {widths}")
for r in bad[:20]:
    print(f"    NOT MATCHING: {r['workload']} {r['bit_width']}-bit r={r['rounds']} {r['challenge_mode']}: "
          f"{(r.get('error') or 'checksum mismatch')[:120]}")
sys.exit(1 if bad or widths != [32, 64, 128, 256] else 0)
PY
    fi
  fi
  if begin_step validate.cuda "$v/cuda_vs_registry.jsonl"; then
    if run "$PY" "$T/validate_cuda_vs_registry.py" --binary "$BIN" --validation-dir "$v" \
      --point-mode specialized --layout element --out "$v/cuda_vs_registry.jsonl"; then
      end_step validate.cuda
    else
      rc=1
    fi
    [ $DRY_RUN = 1 ] || grep -E "^compared" "$PLOG" | tail -n 1 | sed 's/^/  CUDA vs CPU reference: /'
  fi
  [ $rc -eq 0 ] || say "  WARNING: validation found rows that do not match the CPU reference (see $PLOG)"
  return $rc
}

phase_precompile() {
  ensure_warm sweep treduce ablation || return 1
  local g
  for g in sweep treduce ablation; do [ "${WARM_FAILED[$g]:-0}" -eq 0 ] || return 2; done
}

sweep_part() {  # NAME ARGS...
  local name=$1 s=$OUT/sweep; shift
  begin_step "sweep.$name" "$s/$name.csv" "$s/cases_$name" "$s/$name.command.txt" || return 0
  local cmd=("$PY" -u benchmarks/field_sweep/resilient_compare_triton_cuda.py "$@"
    --rounds $SWEEP_ROUNDS --warmups 2 --repeats 10
    --point-mode specialized --layout element --timing-mode whole-run --challenge-mode fixed
    --per-case-timeout 1800 --skip-build --cuda-binary "$BIN" --python "$PY"
    --tmp-dir "$s/cases_$name" --out "$s/$name.csv")
  [ $DRY_RUN = 1 ] || { q "${cmd[@]}" > "$s/$name.command.txt"; echo " TRITON_CACHE_DIR=$SWEEP_CACHE" >> "$s/$name.command.txt"; }
  run env TRITON_CACHE_DIR="$SWEEP_CACHE" "${cmd[@]}" || return 1
  [ $DRY_RUN = 1 ] || say "  sweep.$name: $(( $(wc -l < "$s/$name.csv") - 1 )) rows"
  end_step "sweep.$name"
}
join_zk() {  # the zkPHIRE tier as one CSV (the layout of the paper's 800-configuration sweep)
  local s=$OUT/sweep
  show "awk 'FNR > 1 || NR == 1' $s/zk_32_128.csv $s/zk_256.csv > $s/zkphire_tier.csv"
  [ $DRY_RUN = 1 ] && return 0
  [ -f "$s/zk_32_128.csv" ] && [ -f "$s/zk_256.csv" ] || return 1
  awk 'FNR > 1 || NR == 1' "$s/zk_32_128.csv" "$s/zk_256.csv" > "$s/zkphire_tier.csv"
}
phase_sweep() {
  need_bin "$BIN" || return 1
  [ $INCLUDE_JF256 = 1 ] || say "  NOTE: zk_jellyfish_zerocheck_hp at 256 bits is not run: its Triton eval kernels take hours each to compile (T12), so its cases would only hit the 1800 s timeout. Pass --include-jellyfish256 to compile and run it."
  ensure_warm sweep || return 1
  if pending sweep.diag_tier sweep.zk_32_128 sweep.zk_256; then gate || return 1; fi
  run mkdir -p "$OUT/sweep"
  [ $DRY_RUN = 1 ] || [ -f "$OUT/sweep/started_at.txt" ] || date -u +%FT%TZ > "$OUT/sweep/started_at.txt"
  say "  diagnostic tier: 7 polynomials x 4 widths x r=14..28 (Figure 4)"
  sweep_part diag_tier --workloads $POLY --bit-widths 32 64 128 256 || return 1
  say "  zkPHIRE tier: 25 constraints at 32/64/128 bits, $(echo $ZK256 | wc -w) at 256 bits, r=14..28"
  sweep_part zk_32_128 --zk-only --bit-widths 32 64 128 || return 1
  sweep_part zk_256 --workloads $ZK256 --bit-widths 256 || return 1
  join_zk || return 1
  [ $DRY_RUN = 1 ] || date -u +%FT%TZ > "$OUT/sweep/finished_at.txt"
}

phase_split() {
  need_bin "$BIN" || return 1
  ensure_warm sweep treduce || return 1
  if pending split.baseline split.treduce split.r28; then gate || return 1; fi
  if begin_step split.baseline "$DIAG/baseline_all.jsonl" "$DIAG/raw/baseline_all"; then
    say "  all 25 constraints, 32/256 bits, r=20/24 (T4, T13)"
    diag --label baseline_all --cuda-binary "$BIN" --triton-cache-dir "$SWEEP_CACHE" --workloads $ZK_ALL \
      --bit-widths 32 256 --rounds 20 24 --out "$DIAG/baseline_all.jsonl" || return 1
    end_step split.baseline
  fi
  if begin_step split.treduce "$DIAG/treduce_all.jsonl" "$DIAG/raw/treduce_all"; then
    say "  tl.reduce variant of Triton's block reduction, same configurations (T5)"
    diag --label treduce_all --triton-script "$T/triton_sweep_treduce.py" --cuda-binary "$BIN" \
      --triton-cache-dir "$TREDUCE_CACHE" --workloads $ZK_ALL --bit-widths 32 256 --rounds 20 24 \
      --out "$DIAG/treduce_all.jsonl" || return 1
    end_step split.treduce
  fi
  if begin_step split.r28 "$DIAG/r28_int64_fix.jsonl" "$DIAG/raw/r28_int64_fix"; then
    say "  r=28 configurations with more than 2^31 table elements (int64-index fix; T7)"
    diag --label r28_int64_fix --cuda-binary "$BIN" --triton-cache-dir "$SWEEP_CACHE" --workloads $R28_W \
      --bit-widths 32 64 --rounds 28 --out "$DIAG/r28_int64_fix.jsonl" || return 1
    end_step split.r28
  fi
}

phase_occupancy() {
  need_bin "$OCC_BIN" || return 1
  gate || return 1
  stale_aside occupancy "$DIAG/occupancy.jsonl" "$DIAG"/raw/occ_smem*
  say "  CUDA eval kernels with padded shared memory: totals $OCC_SMEM_TOTALS bytes per block (T6)"
  local bw total base extra
  for bw in 32 256; do
    base=$((128 * bw / 32 * 4))
    for total in $OCC_SMEM_TOTALS; do
      extra=0; [ "$total" -gt 0 ] && extra=$((total - base))
      diag --label occ_smem$total --skip-triton --cuda-binary "$OCC_BIN" --cuda-env ZKDUEL_EVAL_EXTRA_SMEM=$extra \
        --workloads $OCC_W --bit-widths $bw --rounds 20 24 --out "$DIAG/occupancy.jsonl" || return 1
    done
  done
}

phase_ablations() {
  need_bin "$BIN" || return 1
  ensure_warm sweep ablation || return 1
  if pending ablations.layout ablations.block; then gate || return 1; fi
  local layout pm b
  if begin_step ablations.layout "$DIAG/layout_points.jsonl" "$DIAG"/raw/abl_*; then
    say "  layout (element/limb) x interpolation (specialized/generic), both backends (T15, T18)"
    for layout in element limb; do
      for pm in specialized generic; do
        diag --label abl_${layout}_$pm --layout $layout --point-mode $pm --cuda-binary "$BIN" \
          --triton-cache-dir "$SWEEP_CACHE" --workloads $ABL_W --bit-widths 32 256 --rounds 20 24 \
          --out "$DIAG/layout_points.jsonl" || return 1
      done
    done
    end_step ablations.layout
  fi
  if begin_step ablations.block "$DIAG/cuda_block.jsonl" "$DIAG"/raw/cudablock_*; then
    say "  CUDA thread-block size 64/128/256/512 (T16)"
    for b in 64 128 256 512; do
      diag --label cudablock_$b --skip-triton --cuda-binary "$BIN" --cuda-args "--block $b" \
        --workloads $ABL_W --bit-widths 32 256 --rounds 20 24 --out "$DIAG/cuda_block.jsonl" || return 1
    done
    end_step ablations.block
  fi
}

phase_counters() {
  need_bin "$BIN" || return 1
  ensure_warm sweep treduce || return 1
  if begin_step counters.registers "$DIAG/register_anatomy.jsonl"; then
    say "  register anatomy of matched eval kernels, static SASS (T10)"
    run mkdir -p "$DIAG"
    run env TRITON_CACHE_DIR="$SWEEP_CACHE" "$PY" "$T/register_anatomy.py" --variant base --out "$DIAG/register_anatomy.jsonl" || return 1
    run env TRITON_CACHE_DIR="$TREDUCE_CACHE" "$PY" "$T/register_anatomy.py" --variant treduce --out "$DIAG/register_anatomy.jsonl" || return 1
    run "$PY" "$T/register_anatomy.py" --variant cuda --cuda-binary "$BIN" --out "$DIAG/register_anatomy.jsonl" || return 1
    end_step counters.registers
  fi
  pending counters.ncu || return 0
  if [ $DRY_RUN = 1 ]; then
    show "(check: sudo -n true; if it fails, skip Nsight Compute with a warning)"
  elif ! sudo -n true 2> /dev/null; then
    say "  WARNING: 'sudo -n true' failed, so the Nsight Compute counters (T11) are skipped: ncu needs root where the driver sets RmProfilingAdminOnly=1. Run 'sudo -v' (or allow passwordless sudo) and rerun with --phases counters."
    return 2
  fi
  begin_step counters.ncu "$DIAG/ncu" || return 0
  say "  Nsight Compute counters of the eval kernels, Triton / Triton + tl.reduce / CUDA (T11; sudo)"
  run env ZKDUEL_DIAG_DIR="$DIAG" ZKDUEL_SWEEP_CACHE="$SWEEP_CACHE" ZKDUEL_TREDUCE_CACHE="$TREDUCE_CACHE" \
    ZKDUEL_CUDA_BIN="$BIN" NCU="$NCU" NCU_TMP="/tmp/zkduel-ncu-$RUN_ID" "$T/ncu_anatomy.sh" || return 1
  if [ $DRY_RUN = 0 ]; then
    local n; n=$(grep -l '"Metric Name"' "$DIAG"/ncu/*.csv 2> /dev/null | wc -l)
    say "  ncu: $n of 18 profiles have counters"
    [ "$n" -eq 18 ] || { say "  WARNING: some ncu profiles are empty (see $DIAG/ncu/*.log)"; return 1; }
  fi
  end_step counters.ncu
}

phase_compile() {
  local vers=3.7.0
  if [ -n "$TRITON38_PY" ] && [ -x "$TRITON38_PY" ]; then vers="3.7.0 3.8.0"
  else say "  NOTE: no Triton 3.8.0 environment (ZKDUEL_TRITON38_PY; scripts/setup_env.sh --with-triton38), so only Triton 3.7.0 is profiled"; fi
  local e=(env ZKDUEL_DIAG_DIR="$DIAG" ZKDUEL_LOG_DIR="$LOGS" ZKDUEL_TRITON_VERSIONS="$vers")
  [ "$vers" = 3.7.0 ] || e+=(ZKDUEL_TRITON38_PY="$TRITON38_PY")
  if begin_step compile.profiles "$DIAG/compile_profile.jsonl" "$LOGS"/compile_profile_*.log; then
    say "  stage times of zk_jellyfish_zerocheck_hp's point-7 eval kernel, 32/64/128 bits, Triton $vers (T12)"
    run "${e[@]}" "$T/run_compile_profiles.sh" || return 1
    [ $DRY_RUN = 1 ] || [ -s "$DIAG/compile_profile.jsonl" ] || { say "  no compile profiles (see $LOGS/compile_profile_*.log)"; return 1; }
    end_step compile.profiles
  fi
  if begin_step compile.passes "$DIAG/compile_passes.jsonl" "$LOGS"/mlir_passes_*.txt; then
    say "  MLIR per-pass times of the same compiles (T12)"
    run "${e[@]}" "$T/run_compile_passes.sh" || return 1
    [ $DRY_RUN = 1 ] || [ -s "$DIAG/compile_passes.jsonl" ] || { say "  no pass timings (see $LOGS/mlir_passes_*.txt)"; return 1; }
    end_step compile.passes
  fi
  if begin_step compile.coalesce "$DIAG/coalesce_scaling.jsonl"; then
    say "  TritonGPUCoalesce scaling on a synthetic kernel (T17)"
    run "${e[@]}" "$T/run_coalesce_scaling.sh" || return 1
    [ $DRY_RUN = 1 ] || [ -s "$DIAG/coalesce_scaling.jsonl" ] || { say "  no Coalesce scaling records (see $PLOG)"; return 1; }
    end_step compile.coalesce
  fi
}

phase_noise() {
  need_bin "$BIN" || return 1
  ensure_warm sweep || return 1
  gate || return 1
  stale_aside noise "$DIAG/pin_noise.jsonl"
  say "  zk_spartan_2 32-bit r=24 pinned to each vCPU, then 10 unpinned runs, both backends (T18)"
  run env ZKDUEL_DIAG_DIR="$DIAG" ZKDUEL_SWEEP_CACHE="$SWEEP_CACHE" ZKDUEL_CUDA_BIN="$BIN" "$T/pin_noise.sh" || return 1
  [ $DRY_RUN = 1 ] || [ -s "$DIAG/pin_noise.jsonl" ] || { say "  no pin_noise records (see $PLOG)"; return 1; }
}

phase_tables() {
  local s=$OUT/sweep zk=$OUT/sweep/zkphire_tier.csv
  run mkdir -p "$OUT/tables" "$OUT/figures"
  if [ $DRY_RUN = 1 ] || { [ -f "$s/zk_32_128.csv" ] && [ -f "$s/zk_256.csv" ]; }; then
    join_zk || return 1
    say "  sweep summaries (the paper's Tables 2 and 4) and skip classes"
    run "$PY" "$T/summarize_sweep.py" "$zk" --label "zkPHIRE tier (this run)" --md "$OUT/tables/sweep_summary.md" || return 1
    run "$PY" "$T/classify_skips.py" "$zk" --md "$OUT/tables/sweep_skips.md" || return 1
  else
    say "  WARNING: no zkPHIRE-tier sweep CSVs in $s (run the sweep phase); the H100 columns of T1-T3, T7 and T14 will be empty"
  fi
  say "  findings tables T1-T18 (A100 columns from the recorded dataset experiments/a100/)"
  run_to "$OUT/tables/FINDINGS_TABLES.md" "$PY" "$T/make_findings_tables.py" --h100-dir "$OUT" --sweep-csv "$zk" || return 1
  if [ $DRY_RUN = 1 ] || [ -f "$s/diag_tier.csv" ]; then
    say "  Figure 4 (CUDA speedup by workload) from the diagnostic-tier CSV"
    run "$PY" scripts/plot_zkduel.py "$s/diag_tier.csv" --plot-dir "$OUT/figures" --table-dir "$OUT/figures/tables" || return 1
    run cp "$OUT/figures/04_cuda_speedup_by_workload.pdf" "$OUT/figures/figure4_cuda_speedup_by_workload.pdf" || return 1
  else
    say "  WARNING: no $s/diag_tier.csv (run the sweep phase); Figure 4 skipped"
  fi
}

# ------------------------------------------------------------------------------ estimates
# Wall-clock estimates from the paper's H100 PCIe host (26 vCPUs; logs under experiments/h100/logs).
declare -A DESC=(
  [env]="record the environment (Table 5)"
  [build]="CUDA binary from the current source + occupancy-ablation variant"
  [validate]="compile the sweep's Triton kernels; CPU-reference validation of both backends"
  [precompile]="compile the tl.reduce and ablation Triton variants"
  [sweep]="diagnostic tier (Figure 4) + zkPHIRE tier"
  [split]="nsys wall-clock vs GPU-kernel split, 25 constraints; tl.reduce; r=28"
  [occupancy]="CUDA occupancy ablation"
  [ablations]="layout x interpolation; CUDA block size"
  [counters]="register anatomy; Nsight Compute counters (sudo)"
  [compile]="Triton compile-time profiling; Coalesce scaling"
  [noise]="run-to-run noise / vCPU pinning"
  [tables]="findings tables T1-T18, Figure 4"
)
scale() { local m=$(( $1 * 24 / JOBS )); [ $m -lt $2 ] && m=$2; echo $m; }  # compile work at --jobs
declare -A EST=(
  [env]=1 [build]=65 [validate]=$(scale 105 45) [precompile]=$(scale 115 70) [sweep]=100 [split]=75
  [occupancy]=15 [ablations]=40 [counters]=20 [compile]=$([ -x "${TRITON38_PY:-/nonexistent}" ] && echo 125 || echo 95)
  [noise]=15 [tables]=1
)
declare -A EST_NOTE=(
  [build]="nvcc of all workloads ~56 min, variant ~8 min"
  [validate]="cold Triton cache; minutes when warm"
  [precompile]="cold Triton cache; minutes when warm"
  [sweep]="diag tier ~20 min, 32-128 bits ~57 min, 256 bits ~21 min; plus gate wait"
  [split]="~33 min each for baseline and tl.reduce, r=28 ~4 min; plus gate wait"
  [compile]="profiles ~35 min, passes ~27 min, Coalesce scaling ~35 min per Triton version"
)
fmt() { local m=$1 t; if [ "$m" -lt 60 ]; then echo "~${m} min"; else t=$(( (m * 10 + 30) / 60 )); echo "~$((t / 10)).$((t % 10)) h"; fi; }
[ $INCLUDE_JF256 = 1 ] && { EST[validate]=$(( ${EST[validate]} + 420 )); EST_NOTE[validate]="+ jellyfish 256-bit kernels, up to ~7 h each"; }

# ---------------------------------------------------------------------------------- main
if [ $DRY_RUN = 0 ]; then
  [ -x "$PY" ] || die "no Python environment at $PY; run scripts/setup_env.sh (or set ZKDUEL_VENV)"
  mkdir -p "$LOGS" "$DONE" || die "cannot create $OUT"
  if command -v flock > /dev/null; then
    exec 9> "$OUT/.lock"
    flock -n 9 || die "another run_paper_experiments.sh is using $OUT"
  fi
  exec > >(tee -a "$LOGS/run_paper_experiments.log") 2>&1
fi

echo "zkDuel paper experiments$([ $DRY_RUN = 1 ] && echo " (dry run: commands are printed, nothing is executed)")"
echo "  repository  $ROOT (git $GIT_REV$GIT_DIRTY)"
echo "  output      $OUT"
echo "  caches      $SWEEP_CACHE, $TREDUCE_CACHE"
echo "  python      $PY$([ -x "$PY" ] || echo " (missing: run scripts/setup_env.sh)")"
echo "  options     jobs=$JOBS skip_gate=$SKIP_GATE force=$FORCE include_jellyfish256=$INCLUDE_JF256 gate_baseline=${GATE_BASELINE}ms gate_max_wait=${GATE_MAX_WAIT}s"
echo
printf '  %-11s %-9s %s\n' Phase Estimate "What (notes)"
total=0
for p in $SELECTED; do
  st=""; [ $FORCE = 0 ] && is_done "$p" && [ "$p" != tables ] && st=" [done; will skip]"
  printf '  %-11s %-9s %s\n' "$p" "$(fmt "${EST[$p]}")" "${DESC[$p]}${EST_NOTE[$p]:+ (${EST_NOTE[$p]})}$st"
  [ -z "$st" ] && total=$(( total + ${EST[$p]} ))
done
echo "  total       $(fmt $total) for the phases still to run, plus host-latency gate waits"
echo

FAILED=() INCOMPLETE=() DONE_NOW=() SKIPPED=()
for p in $SELECTED; do
  PHASE=$p
  PLOG=$LOGS/$p.log
  [ $DRY_RUN = 1 ] && PLOG=/dev/null
  if [ $FORCE = 1 ]; then
    if [ $DRY_RUN = 1 ]; then compgen -G "$DONE/$p*" > /dev/null && show "rm -f $DONE/$p $DONE/$p.*"
    else rm -f "$DONE/$p" "$DONE/$p".*; fi
  elif is_done "$p"; then
    if [ "$p" != tables ] || [ -z "$(find "$OUT/sweep" "$DIAG" "$LOGS" -newer "$DONE/tables" -type f ! -name '*.log' -print -quit 2> /dev/null)" ]; then
      say "phase $p: done earlier ($(cut -d' ' -f1 "$DONE/$p")); skipping (--force reruns it)"
      SKIPPED+=("$p"); continue
    fi
  fi
  echo
  say "=== phase $p: ${DESC[$p]} (estimate $(fmt "${EST[$p]}"); log $LOGS/$p.log)"
  [ $DRY_RUN = 1 ] || echo "=== $(date -u +%FT%TZ) phase $p start (git $GIT_REV$GIT_DIRTY)" >> "$PLOG"
  "phase_$p"; rc=$?
  case $rc in
    0) end_step "$p"; DONE_NOW+=("$p"); [ $DRY_RUN = 1 ] || say "phase $p: done" ;;
    2) INCOMPLETE+=("$p"); say "phase $p: incomplete (see the warnings above); rerun it later" ;;
    *) FAILED+=("$p"); say "phase $p: FAILED (rc=$rc; log $LOGS/$p.log)"
       if [ "$p" = build ]; then say "stopping: the later phases need the CUDA binaries"; break; fi ;;
  esac
done
PHASE="" PLOG=/dev/null

echo
if [ $DRY_RUN = 1 ]; then
  echo "dry run complete: $(echo $SELECTED | wc -w) phase(s) listed; nothing was executed."
  exit 0
fi
say "summary: done ${DONE_NOW[*]:-none}; skipped ${SKIPPED[*]:-none}; incomplete ${INCOMPLETE[*]:-none}; failed ${FAILED[*]:-none}"
[ -f "$OUT/tables/FINDINGS_TABLES.md" ] && say "tables: $OUT/tables/FINDINGS_TABLES.md; Figure 4: $OUT/figures/figure4_cuda_speedup_by_workload.pdf"
[ ${#FAILED[@]} -eq 0 ]
