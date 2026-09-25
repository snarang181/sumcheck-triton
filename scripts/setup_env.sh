#!/usr/bin/env bash
# Set up the zkDuel paper's software stack (the paper's Table 5) and check everything else the
# experiments need. Run it before scripts/run_paper_experiments.sh.
#
#   scripts/setup_env.sh [--venv DIR] [--with-triton38] [--triton38-venv DIR] [--check-only]
#                        [--fetch-local DIR] [--install-system] [--no-config]
#
# What it does:
#   1. Python environment (Python 3.12, torch 2.12.0+cu130, triton 3.7.0, numpy, matplotlib,
#      pytest), with uv if it is installed (uv also provides Python 3.12), else python3.12 -m venv.
#      --with-triton38 adds a second environment with Triton 3.8.0 (and torch 2.14.0+cu130, which
#      requires it); it is used only by the compile-time profiling phase.
#   2. Checks, each reported as ok / WARN / MISSING with what to do:
#        nvcc 12.9 (the CUDA baseline was built with 12.9.1) and a host C++ compiler;
#        the OpenSSL 3 headers and libcrypto.so.3 the CUDA binary links for its SHA3 path;
#        an NVIDIA driver that runs CUDA 13.0 (see below);
#        nsys (split, occupancy, ablations), ncu and passwordless sudo (counters), cuobjdump and
#        c++filt (register anatomy), taskset (noise), flock.
#   3. Writes .zkduel_env.sh at the repository root with the paths it found; the experiment
#      scripts read it through experiments/h100/env_table5.sh. Variables already set in the
#      environment (ZKDUEL_VENV, CUDA_HOME, ZKDUEL_CUDA_COMPAT, ZKDUEL_OPENSSL_INCLUDE,
#      ZKDUEL_TRITON38_PY) take precedence.
# It installs no system packages unless asked:
#   --install-system    run 'sudo apt-get install -y libssl-dev' if the OpenSSL headers are missing
#   --fetch-local DIR   without root, download into DIR what the paper's host installed by hand:
#                       nvcc 12.9.1 (NVIDIA redistributable tarballs, sha256-checked), the
#                       libssl-dev headers (apt-get download + dpkg-deb -x) and, if the driver is
#                       older than R580, the CUDA 13.0 forward-compatibility driver (below)
#   --check-only        only check; create nothing
#
# CUDA 13 forward compatibility. torch 2.12.0+cu130 ships the CUDA 13.0 runtime, which needs a
# driver that supports CUDA 13.0, i.e. branch R580 or newer. The paper's H100 host had kernel
# module 570.148.08 (CUDA 12.8), so it used NVIDIA's forward-compatibility package
# cuda-compat-13-0 580.126.20: its user-mode driver (libcuda.so.580.126.20, in
# usr/local/cuda-13.0/compat inside the package) is put first on LD_LIBRARY_PATH, and Triton is
# pointed at the same directory with TRITON_LIBCUDA_PATH (env_table5.sh does both when
# ZKDUEL_CUDA_COMPAT names that directory). Forward compatibility works only on data-center GPUs
# (A100, H100, ...) and on kernel-module branches NVIDIA supports for it; see NVIDIA's "CUDA
# Compatibility" guide. The package can be unpacked without root (dpkg-deb -x); the kernel module
# is not changed. With a driver of R580 or newer, none of this is needed (ZKDUEL_CUDA_COMPAT is
# left empty). Table 5 of the paper lists driver 580.126.20, which is this user-mode driver.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$ROOT" || exit 1

TORCH_VERSION=2.12.0+cu130
TRITON_VERSION=3.7.0
TORCH38_VERSION=2.14.0+cu130   # the torch release that pairs with Triton 3.8.0
TRITON38_VERSION=3.8.0
TORCH_INDEX=https://download.pytorch.org/whl/cu130
PYTHON_VERSION=3.12
NVCC_WANT=12.9
CUDA_REDIST_VERSION=12.9.1
CUDA_REDIST_URL=https://developer.download.nvidia.com/compute/cuda/redist
CUDA_REDIST_COMPONENTS="cuda_nvcc cuda_cudart cuda_cccl cuda_cuobjdump cuda_nvdisasm cuda_nvrtc"
COMPAT_PKG=cuda-compat-13-0 COMPAT_VERSION=580.126.20-1ubuntu1 DRIVER_WANT=580

# Start from an earlier run's findings (environment variables still take precedence).
[ -f "$ROOT/.zkduel_env.sh" ] && . "$ROOT/.zkduel_env.sh"
CUDA_HOME=${CUDA_HOME:-}
VENV=${ZKDUEL_VENV:-$ROOT/.venv-table5}
VENV38=${ZKDUEL_TRITON38_PY:+$(dirname "$(dirname "$ZKDUEL_TRITON38_PY")")}
VENV38=${VENV38:-$ROOT/.venv-triton38}
WITH38=0 CHECK_ONLY=0 FETCH_DIR="" INSTALL_SYSTEM=0 WRITE_CONFIG=1

usage() { sed -n '2,/^set -uo pipefail/{/^set -uo/d;s/^# \{0,1\}//;p}' "$0"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV=${2:?--venv needs a directory}; shift ;;
    --with-triton38) WITH38=1 ;;
    --triton38-venv) VENV38=${2:?--triton38-venv needs a directory}; WITH38=1; shift ;;
    --check-only) CHECK_ONLY=1 ;;
    --fetch-local) FETCH_DIR=${2:?--fetch-local needs a directory}; shift ;;
    --install-system) INSTALL_SYSTEM=1 ;;
    --no-config) WRITE_CONFIG=0 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
[ -n "$FETCH_DIR" ] && FETCH_DIR=$(realpath -m -- "$FETCH_DIR")
VENV=$(realpath -m -- "$VENV") VENV38=$(realpath -m -- "$VENV38")

OK=() WARN=() MISSING=()
ok() { OK+=("$1"); echo "  ok       $1"; }
warn() { WARN+=("$1"); echo "  WARN     $1"; }
missing() { MISSING+=("$1"); echo "  MISSING  $1"; }
step() { echo; echo "== $*"; }

# ------------------------------------------------------------------ 1. Python environments
make_venv() {  # DIR TORCH TRITON
  local dir=$1 torch=$2 triton=$3 py=$1/bin/python
  if [ ! -x "$py" ]; then
    if command -v uv > /dev/null; then
      echo "  uv venv --python $PYTHON_VERSION $dir"
      uv venv --python "$PYTHON_VERSION" "$dir" || return 1
    elif command -v "python$PYTHON_VERSION" > /dev/null; then
      echo "  python$PYTHON_VERSION -m venv $dir"
      "python$PYTHON_VERSION" -m venv "$dir" && "$py" -m pip install -q --upgrade pip || return 1
    else
      echo "  neither uv nor python$PYTHON_VERSION is installed (install uv: https://docs.astral.sh/uv/)"
      return 1
    fi
  fi
  local pip=("$py" -m pip install)
  command -v uv > /dev/null && pip=(uv pip install --python "$py")
  echo "  ${pip[*]} --index-url $TORCH_INDEX torch==$torch"
  "${pip[@]}" --index-url "$TORCH_INDEX" "torch==$torch" || return 1
  echo "  ${pip[*]} triton==$triton numpy matplotlib pytest"
  "${pip[@]}" "triton==$triton" numpy matplotlib pytest || return 1
}
check_venv() {  # DIR TORCH TRITON LABEL
  local py=$1/bin/python out
  if [ ! -x "$py" ]; then missing "$4: no Python environment at $1 (run without --check-only)"; return 1; fi
  # Imports only; nothing here initializes CUDA.
  out=$("$py" -c "import sys, torch, triton, numpy; import matplotlib
print(f'{sys.version_info[0]}.{sys.version_info[1]} {torch.__version__} {triton.__version__}')" 2>&1 | tail -1)
  read -r pyv tv trv <<< "$out"
  if [ "${pyv:-}" = "$PYTHON_VERSION" ] && [ "${tv:-}" = "$2" ] && [ "${trv:-}" = "$3" ]; then
    ok "$4: $1 (Python $pyv, torch $tv, triton $trv)"
  else
    missing "$4: $1 has '$out'; want Python $PYTHON_VERSION, torch $2, triton $3 (rerun without --check-only)"
    return 1
  fi
}
step "Python environment (paper: Python $PYTHON_VERSION, torch $TORCH_VERSION, triton $TRITON_VERSION)"
if [ $CHECK_ONLY = 0 ]; then make_venv "$VENV" "$TORCH_VERSION" "$TRITON_VERSION" || echo "  (installation failed; see above)"; fi
check_venv "$VENV" "$TORCH_VERSION" "$TRITON_VERSION" "main environment"
if [ $WITH38 = 1 ] || [ -x "$VENV38/bin/python" ]; then
  step "Triton $TRITON38_VERSION environment (compile-time profiling only)"
  if [ $CHECK_ONLY = 0 ] && [ $WITH38 = 1 ]; then
    make_venv "$VENV38" "$TORCH38_VERSION" "$TRITON38_VERSION" || echo "  (installation failed; see above)"
  fi
  check_venv "$VENV38" "$TORCH38_VERSION" "$TRITON38_VERSION" "Triton $TRITON38_VERSION environment" && HAVE38=1
else
  echo "  (no Triton $TRITON38_VERSION environment; --with-triton38 adds one; without it the compile phase profiles $TRITON_VERSION only)"
fi

# ------------------------------------------------------------------ user-local downloads
fetch_cuda() {  # DEST: NVIDIA redistributable tarballs, merged into one tree, sha256-checked
  python3 - "$1" "$CUDA_REDIST_URL" "$CUDA_REDIST_VERSION" $CUDA_REDIST_COMPONENTS <<'PY'
import hashlib, json, os, shutil, sys, tarfile, tempfile, urllib.request
dest, base, version, comps = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
manifest = json.load(urllib.request.urlopen(f"{base}/redistrib_{version}.json"))
os.makedirs(dest, exist_ok=True)
for c in comps:
    info = manifest[c]["linux-x86_64"]
    url = f"{base}/{info['relative_path']}"
    print(f"  {url}", flush=True)
    with tempfile.TemporaryDirectory() as td:
        tar = os.path.join(td, "a.tar.xz")
        with urllib.request.urlopen(url) as r, open(tar, "wb") as f:
            shutil.copyfileobj(r, f)
        if hashlib.sha256(open(tar, "rb").read()).hexdigest() != info["sha256"]:
            raise SystemExit(f"sha256 mismatch for {url}")
        with tarfile.open(tar) as t:
            try:
                t.extractall(td, filter="data")
            except TypeError:  # Python without extraction filters
                t.extractall(td)
        top = os.path.join(td, os.path.basename(info["relative_path"])[: -len(".tar.xz")])
        shutil.copytree(top, dest, dirs_exist_ok=True, symlinks=True)
print(f"  nvcc {version} in {dest}")
PY
}
fetch_deb() {  # PACKAGE[=VERSION] DEST [FALLBACK_URL]: download a .deb without root and unpack it
  local pkg=$1 dest=$2 url=${3:-} tmp deb
  tmp=$(mktemp -d)
  if (cd "$tmp" && apt-get download "$pkg" > /dev/null 2>&1); then :
  elif [ -n "$url" ]; then (cd "$tmp" && curl -fsSLO "$url") || { rm -rf "$tmp"; return 1; }
  else rm -rf "$tmp"; return 1; fi
  deb=$(ls "$tmp"/*.deb | head -1)
  mkdir -p "$dest" && dpkg-deb -x "$deb" "$dest"
  local rc=$?; rm -rf "$tmp"; return $rc
}

# ------------------------------------------------------------------ 2. CUDA toolchain
step "CUDA toolchain (paper: nvcc $CUDA_REDIST_VERSION)"
if [ -n "$FETCH_DIR" ] && [ $CHECK_ONLY = 0 ]; then
  echo "  downloading nvcc $CUDA_REDIST_VERSION to $FETCH_DIR/cuda-$CUDA_REDIST_VERSION"
  fetch_cuda "$FETCH_DIR/cuda-$CUDA_REDIST_VERSION" && CUDA_HOME=$FETCH_DIR/cuda-$CUDA_REDIST_VERSION \
    || echo "  (download failed; see above)"
fi
NVCC=""
for c in "${CUDA_HOME:+$CUDA_HOME/bin/nvcc}" /usr/local/cuda-$NVCC_WANT/bin/nvcc "$(command -v nvcc 2>/dev/null)" /usr/local/cuda/bin/nvcc; do
  [ -n "$c" ] && [ -x "$c" ] || continue
  v=$("$c" --version 2>/dev/null | sed -n 's/.*release \([0-9.]*\),.*/\1/p')
  [ -z "$NVCC" ] && NVCC=$c NVCC_V=$v
  [ "$v" = "$NVCC_WANT" ] && { NVCC=$c NVCC_V=$v; break; }
done
if [ -z "$NVCC" ]; then
  missing "nvcc: not found. Install CUDA $NVCC_WANT (toolkit, or rerun with --fetch-local DIR for a user-local copy) and set CUDA_HOME"
else
  CUDA_HOME=$(dirname "$(dirname "$NVCC")")
  if [ "$NVCC_V" = "$NVCC_WANT" ]; then ok "nvcc $NVCC_V ($NVCC)"
  else warn "nvcc $NVCC_V at $NVCC; the paper used $NVCC_WANT (--fetch-local DIR installs it)"; fi
  [ -x "$CUDA_HOME/bin/cuobjdump" ] || command -v cuobjdump > /dev/null \
    && ok "cuobjdump" || warn "cuobjdump not found (counters phase: register anatomy)"
fi
if command -v g++ > /dev/null; then ok "host compiler: $(g++ --version | head -1)"; else missing "g++ (host compiler for nvcc): install build-essential"; fi

# ------------------------------------------------------------------ OpenSSL headers
step "OpenSSL 3 headers and libcrypto.so.3 (SHA3 in the CUDA binary)"
if [ -n "$FETCH_DIR" ] && [ $CHECK_ONLY = 0 ] && [ -z "${ZKDUEL_OPENSSL_INCLUDE:-}" ] && [ ! -f /usr/include/openssl/evp.h ]; then
  echo "  unpacking libssl-dev into $FETCH_DIR/openssl-dev (apt-get download, dpkg-deb -x; no root)"
  fetch_deb libssl-dev "$FETCH_DIR/openssl-dev.tmp" \
    && mkdir -p "$FETCH_DIR/openssl-dev" && rm -rf "$FETCH_DIR/openssl-dev/include" \
    && mv "$FETCH_DIR/openssl-dev.tmp/usr/include" "$FETCH_DIR/openssl-dev/include" \
    && rm -rf "$FETCH_DIR/openssl-dev.tmp" && ZKDUEL_OPENSSL_INCLUDE=$FETCH_DIR/openssl-dev/include \
    || echo "  (could not download libssl-dev; see below)"
fi
if [ $INSTALL_SYSTEM = 1 ] && [ $CHECK_ONLY = 0 ] && [ ! -f /usr/include/openssl/evp.h ] && [ -z "${ZKDUEL_OPENSSL_INCLUDE:-}" ]; then
  echo "  sudo apt-get install -y libssl-dev"
  sudo apt-get install -y libssl-dev || echo "  (apt-get failed)"
fi
OSSL=""
for d in "${ZKDUEL_OPENSSL_INCLUDE:-}" /usr/include; do
  [ -n "$d" ] && [ -f "$d/openssl/evp.h" ] && { OSSL=$d; break; }
done
if [ -n "$OSSL" ]; then
  ok "OpenSSL headers in $OSSL"
  ZKDUEL_OPENSSL_INCLUDE=$OSSL
else
  missing "OpenSSL headers (openssl/evp.h): 'sudo apt-get install libssl-dev' (or rerun with --install-system), or without root: 'apt-get download libssl-dev && dpkg-deb -x libssl-dev_*.deb DIR' and set ZKDUEL_OPENSSL_INCLUDE=DIR/usr/include (--fetch-local does this)"
fi
if [ -e /usr/lib/x86_64-linux-gnu/libcrypto.so.3 ]; then ok "libcrypto.so.3"
else missing "/usr/lib/x86_64-linux-gnu/libcrypto.so.3 (the OpenSSL 3 runtime, package libssl3; the CUDA build links it)"; fi

# ------------------------------------------------------------------ driver
step "NVIDIA driver (CUDA 13.0 needs R$DRIVER_WANT or newer, or the forward-compatibility driver)"
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
GPU=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
COMPAT=${ZKDUEL_CUDA_COMPAT-}
if [ -z "$DRV" ]; then
  missing "nvidia-smi: no NVIDIA driver or GPU visible"
else
  ok "GPU: $GPU, kernel driver $DRV"
  case "$GPU" in *H100*) ;; *) warn "the paper's primary platform is an H100; the gate baseline (--gate-baseline) and occupancy-ablation settings (ZKDUEL_OCC_SMEM_TOTALS) are H100 values" ;; esac
  if [ "${DRV%%.*}" -ge "$DRIVER_WANT" ] 2>/dev/null; then
    ok "driver $DRV runs CUDA 13.0 directly; no forward-compatibility package needed"
    [ -n "${ZKDUEL_CUDA_COMPAT+x}" ] || COMPAT=""
  else
    if [ -z "$COMPAT" ] || [ ! -e "$COMPAT/libcuda.so.1" ]; then
      COMPAT=""
      if [ -n "$FETCH_DIR" ] && [ $CHECK_ONLY = 0 ]; then
        . /etc/os-release 2>/dev/null
        VERSION_ID=${VERSION_ID:-22.04}
        url=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/${COMPAT_PKG}_${COMPAT_VERSION}_amd64.deb
        echo "  unpacking $COMPAT_PKG $COMPAT_VERSION into $FETCH_DIR/cuda-compat-${COMPAT_VERSION%%-*} (no root)"
        if fetch_deb "$COMPAT_PKG=$COMPAT_VERSION" "$FETCH_DIR/cuda-compat-${COMPAT_VERSION%%-*}" "$url"; then
          COMPAT=$FETCH_DIR/cuda-compat-${COMPAT_VERSION%%-*}/usr/local/cuda-13.0/compat
        else
          echo "  (could not download $COMPAT_PKG from apt or $url)"
        fi
      fi
      for d in "$COMPAT" /usr/local/cuda-13.0/compat; do
        [ -n "$d" ] && [ -e "$d/libcuda.so.1" ] && { COMPAT=$d; break; }
        COMPAT=""
      done
    fi
    if [ -n "$COMPAT" ]; then ok "forward-compatibility user-mode driver: $COMPAT ($(basename "$(realpath "$COMPAT/libcuda.so.1")"))"
    else missing "driver $DRV is older than R$DRIVER_WANT: install NVIDIA's $COMPAT_PKG ($COMPAT_VERSION) forward-compatibility package (data-center GPUs only; 'dpkg-deb -x' works without root, --fetch-local does it) and set ZKDUEL_CUDA_COMPAT to its usr/local/cuda-13.0/compat directory, or upgrade the driver to R$DRIVER_WANT+"; fi
  fi
fi

# ------------------------------------------------------------------ profilers and utilities
step "Profilers and utilities"
NSYS=$(command -v nsys || ls "$CUDA_HOME"/bin/nsys 2>/dev/null || true)
[ -n "$NSYS" ] && ok "nsys: $NSYS" || missing "nsys (Nsight Systems; split, occupancy and ablations phases): install nsight-systems (NVIDIA's CUDA repository or developer.nvidia.com/nsight-systems)"
NCU_P=$(command -v ncu || ls /usr/local/bin/ncu 2>/dev/null || true)
[ -n "$NCU_P" ] && ok "ncu: $NCU_P" || warn "ncu (Nsight Compute; counters phase, T11) not found: that part is skipped"
if sudo -n true 2>/dev/null; then ok "passwordless sudo (ncu needs root when RmProfilingAdminOnly=1)"
else warn "sudo -n fails: the counters phase skips Nsight Compute unless sudo works without a prompt (e.g. run 'sudo -v' just before)"; fi
for t in c++filt taskset flock awk; do
  command -v $t > /dev/null && ok "$t" || warn "$t not found"
done

# ------------------------------------------------------------------ 3. .zkduel_env.sh
if [ $WRITE_CONFIG = 1 ] && [ $CHECK_ONLY = 0 ]; then
  step "Writing $ROOT/.zkduel_env.sh"
  {
    echo "# Written by scripts/setup_env.sh on $(date -u +%FT%TZ); read by experiments/h100/env_table5.sh."
    echo "# Variables set in the environment take precedence."
    echo ": \"\${ZKDUEL_VENV:=$VENV}\""
    [ -n "${CUDA_HOME:-}" ] && echo ": \"\${CUDA_HOME:=$CUDA_HOME}\""
    [ -n "${ZKDUEL_OPENSSL_INCLUDE:-}" ] && echo ": \"\${ZKDUEL_OPENSSL_INCLUDE:=$ZKDUEL_OPENSSL_INCLUDE}\""
    echo "# CUDA 13.0 forward-compatibility user-mode driver; empty when the driver is R$DRIVER_WANT or newer."
    echo ": \"\${ZKDUEL_CUDA_COMPAT=$COMPAT}\""
    [ "${HAVE38:-0}" = 1 ] && echo ": \"\${ZKDUEL_TRITON38_PY:=$VENV38/bin/python}\"; export ZKDUEL_TRITON38_PY"
  } > "$ROOT/.zkduel_env.sh"
  sed 's/^/  /' "$ROOT/.zkduel_env.sh"
fi

echo
echo "summary: ${#OK[@]} ok, ${#WARN[@]} warnings, ${#MISSING[@]} missing"
for m in ${MISSING[@]+"${MISSING[@]}"}; do echo "  MISSING  $m"; done
if [ ${#MISSING[@]} -eq 0 ]; then
  echo "next: scripts/run_paper_experiments.sh --dry-run, then scripts/run_paper_experiments.sh"
  exit 0
fi
exit 1
