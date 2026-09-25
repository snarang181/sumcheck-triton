# Source this file to reproduce the paper's Table 5 software stack on this H100 host.
#
#   Table 5 (A100 run)          This host
#   GPU    A100-SXM4-80GB       H100 PCIe 80GB (intended change)
#   driver 580.126.20           kernel module 570.148.08 + CUDA 13.0 forward-compat
#                               user-mode driver 580.126.20 (libcuda.so.580.126.20)
#   nvcc   12.9                 12.9.1 (NVIDIA redist tarball)
#   torch  2.12.0+cu130         2.12.0+cu130
#   triton 3.7.0                3.7.0
#   OS     Ubuntu 24.04.4       Ubuntu 22.04 (host compiler / glibc only)
#
# See experiments/h100/README.md for how each component was installed.
#
# On another machine, run scripts/setup_env.sh: it writes the machine's paths to .zkduel_env.sh
# at the repository root, which this file reads first. Variables already set in the environment
# take precedence over both; the defaults below are the original host's paths.

_zkduel_local="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/.zkduel_env.sh"
[ -f "$_zkduel_local" ] && . "$_zkduel_local"
unset _zkduel_local

export ZKDUEL_VENV="${ZKDUEL_VENV:-/home/ubuntu/venvs/zkduel-table5}"
export CUDA_HOME="${CUDA_HOME:-/home/ubuntu/opt/cuda-12.9.1}"
export ZKDUEL_CUDA_COMPAT="${ZKDUEL_CUDA_COMPAT-/home/ubuntu/opt/cuda-compat-580.126.20/usr/local/cuda-13.0/compat}"
export ZKDUEL_OPENSSL_INCLUDE="${ZKDUEL_OPENSSL_INCLUDE:-/home/ubuntu/opt/openssl-dev/include}"

export PATH="$CUDA_HOME/bin:$ZKDUEL_VENV/bin:$PATH"
# CUDA 13.0 forward-compatibility user-mode driver (needed when the kernel module is older than
# 580; see scripts/setup_env.sh). Set ZKDUEL_CUDA_COMPAT= (empty) when the installed driver is
# already new enough.
if [ -n "$ZKDUEL_CUDA_COMPAT" ] && [ -d "$ZKDUEL_CUDA_COMPAT" ]; then
  export LD_LIBRARY_PATH="$ZKDUEL_CUDA_COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export TRITON_LIBCUDA_PATH="$ZKDUEL_CUDA_COMPAT"
fi
export PYTHONPATH=src
export PYTHONUNBUFFERED=1
