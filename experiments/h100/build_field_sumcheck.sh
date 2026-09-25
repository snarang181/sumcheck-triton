#!/usr/bin/env bash
# Same command as cuda/build/build_field_sumcheck.sh, plus include/link paths for the
# OpenSSL 3.0.2 headers unpacked from libssl-dev (the host has only the libssl3 runtime).
#   experiments/h100/build_field_sumcheck.sh [OUTPUT]   (default build/field_sumcheck_cuda)
# Run from the repository root with env_table5.sh sourced.
set -euo pipefail
OUT_BIN="${1:-build/field_sumcheck_cuda}"
mkdir -p "$(dirname "$OUT_BIN")"
ARCH="${CUDA_ARCH:-native}"
nvcc -O3 -std=c++17 -arch="${ARCH}" \
  -I"${ZKDUEL_OPENSSL_INCLUDE}" -I"${ZKDUEL_OPENSSL_INCLUDE}/x86_64-linux-gnu" \
  cuda/src/field_sumcheck.cu -o "$OUT_BIN" \
  -L/usr/lib/x86_64-linux-gnu -l:libcrypto.so.3
