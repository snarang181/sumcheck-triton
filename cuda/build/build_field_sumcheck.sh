#!/usr/bin/env bash
set -euo pipefail
mkdir -p build
ARCH="${CUDA_ARCH:-native}"
nvcc -O3 -std=c++17 -arch="${ARCH}" cuda/src/field_sumcheck.cu -o build/field_sumcheck_cuda -lcrypto
