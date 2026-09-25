#!/usr/bin/env bash
set -euo pipefail
mkdir -p build
ARCH="${CUDA_ARCH:-native}"
nvcc -O3 -std=c++17 -arch="${ARCH}" cuda/src/wraparound_sumcheck.cu -o build/wraparound_sumcheck_cuda
