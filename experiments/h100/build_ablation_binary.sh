#!/usr/bin/env bash
# Build an ablation variant of the CUDA field-sweep binary:
#   experiments/h100/build_ablation_binary.sh OUT_NAME "W1 W2 ..." PATCH [PATCH ...]
# Copies cuda/src/field_sumcheck.cu to a temp dir, applies the patches in order, keeps only
# the listed workload ids in WORKLOAD_LIST (fewer template instantiations, much faster to
# build), and compiles with the same nvcc flags as build_field_sumcheck.sh.
# The binary goes to $BUILD_DIR/OUT_NAME (default BUILD_DIR=build, relative to the repo root).
set -euo pipefail
cd "$(dirname "$0")/../.."
source experiments/h100/env_table5.sh
out="$1"; ids="$2"; shift 2
tmp=$(mktemp -d); mkdir -p "$tmp/cuda/src"; cp cuda/src/field_sumcheck.cu "$tmp/cuda/src/"
for p in "$@"; do (cd "$tmp" && patch -s -p1 < "$OLDPWD/$p"); done
python3 - "$tmp/cuda/src/field_sumcheck.cu" "$ids" <<'PY'
import re, sys
path, ids = sys.argv[1], sys.argv[2].split()
src = open(path).read()
start = src.index("#define WORKLOAD_LIST(X)"); end = src.index("\n\n", start)
table = dict(re.findall(r"X\((\d+),\s*(\d+)\)", src[start:end]))
new = "#define WORKLOAD_LIST(X) \\\n  " + " ".join(f"X({i}, {table[i]})" for i in ids)
open(path, "w").write(src[:start] + new + src[end:])
PY
BUILD_DIR="${BUILD_DIR:-build}"
mkdir -p "$BUILD_DIR"
nvcc -O3 -std=c++17 -arch="${CUDA_ARCH:-native}" \
  -I"${ZKDUEL_OPENSSL_INCLUDE}" -I"${ZKDUEL_OPENSSL_INCLUDE}/x86_64-linux-gnu" \
  "$tmp/cuda/src/field_sumcheck.cu" -o "$BUILD_DIR/$out" -L/usr/lib/x86_64-linux-gnu -l:libcrypto.so.3
rm -rf "$tmp"
echo "built $BUILD_DIR/$out (workloads: $ids; patches: $*)"
