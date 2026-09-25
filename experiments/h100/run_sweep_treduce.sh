#!/usr/bin/env bash
# Full-matrix ablation: the main sweep's settings and CUDA binary, with Triton's block
# reduction swapped for tl.reduce (tools/triton_sweep_treduce.py) and the int64-offset fix
# (now in triton_sweep.py). jellyfish zerocheck at 256 bits is left out: its eval kernels
# take hours each to compile and were not precompiled for this variant. The jellyfish run
# uses the int64-fixed CUDA binary, since the original wraps at 32-bit rounds=28.
set -uo pipefail
cd "$(dirname "$0")/../.."
source experiments/h100/env_table5.sh
export TRITON_CACHE_DIR="$PWD/.triton-cache-h100-treduce"
H=experiments/h100/sweep_treduce
mkdir -p $H
ZK=$(python3 -c "import sys; sys.path.insert(0, 'benchmarks/field_sweep'); from workload_specs import ZK_WORKLOAD_NAMES as z; print(' '.join(w for w in z if w != 'zk_jellyfish_zerocheck_hp'))")
COMMON=(--rounds 14 16 18 20 22 24 26 28 --warmups 2 --repeats 10 --point-mode specialized
        --per-case-timeout 1800 --skip-build --python "$ZKDUEL_VENV/bin/python"
        --triton-script experiments/h100/tools/triton_sweep_treduce.py --tmp-dir $H/cases)
date -u +%FT%TZ > $H/started_at.txt
python -u benchmarks/field_sweep/resilient_compare_triton_cuda.py --workloads $ZK --bit-widths 32 64 128 256 \
  --cuda-binary build/field_sumcheck_cuda "${COMMON[@]}" --out $H/part_a.csv
python -u benchmarks/field_sweep/resilient_compare_triton_cuda.py --workloads zk_jellyfish_zerocheck_hp --bit-widths 32 64 128 \
  --cuda-binary build/field_sumcheck_cuda_int64_r28 "${COMMON[@]}" --out $H/part_b.csv
python3 - <<'PY'
import csv
rows = []
for part in ("part_a", "part_b"):
    with open(f"experiments/h100/sweep_treduce/{part}.csv", newline="") as f:
        r = csv.DictReader(f); fields = r.fieldnames; rows += list(r)
with open("experiments/h100/sweep_treduce/triton_treduce_cuda_field_resilient_r14_28_h100.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
PY
python3 experiments/h100/tools/summarize_sweep.py $H/triton_treduce_cuda_field_resilient_r14_28_h100.csv --label "H100, Triton with tl.reduce" --md $H/summary_treduce.md > /dev/null
python3 experiments/h100/tools/classify_skips.py $H/triton_treduce_cuda_field_resilient_r14_28_h100.csv --md $H/skips_treduce.md > /dev/null
date -u +%FT%TZ > $H/finished_at.txt
