"""Compile one field-sweep eval kernel into TRITON_CACHE_DIR without running it.

  python experiments/h100/tools/warm_cache.py WORKLOAD BIT_WIDTH POINT_IDX [--point-mode specialized]

Uses the same argument types, alignments and constexprs as triton_sweep._launch_round_eval,
so the cache key matches what the sweep launches. Run many of these in parallel to warm the
cache per kernel instead of per workload (Triton compiles one kernel per process at a time).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks" / "field_sweep"), str(ROOT / "src")]

import torch  # noqa: E402
import triton_sweep as ts  # noqa: E402
from workload_specs import WORKLOAD_BY_NAME  # noqa: E402

from zkduel.fields import get_field  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("bit_width", type=int)
    ap.add_argument("point_idx", type=int)
    ap.add_argument("--point-mode", default="specialized")
    ap.add_argument("--layout", default="element")
    args = ap.parse_args()
    spec, field = WORKLOAD_BY_NAME[args.workload], get_field(args.bit_width)
    dev = torch.device("cuda")
    tables = torch.empty((spec.num_vars, 256, field.limbs), dtype=torch.uint32, device=dev)
    partials = torch.empty((1, field.limbs), dtype=torch.uint32, device=dev)
    modulus = torch.empty(field.limbs, dtype=torch.uint32, device=dev)
    points = torch.empty((spec.degree + 1, field.limbs), dtype=torch.uint32, device=dev)
    t0 = time.time()
    ts._round_eval_specialized_kernel.warmup(
        tables,
        partials,
        128,
        modulus,
        points,
        field.nprime,
        WORKLOAD=spec.ident,
        NUM_VARS=spec.num_vars,
        POINT_IDX=args.point_idx,
        POINT_MODE=1 if args.point_mode == "specialized" else 0,
        LIMBS=field.limbs,
        LAYOUT=0 if args.layout == "element" else 1,
        BLOCK_SIZE=ts.BLOCK,
        grid=(1,),
    )
    print(
        f"{args.workload} {args.bit_width}-bit point {args.point_idx}: {time.time() - t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
