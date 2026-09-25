"""Compile field-sweep eval kernels and print a SHA-256 per IR stage (ttir, ttgir, llir, ptx, cubin),
so two Triton builds can be checked for byte-identical output.

  TRITON_CACHE_DIR=$(mktemp -d) python ir_hashes.py --cases zk_spartan_2:32:2 ... >> out.jsonl

Use a fresh cache per run so every kernel really compiles with the interpreter's Triton.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks" / "field_sweep"), str(ROOT / "src")]

import torch  # noqa: E402
import triton  # noqa: E402
import triton_sweep as ts  # noqa: E402
from workload_specs import WORKLOAD_BY_NAME  # noqa: E402

from zkduel.fields import get_field  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", required=True, help="WORKLOAD:BITS:POINT")
    ap.add_argument("--point-mode", default="specialized")
    ap.add_argument("--layout", default="element")
    ap.add_argument("--label", required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    for case in args.cases:
        w, bw, p = case.split(":")
        spec, field = WORKLOAD_BY_NAME[w], get_field(int(bw))
        t0 = time.time()
        k = ts._round_eval_specialized_kernel.warmup(
            torch.empty((spec.num_vars, 256, field.limbs), dtype=torch.uint32, device=dev),
            torch.empty((1, field.limbs), dtype=torch.uint32, device=dev),
            128,
            torch.empty(field.limbs, dtype=torch.uint32, device=dev),
            torch.empty((spec.degree + 1, field.limbs), dtype=torch.uint32, device=dev),
            field.nprime,
            WORKLOAD=spec.ident,
            NUM_VARS=spec.num_vars,
            POINT_IDX=int(p),
            POINT_MODE=1 if args.point_mode == "specialized" else 0,
            LIMBS=field.limbs,
            LAYOUT=0 if args.layout == "element" else 1,
            BLOCK_SIZE=ts.BLOCK,
            grid=(1,),
        )
        rec = {
            "label": args.label,
            "triton": triton.__version__,
            "triton_path": str(Path(triton.__file__).parent),
            "case": case,
            "compile_s": round(time.time() - t0, 2),
        }
        for stage in ("ttir", "ttgir", "llir", "ptx", "cubin"):
            data = k.asm[stage]
            rec[stage] = hashlib.sha256(data if isinstance(data, bytes) else data.encode()).hexdigest()
        print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
