"""Isolate the cost of Triton's TritonGPUCoalesce pass on a synthetic kernel.

For every load/store, buildCoalescedEncoding (lib/Dialect/TritonGPU/Transforms/CoalesceUtils.cpp)
calls mlir::getSlice(op), which recomputes a full backward and forward slice for every op it adds.
In a kernel whose ops are all connected, one call costs O(N^2) for N ops, so the pass costs
O(M * N^2) for M memory ops. The synthetic kernel has M loads summed into one accumulator,
followed by a chain of C multiply-adds (about 3 IR ops each), so M and N can be varied separately:

  python coalesce_scaling.py --out ../diag/coalesce_scaling.jsonl

Each point compiles in a fresh process and cache with MLIR_ENABLE_TIMING=1 (compile only; nothing
runs on the GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

POINTS = [("M", m, 2000) for m in (16, 32, 64, 128)] + [("C", 32, c) for c in (1000, 2000, 4000, 8000)]


def child(m: int, c: int) -> None:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def micro_kernel(in_ptr, out_ptr, M: tl.constexpr, C: tl.constexpr):
        offs = tl.arange(0, 128)
        acc = tl.zeros([128], dtype=tl.uint32)
        for i in tl.static_range(M):
            acc = acc + tl.load(in_ptr + i * 128 + offs)
        for j in tl.static_range(C):
            acc = acc * acc + j
        tl.store(out_ptr + offs, acc)

    x = torch.empty(m * 128, dtype=torch.uint32, device="cuda")
    y = torch.empty(128, dtype=torch.uint32, device="cuda")
    k = micro_kernel.warmup(x, y, M=m, C=c, grid=(1,))
    ttir = k.asm["ttir"]
    print(
        json.dumps(
            {
                "triton": triton.__version__,
                "ttir_ops": sum(1 for ln in ttir.splitlines() if " = " in ln or "tt.store" in ln),
                "mem_ops": ttir.count("tt.load") + ttir.count("tt.store"),
            }
        )
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", nargs=2, type=int)
    ap.add_argument("--out", type=Path)
    ap.add_argument(
        "--points", default=None, help="subset as SWEEP:M:C,... (default: the full list above)"
    )
    args = ap.parse_args()
    if args.child:
        child(*args.child)
        return 0
    points = POINTS
    if args.points:
        points = [(s, int(m), int(c)) for s, m, c in (x.split(":") for x in args.points.split(","))]
    for sweep, m, c in points:
        with tempfile.TemporaryDirectory() as cache:
            env = dict(os.environ, MLIR_ENABLE_TIMING="1", TRITON_CACHE_DIR=cache)
            p = subprocess.run(
                [sys.executable, __file__, "--child", str(m), str(c)],
                env=env,
                capture_output=True,
                text=True,
            )
        rec = {"sweep": sweep, "M": m, "C": c}
        try:
            rec.update(json.loads(p.stdout.strip().splitlines()[-1]))
        except (IndexError, json.JSONDecodeError):
            rec["error"] = p.stderr[-500:]
        mt = re.search(r"^\s+([0-9.]+) \(\s*([0-9.]+)%\)\s+TritonGPUCoalesce$", p.stderr, re.M)
        if mt:
            rec["coalesce_s"] = float(mt.group(1))
            rec["coalesce_share"] = float(mt.group(2))
        print(json.dumps(rec), flush=True)
        if args.out:
            with args.out.open("a") as f:
                f.write(json.dumps(rec) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
