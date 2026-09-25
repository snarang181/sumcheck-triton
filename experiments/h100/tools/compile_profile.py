"""Profile the Triton compile of one field-sweep eval kernel: time per lowering stage,
IR size per stage, SASS size, registers, and peak memory of the compiler and of ptxas.

  TRITON_CACHE_DIR=$(mktemp -d) python experiments/h100/tools/compile_profile.py WORKLOAD BIT_WIDTH POINT

Use a fresh TRITON_CACHE_DIR so the kernel actually compiles. Prints one JSON line. Works
with Triton 3.7 and 3.8 (knobs.compilation.listener).
"""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks" / "field_sweep"), str(ROOT / "src")]

import torch  # noqa: E402
import triton  # noqa: E402
import triton_sweep as ts  # noqa: E402
from triton import knobs  # noqa: E402
from workload_specs import WORKLOAD_BY_NAME  # noqa: E402

from zkduel.fields import get_field  # noqa: E402

RECORDS: list[dict] = []


def listener(*, src, metadata, metadata_group, times, cache_hit):
    if metadata.get("name") != "_round_eval_specialized_kernel":
        return
    RECORDS.append(
        {
            "cache_hit": cache_hit,
            "ir_init_s": times.ir_initialization / 1e6,
            "stages_s": {k: v / 1e6 for k, v in times.lowering_stages},
            "total_s": times.total / 1e6,
            "files": dict(metadata_group),
        }
    )


def lines(path: str) -> int:
    with open(path, errors="replace") as f:
        return sum(1 for _ in f)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("workload")
    ap.add_argument("bit_width", type=int)
    ap.add_argument("point_idx", type=int)
    ap.add_argument("--point-mode", default="specialized")
    ap.add_argument("--layout", default="element")
    ap.add_argument("--variant", choices=["base", "vecload"], default="base")
    args = ap.parse_args()
    if args.variant == "vecload":
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import vecload

        vecload.install()
    knobs.compilation.listener = listener
    spec, field = WORKLOAD_BY_NAME[args.workload], get_field(args.bit_width)
    dev = torch.device("cuda")
    tables = torch.empty((spec.num_vars, 256, field.limbs), dtype=torch.uint32, device=dev)
    partials = torch.empty((1, field.limbs), dtype=torch.uint32, device=dev)
    modulus = torch.empty(field.limbs, dtype=torch.uint32, device=dev)
    points = torch.empty((spec.degree + 1, field.limbs), dtype=torch.uint32, device=dev)
    t0 = time.time()
    kernel = ts._round_eval_specialized_kernel.warmup(
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
    wall = time.time() - t0
    rec = {
        "triton": triton.__version__,
        "workload": args.workload,
        "bit_width": args.bit_width,
        "point": args.point_idx,
        "point_mode": args.point_mode,
        "layout": args.layout,
        "variant": args.variant,
        "wall_s": round(wall, 1),
    }
    if RECORDS:
        r = RECORDS[-1]
        rec.update(
            cache_hit=r["cache_hit"],
            ir_init_s=round(r["ir_init_s"], 2),
            stages_s={k: round(v, 2) for k, v in r["stages_s"].items()},
            total_s=round(r["total_s"], 1),
        )
        files = r["files"]
        rec["ir_lines"] = {
            ext: lines(p)
            for ext in ("ttir", "ttgir", "llir", "ptx")
            for name, p in files.items()
            if name.endswith("." + ext)
        }
        cubin = next((p for n, p in files.items() if n.endswith(".cubin")), None)
        if cubin:
            sass = subprocess.run(
                ["cuobjdump", "-sass", cubin], capture_output=True, text=True
            ).stdout
            rec["sass_instructions"] = sum(
                1
                for ln in sass.splitlines()
                if ln.strip().startswith("/*") and "*/" in ln and len(ln.strip()) > 12
            )
    kernel._init_handles() if hasattr(kernel, "_init_handles") else None
    rec["n_regs"], rec["n_spills"] = (
        getattr(kernel, "n_regs", None),
        getattr(kernel, "n_spills", None),
    )
    rec["peak_rss_gb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
    rec["peak_rss_children_gb"] = round(
        resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e6, 2
    )
    print(json.dumps(rec), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
