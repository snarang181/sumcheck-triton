"""Side-by-side anatomy of matched Triton and CUDA eval kernels (why does Triton use more
registers?).

For each (workload, bit width, point index), element layout, specialized points:
  registers, spills, static/dynamic shared memory, SASS instruction count and mix,
  PTX virtual-register declarations by width.
Triton kernels come from the Triton cache (no compile when cached); run once per variant:

  TRITON_CACHE_DIR=.triton-cache-h100-table5 python .../register_anatomy.py --variant base
  TRITON_CACHE_DIR=.triton-cache-h100-treduce python .../register_anatomy.py --variant treduce
  python .../register_anatomy.py --variant cuda
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [
    str(Path(__file__).resolve().parent),
    str(ROOT / "benchmarks" / "field_sweep"),
    str(ROOT / "src"),
]

CASES = [
    ("zk_spartan_2", 32, 2),
    ("zk_spartan_2", 256, 2),
    ("zk_complete_add_3", 32, 2),
    ("zk_complete_add_3", 256, 2),
    ("zk_witness_id_point_1", 256, 2),
    ("zk_vanilla_permcheck_hp", 256, 2),
]
GROUPS = {
    "mul": ("IMAD", "IMUL"),
    "add": ("IADD3", "IADD", "IADD32I", "LEA"),
    "logic/shift": ("LOP3", "SHF", "SHL", "SHR", "PRMT", "BFE", "BFI", "FLO", "POPC"),
    "compare/select": ("ISETP", "SEL", "PLOP3", "P2R", "R2P", "CSET", "ICMP", "VOTE"),
    "global mem": ("LDG", "STG", "LD", "ST", "RED", "ATOM"),
    "shared mem": ("LDS", "STS", "LDSM"),
    "shuffle": ("SHFL",),
    "sync/branch": ("BAR", "BRA", "BSSY", "BSYNC", "WARPSYNC", "EXIT", "RET", "CALL", "JMP"),
    "move/const": ("MOV", "S2R", "S2UR", "CS2R", "ULDC", "UMOV", "LDC"),
}


def sass_mix(sass: str) -> dict:
    ops = Counter()
    for ln in sass.splitlines():
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(?:@!?U?P[T0-9]\s+)?([A-Z0-9_]+)", ln)
        if m:
            ops[m.group(1).split(".")[0]] += 1
    mix = {g: sum(ops[o] for o in names) for g, names in GROUPS.items()}
    mix["total"] = sum(ops.values())
    mix["other"] = mix["total"] - sum(v for k, v in mix.items() if k != "total")
    return mix


def ptx_regs(ptx: str) -> dict:
    out = {}
    for kind, n in re.findall(r"\.reg\s+\.(b64|b32|b16|pred|u64|u32|s64|s32)\s+%\w+<(\d+)>", ptx):
        key = "64-bit" if "64" in kind else ("pred" if kind == "pred" else "32-bit")
        out[key] = out.get(key, 0) + int(n)
    return out


def triton_case(w: str, bw: int, p: int, variant: str) -> dict:
    import torch
    import triton_sweep as ts
    from workload_specs import WORKLOAD_BY_NAME

    from zkduel.fields import get_field

    if variant == "treduce":
        import treduce

        treduce.install()
    spec, field = WORKLOAD_BY_NAME[w], get_field(bw)
    dev = torch.device("cuda")
    args = (
        torch.empty((spec.num_vars, 256, field.limbs), dtype=torch.uint32, device=dev),
        torch.empty((1, field.limbs), dtype=torch.uint32, device=dev),
        128,
        torch.empty(field.limbs, dtype=torch.uint32, device=dev),
        torch.empty((spec.degree + 1, field.limbs), dtype=torch.uint32, device=dev),
        field.nprime,
    )
    k = ts._round_eval_specialized_kernel.warmup(
        *args,
        WORKLOAD=spec.ident,
        NUM_VARS=spec.num_vars,
        POINT_IDX=p,
        POINT_MODE=1,
        LIMBS=field.limbs,
        LAYOUT=0,
        BLOCK_SIZE=ts.BLOCK,
        grid=(1,),
    )
    k._init_handles()
    with tempfile.NamedTemporaryFile(suffix=".cubin") as f:
        f.write(k.asm["cubin"])
        f.flush()
        sass = subprocess.run(["cuobjdump", "-sass", f.name], capture_output=True, text=True).stdout
    return {
        "regs": k.n_regs,
        "spills": k.n_spills,
        "shared_bytes": k.metadata.shared,
        "sass": sass_mix(sass),
        "ptx_regs": ptx_regs(k.asm["ptx"]),
    }


def cuda_cases(binary: Path, cases) -> dict:
    from workload_specs import WORKLOAD_BY_NAME

    usage = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(binary)], capture_output=True, text=True
    ).stdout
    names = re.findall(r"Function (\S+):", usage)
    dem = subprocess.run(
        ["c++filt"], input="\n".join(names), capture_output=True, text=True
    ).stdout.splitlines()
    by_demangled = dict(zip(dem, names))
    blocks = usage.split("Function ")
    res = {}
    for w, bw, p in cases:
        spec = WORKLOAD_BY_NAME[w]
        want = f"void eval_kernel<{spec.ident}, {spec.num_vars}, {bw // 32}, false, 1, {p}>"
        mangled = next(m for d, m in by_demangled.items() if d.startswith(want))
        blk = next(b for b in blocks if b.startswith(mangled + ":"))
        reg = int(re.search(r"REG:(\d+)", blk).group(1))
        spill = int(re.search(r"STACK:(\d+)", blk).group(1)) + int(
            re.search(r"LOCAL:(\d+)", blk).group(1)
        )
        shared = int(re.search(r"SHARED:(\d+)", blk).group(1))
        sass = subprocess.run(
            ["cuobjdump", "-sass", "-fun", mangled, str(binary)], capture_output=True, text=True
        ).stdout
        ptx = (
            subprocess.run(
                ["cuobjdump", "-ptx", str(binary)], capture_output=True, text=True
            ).stdout
            if False
            else ""
        )
        res[f"{w}|{bw}|{p}"] = {
            "regs": reg,
            "spills": spill,
            "shared_bytes": f"{shared} static + {128 * (bw // 32) * 4} dynamic",
            "sass": sass_mix(sass),
            "ptx_regs": ptx_regs(ptx) if ptx else None,
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "treduce", "cuda"], required=True)
    ap.add_argument("--cuda-binary", type=Path, default=ROOT / "build" / "field_sumcheck_cuda")
    ap.add_argument(
        "--out",
        type=Path,
        default=ROOT / "experiments" / "h100" / "diag" / "register_anatomy.jsonl",
    )
    args = ap.parse_args()
    if args.variant == "cuda":
        res = cuda_cases(args.cuda_binary, CASES)
    else:
        res = {f"{w}|{bw}|{p}": triton_case(w, bw, p, args.variant) for w, bw, p in CASES}
    with args.out.open("a") as f:
        for key, v in res.items():
            w, bw, p = key.split("|")
            f.write(
                json.dumps(
                    {
                        "variant": args.variant,
                        "workload": w,
                        "bit_width": int(bw),
                        "point": int(p),
                        **v,
                    }
                )
                + "\n"
            )
    for key, v in res.items():
        print(args.variant, key, "regs", v["regs"], "sass", v["sass"]["total"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
