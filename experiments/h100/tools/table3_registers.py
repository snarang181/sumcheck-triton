"""Regenerate the paper's Table 3 (registers/thread and spills) on the current GPU.

CUDA: `cuobjdump --dump-resource-usage` on the field-sweep binary, per eval_kernel
instantiation <workload, vars, limbs, layout, point mode, point index>.
Triton: compile the same eval kernels through triton_sweep.py (tiny problem, element
layout) and read n_regs / n_spills from the loaded kernels.

Run with experiments/h100/env_table5.sh sourced, from the repo root:
  python experiments/h100/tools/table3_registers.py --md experiments/h100/env/table3_h100.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks" / "field_sweep"), str(ROOT / "src")]

from workload_specs import WORKLOAD_BY_NAME  # noqa: E402

MODES = {0: "generic", 1: "specialized"}


def cuda_regs(binary: Path) -> dict[tuple[int, int, str], list[tuple[int, int, int]]]:
    out = subprocess.run(
        ["cuobjdump", "--dump-resource-usage", str(binary)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    names, usage = [], []
    for line in out.splitlines():
        m = re.search(r"Function (\S+):", line)
        if m:
            names.append(m.group(1))
            continue
        m = re.search(r"REG:(\d+) STACK:(\d+).*LOCAL:(\d+)", line)
        if m and len(usage) < len(names):
            usage.append(tuple(int(x) for x in m.groups()))
    demangled = subprocess.run(
        ["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True
    ).stdout.splitlines()
    regs: dict[tuple[int, int, str], list[tuple[int, int, int]]] = defaultdict(list)
    for fn, (reg, stack, local) in zip(demangled, usage):
        m = re.search(r"eval_kernel<(\d+), (\d+), (\d+), (\w+), (\d+), (\d+)>", fn)
        if m and m.group(4) == "false":
            w, _nv, limbs, _lm, mode, idx = m.groups()
            regs[(int(w), 32 * int(limbs), MODES[int(mode)])].append((int(idx), reg, stack + local))
    return regs


def triton_regs(
    cases: list[tuple[str, int, str]],
) -> dict[tuple[int, int, str], list[tuple[int, int, int]]]:
    import triton_sweep as ts

    for w, bw, mode in cases:
        ts._time_one(
            w,
            bw,
            3,
            warmups=0,
            repeats=1,
            seed=1,
            timing_mode="whole-run",
            challenge_mode="fixed",
            layout="element",
            point_mode=mode,
        )
    regs: dict[tuple[int, int, str], list[tuple[int, int, int]]] = defaultdict(list)
    for entry in ts._round_eval_specialized_kernel.device_caches.values():
        for key, kernel in entry[0].items():
            consts = (
                [v for kind, v in key if kind == "constexpr"]
                if isinstance(key, (list, tuple))
                else [int(x) for x in re.findall(r"\('constexpr', (\d+)\)", str(key))]
            )
            if not consts:
                consts = [int(x) for x in re.findall(r"\('constexpr', (\d+)\)", str(key))]
            workload, _nv, point_idx, point_mode, limbs, layout = consts[:6]
            if layout != 0:
                continue
            regs[(workload, 32 * limbs, MODES[point_mode])].append(
                (point_idx, kernel.n_regs, kernel.n_spills)
            )
    return regs


def span(v: list[tuple[int, int, int]]) -> str:
    r = sorted(x[1] for x in v)
    return f"{r[0]}" if r[0] == r[-1] else f"{r[0]}-{r[-1]}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cuda-binary", type=Path, default=ROOT / "build" / "field_sumcheck_cuda")
    ap.add_argument(
        "--triton-cases",
        nargs="*",
        default=[
            "poly_abc:32:generic",
            "poly_abc:256:generic",
            "poly_abc:256:specialized",
            "zk_complete_add_3:256:generic",
            "zk_complete_add_3:256:specialized",
        ],
    )
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()
    cases = [(w, int(bw), mode) for w, bw, mode in (c.split(":") for c in args.triton_cases)]
    cu = cuda_regs(args.cuda_binary)
    tr = triton_regs(cases)
    ident = {spec.ident: name for name, spec in WORKLOAD_BY_NAME.items()}
    rows = [
        "| Workload | Interp. | Width | CUDA regs | Triton regs | Spills (CUDA/Triton) |",
        "|---|---|---:|---:|---:|---|",
    ]
    keys = sorted(
        set(tr)
        | {
            k
            for k in cu
            if ident.get(k[0], "").startswith(("poly_abc", "zk_complete_add", "zk_jellyfish"))
        }
    )
    for w, bw, mode in keys:
        if bw not in (32, 256):
            continue
        c, t = cu.get((w, bw, mode)), tr.get((w, bw, mode))
        cs = max(x[2] for x in c) if c else "-"
        ts_ = max(x[2] for x in t) if t else "-"
        rows.append(
            f"| {ident.get(w, w)} | {mode} | {bw} | {span(c) if c else '-'} | "
            f"{span(t) if t else '-'} | {cs}/{ts_} |"
        )
    text = "\n".join(rows)
    print(text)
    if args.md:
        args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
