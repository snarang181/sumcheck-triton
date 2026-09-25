"""Split Triton-vs-CUDA wall time into GPU kernel time and host-side time.

For each (workload, bit width, rounds), with the sweep's settings:
  1. run the Triton sweep script and the CUDA binary as the harness does -> wall median;
  2. run both again under `nsys profile --trace=cuda` and sum the durations of the
     SumCheck kernels (eval + reduce + fold) per SumCheck run, i.e. GPU-busy time;
  3. read registers/thread for every launched kernel from the nsys kernel trace.

The ratio of GPU-busy times is the kernel-only CUDA advantage; the rest of the wall-clock
gap is host-side (Python/Triton launch path, synchronizing copies) and does not depend
on register allocation or occupancy.

Variants: --triton-script experiments/h100/tools/triton_sweep_treduce.py (with its own
--triton-cache-dir) measures the tl.reduce ablation; --cuda-binary build/field_sumcheck_cuda_occ
with --cuda-env ZKDUEL_EVAL_EXTRA_SMEM=N measures CUDA at reduced occupancy (achieved
blocks/SM are read from the binary's ZKDUEL_REPORT_OCCUPANCY output). --label names the
variant in the output records.

Run with experiments/h100/env_table5.sh sourced, from the repo root.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "experiments" / "h100" / "diag"
# Configurations whose Triton kernels cannot be compiled within the run: the diagnostic
# has no timeout, so it records these as skipped instead of compiling for hours.
DEFAULT_SKIPS = {
    ("zk_jellyfish_zerocheck_hp", 256): "Triton compile of the 256-bit eval kernels takes "
    "1.8-6.7 h per kernel on this host; point 7 did not finish within 6.8 h",
}
TRITON_KERNELS = ("_round_eval_specialized_kernel", "_reduce_pairs_kernel", "_fold_kernel")
CUDA_KERNELS = ("eval_kernel", "reduce_pairs_kernel", "fold_kernel")
EVAL_KERNELS = ("_round_eval_specialized_kernel", "eval_kernel")


def sh(cmd: list[str], log: Path, env: dict[str, str]) -> None:
    with log.open("w") as f:
        subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, check=True, env=env)


def first_row(path: Path) -> dict[str, str]:
    with path.open(newline="") as f:
        return next(csv.DictReader(f))


def base_name(name: str) -> str:
    name = name.removeprefix("void ")
    return name.split("<", 1)[0].split("(", 1)[0]


def kernel_stats(rep: Path, prefixes: tuple[str, ...]) -> dict[str, object]:
    stem = rep.with_suffix("")
    subprocess.run(
        [
            "nsys",
            "stats",
            "-q",
            "-r",
            "cuda_gpu_trace",
            "-f",
            "csv",
            "-o",
            str(stem),
            "--force-overwrite",
            "true",
            str(rep),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    total_ns, eval_ns, launches, regs = 0.0, 0.0, 0, {}
    with Path(f"{stem}_cuda_gpu_trace.csv").open(newline="") as f:
        for row in csv.DictReader(f):
            name = base_name(row.get("Name", ""))
            if name not in prefixes:
                continue
            d = float(row["Duration (ns)"])
            total_ns += d
            if name in EVAL_KERNELS:
                eval_ns += d
            launches += 1
            r = int(float(row.get("Reg/Trd") or 0))
            lo, hi = regs.get(name, (r, r))
            regs[name] = (min(lo, r), max(hi, r))
    # The per-launch trace and nsys's sqlite export are MBs per run; the totals above are
    # what the jsonl records keep.
    Path(f"{stem}_cuda_gpu_trace.csv").unlink(missing_ok=True)
    Path(f"{stem}.sqlite").unlink(missing_ok=True)
    return {
        "ns": total_ns,
        "eval_ns": eval_ns,
        "launches": launches,
        "regs": {k: list(v) for k, v in regs.items()},
    }


def occupancy(log: Path) -> list[int] | None:
    blocks = [int(x) for x in re.findall(r"blocks_per_sm=(\d+)", log.read_text(errors="replace"))]
    return [min(blocks), max(blocks)] if blocks else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--workloads",
        nargs="+",
        default=["zk_spartan_2", "zk_witness_id_point_1", "zk_complete_add_3"],
    )
    ap.add_argument("--bit-widths", nargs="+", type=int, default=[32, 256])
    ap.add_argument("--rounds", nargs="+", type=int, default=[14, 20, 24])
    ap.add_argument("--warmups", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--point-mode", default="specialized")
    ap.add_argument("--layout", default="element", choices=["element", "limb"])
    ap.add_argument("--cuda-args", default="", help="extra CUDA binary args, e.g. '--block 256'")
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--triton-script", default="benchmarks/field_sweep/triton_sweep.py")
    ap.add_argument(
        "--triton-cache-dir",
        default=os.environ.get("TRITON_CACHE_DIR", str(ROOT / ".triton-cache-h100-table5")),
    )
    ap.add_argument("--skip-triton", action="store_true", help="CUDA-only (e.g. occupancy runs)")
    ap.add_argument("--cuda-binary", default=str(ROOT / "build" / "field_sumcheck_cuda"))
    ap.add_argument("--cuda-env", action="append", default=[], metavar="KEY=VAL")
    ap.add_argument(
        "--diag-dir", type=Path, default=OUT, help="raw outputs go under DIAG_DIR/raw/LABEL"
    )
    ap.add_argument("--out", type=Path, default=None, help="default: DIAG_DIR/overhead.jsonl")
    args = ap.parse_args()
    args.out = args.out or args.diag_dir / "overhead.jsonl"
    raw = args.diag_dir / "raw" / args.label
    raw.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    runs = 1 + args.warmups + args.repeats
    tri_env = dict(os.environ, TRITON_CACHE_DIR=args.triton_cache_dir)
    cu_env = dict(os.environ, ZKDUEL_REPORT_OCCUPANCY="1")
    cu_env.update(kv.split("=", 1) for kv in args.cuda_env)

    mismatches: list[str] = []
    with args.out.open("a") as fout:
        for w in args.workloads:
            for bw in args.bit_widths:
                for r in args.rounds:
                    t0 = time.time()
                    tag = f"{w}_bw{bw}_r{r}_{args.point_mode}_{args.layout}"
                    common = [
                        "--workloads",
                        w,
                        "--bit-widths",
                        str(bw),
                        "--rounds",
                        str(r),
                        "--warmups",
                        str(args.warmups),
                        "--repeats",
                        str(args.repeats),
                        "--point-mode",
                        args.point_mode,
                        "--layout",
                        args.layout,
                    ]
                    sides = [
                        (
                            "cuda",
                            [args.cuda_binary, *common, *args.cuda_args.split()],
                            CUDA_KERNELS,
                            cu_env,
                        )
                    ]
                    if not args.skip_triton:
                        sides.insert(
                            0,
                            (
                                "triton",
                                [sys.executable, "-u", args.triton_script, *common],
                                TRITON_KERNELS,
                                tri_env,
                            ),
                        )
                    rec: dict[str, object] = {
                        "label": args.label,
                        "workload": w,
                        "bit_width": bw,
                        "rounds": r,
                        "point_mode": args.point_mode,
                        "runs": runs,
                        "cuda_env": args.cuda_env,
                        "layout": args.layout,
                        "cuda_args": args.cuda_args,
                    }
                    if not args.skip_triton and (w, bw) in DEFAULT_SKIPS:
                        rec["skipped"] = DEFAULT_SKIPS[(w, bw)]
                        fout.write(json.dumps(rec) + "\n")
                        fout.flush()
                        continue
                    try:
                        for side, cmd, names, env in sides:
                            sh(
                                cmd + ["--out", str(raw / f"{tag}_{side}.csv")],
                                raw / f"{tag}_{side}.log",
                                env,
                            )
                            row = first_row(raw / f"{tag}_{side}.csv")
                            rec[f"{side}_wall_ms"] = float(row["median_ms"])
                            rec[f"{side}_checksum"] = row["checksum"]
                            rep = raw / f"{tag}_{side}.nsys-rep"
                            sh(
                                [
                                    "nsys",
                                    "profile",
                                    "--trace=cuda",
                                    "--sample=none",
                                    "--cpuctxsw=none",
                                    "--force-overwrite=true",
                                    "-o",
                                    str(rep.with_suffix("")),
                                    *cmd,
                                    "--out",
                                    str(raw / f"{tag}_{side}_nsys.csv"),
                                ],
                                raw / f"{tag}_{side}_nsys.log",
                                env,
                            )
                            st = kernel_stats(rep, names)
                            rec[f"{side}_gpu_ms"] = st["ns"] / 1e6 / runs
                            rec[f"{side}_eval_ms"] = st["eval_ns"] / 1e6 / runs
                            rec[f"{side}_launches_per_run"] = st["launches"] / runs
                            rec[f"{side}_regs"] = st["regs"]
                            rec[f"{side}_host_ms"] = rec[f"{side}_wall_ms"] - rec[f"{side}_gpu_ms"]
                            rep.unlink()  # keep the small CSV summaries, drop the multi-MB report
                        rec["cuda_blocks_per_sm"] = occupancy(raw / f"{tag}_cuda.log")
                        if not args.skip_triton:
                            rec["checksum_match"] = rec["triton_checksum"] == rec["cuda_checksum"]
                            rec["cuda_speedup_wall"] = rec["triton_wall_ms"] / rec["cuda_wall_ms"]
                            rec["cuda_speedup_gpu_only"] = rec["triton_gpu_ms"] / rec["cuda_gpu_ms"]
                    except (subprocess.CalledProcessError, StopIteration, OSError) as exc:
                        rec["error"] = f"{type(exc).__name__}: {exc}"[:300]
                    rec["seconds"] = round(time.time() - t0, 1)
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                    print(
                        json.dumps(
                            {
                                k: (round(v, 3) if isinstance(v, float) else v)
                                for k, v in rec.items()
                                if not k.endswith("checksum")
                            }
                        ),
                        flush=True,
                    )
                    if rec.get("checksum_match") is False:
                        mismatches.append(tag)
                        print(
                            f"WARNING: CHECKSUM MISMATCH {tag}: triton={rec['triton_checksum']} "
                            f"cuda={rec['cuda_checksum']}; not a validated result",
                            flush=True,
                        )
    if mismatches:
        print(
            f"WARNING: {len(mismatches)} configuration(s) with mismatching Triton/CUDA checksums "
            f"(checksum_match=false in {args.out}): {', '.join(mismatches)}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    os.chdir(ROOT)
    raise SystemExit(main())
