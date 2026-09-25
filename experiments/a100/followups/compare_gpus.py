"""A100 vs H100 tables from the follow-up diagnostics (run by run_a100_followups.sh).

  python experiments/a100/followups/compare_gpus.py > experiments/a100/followups/COMPARISON.md

Only configurations measured on both GPUs, with agreeing Triton and CUDA checksums on each,
are compared. Excluded configurations are named in the output.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = Path(os.environ.get("A100_DIAG_DIR", ROOT / "experiments" / "a100" / "followups" / "diag"))
H = ROOT / "experiments" / "h100" / "diag"


def load(p: Path) -> dict[tuple, dict]:
    out = {}
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            if "cuda_speedup_gpu_only" in r:
                out[(r["workload"], r["bit_width"], r["rounds"])] = r
    return out


def agree(r: dict) -> bool:
    """A record whose Triton and CUDA checksums agree."""
    return r.get("checksum_match") is True


def warn_excluded(keys: list[tuple], what: str) -> None:
    if keys:
        names = ", ".join(f"{w} {bw}-bit r={rr}" for w, bw, rr in sorted(keys))
        print(
            f"\n**Excluded {len(keys)} configuration(s) from {what} because the Triton and CUDA "
            f"checksums disagree:** {names}."
        )


def gmean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def split_table(title: str, a: dict, h: dict) -> None:
    print(f"\n## {title}\n")
    common_all = sorted(set(a) & set(h))
    if not common_all:
        print("(no configurations measured on both GPUs yet)")
        return
    common = [k for k in common_all if agree(a[k]) and agree(h[k])]
    g = defaultdict(list)
    for k in common:
        g[(k[1], k[2])].append(k)
    print(
        "| Width | r | n | Wall lead A100 | Wall lead H100 | Kernel-only lead A100 | Kernel-only lead H100 "
        "| GPU idle share of Triton wall A100 / H100 | Triton host us/launch A100 / H100 |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for bw, rr in sorted(g):
        ks = g[(bw, rr)]

        def med(d, f):
            return statistics.median(f(d[k]) for k in ks)

        print(
            f"| {bw} | {rr} | {len(ks)} "
            f"| {med(a, lambda r: r['cuda_speedup_wall']):.2f}x | {med(h, lambda r: r['cuda_speedup_wall']):.2f}x "
            f"| {med(a, lambda r: r['cuda_speedup_gpu_only']):.2f}x | {med(h, lambda r: r['cuda_speedup_gpu_only']):.2f}x "
            f"| {med(a, lambda r: r['triton_host_ms'] / r['triton_wall_ms']):.0%} / "
            f"{med(h, lambda r: r['triton_host_ms'] / r['triton_wall_ms']):.0%} "
            f"| {med(a, lambda r: 1e3 * r['triton_host_ms'] / r['triton_launches_per_run']):.1f} / "
            f"{med(h, lambda r: 1e3 * r['triton_host_ms'] / r['triton_launches_per_run']):.1f} |"
        )
    print(f"\nChecksums agree (A100 records): {all(agree(a[k]) for k in common_all)}.")
    warn_excluded([k for k in common_all if k not in common], "this table")


def main() -> int:
    ab, at = load(A / "base_all.jsonl"), load(A / "treduce_all.jsonl")
    hb, ht = load(H / "baseline_all.jsonl"), load(H / "treduce_all.jsonl")
    print("# A100 vs H100 follow-ups\n")
    print(
        "Leads are CUDA's (Triton time / CUDA time); kernel-only uses summed nsys kernel durations."
    )
    split_table("1. Original Triton: wall clock vs GPU-kernel time", ab, hb)
    split_table("2. Triton with tl.reduce: wall clock vs GPU-kernel time", at, ht)

    print("\n## 3. Effect of tl.reduce on the kernel-only CUDA lead\n")
    print(
        "| GPU | Width | r | n | Kernel-only lead original -> tl.reduce | Eval-kernel speedup (geomean) |"
    )
    print("|---|---:|---:|---:|---:|---:|")
    excluded3 = []
    for gpu, b, t in (("A100", ab, at), ("H100", hb, ht)):
        g = defaultdict(list)
        for k in set(b) & set(t) & (set(ab) | set(at)):
            if not (agree(b[k]) and agree(t[k])):
                excluded3.append(k)
                continue
            g[(k[1], k[2])].append(k)
        for bw, rr in sorted(g):
            ks = g[(bw, rr)]
            print(
                f"| {gpu} | {bw} | {rr} | {len(ks)} "
                f"| {statistics.median(b[k]['cuda_speedup_gpu_only'] for k in ks):.2f}x -> "
                f"{statistics.median(t[k]['cuda_speedup_gpu_only'] for k in ks):.2f}x "
                f"| {gmean([b[k]['triton_eval_ms'] / t[k]['triton_eval_ms'] for k in ks]):.2f}x |"
            )
    warn_excluded(sorted(set(excluded3)), "this table")

    print("\n## 4. CUDA occupancy ablation on A100 (eval-kernel time vs achieved blocks/SM)\n")
    occ = defaultdict(list)
    p = A / "occupancy.jsonl"
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            if "error" not in r and r.get("cuda_blocks_per_sm"):
                occ[(r["workload"], r["bit_width"], r["rounds"])].append(r)

    def blocks(regs: int) -> int:  # 128-thread CTAs, 8-register granularity, 64K registers/SM
        return min(16, 65536 // (((regs + 7) // 8) * 8 * 128))

    print(
        "| Workload | Width | r | Triton eval ms (blocks/SM) | CUDA eval ms own occupancy (blocks/SM) "
        "| CUDA eval ms at <= Triton's occupancy (blocks/SM) | Occupancy cost | Eval gap |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in sorted(occ):
        if k not in ab or not agree(ab[k]):
            continue
        t = ab[k]
        tb = blocks(max(t["triton_regs"]["_round_eval_specialized_kernel"]))
        own = min(occ[k], key=lambda r: r["cuda_env"] != ["ZKDUEL_EVAL_EXTRA_SMEM=0"])
        cands = [r for r in occ[k] if r["cuda_blocks_per_sm"][1] <= tb] or occ[k]
        at_ = max(cands, key=lambda r: r["cuda_blocks_per_sm"][1])
        print(
            f"| {k[0]} | {k[1]} | {k[2]} | {t['triton_eval_ms']:.2f} ({tb}) | {own['cuda_eval_ms']:.2f} "
            f"({own['cuda_blocks_per_sm'][1]}) | {at_['cuda_eval_ms']:.2f} ({at_['cuda_blocks_per_sm'][1]}) "
            f"| {at_['cuda_eval_ms'] / own['cuda_eval_ms']:.2f}x | {t['triton_eval_ms'] / own['cuda_eval_ms']:.2f}x |"
        )

    print("\n## 5. r=28 cases with more than 2^31 table elements, fixed source, A100\n")
    print("| Workload | Width | Checksums agree | Wall lead | Kernel-only lead | Error |")
    print("|---|---:|---|---:|---:|---|")
    p = A / "r28_fixed.jsonl"
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            if "error" in r:
                print(f"| {r['workload']} | {r['bit_width']} | - | - | - | {r['error'][:60]} |")
            elif not agree(r):
                print(f"| {r['workload']} | {r['bit_width']} | {r['checksum_match']} | - | - | |")
            else:
                print(
                    f"| {r['workload']} | {r['bit_width']} | {r['checksum_match']} | {r['cuda_speedup_wall']:.2f}x "
                    f"| {r['cuda_speedup_gpu_only']:.2f}x | |"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
