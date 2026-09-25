"""Recompute the paper's Table 2 and Table 4 from a joined resilient-sweep CSV.

Works on the A100 CSV (triton_cuda_field_resilient_r14_28_new.csv) and on this run's
experiments/h100/sweep/triton_cuda_field_resilient_r14_28_h100.csv, so the two can be
compared with identical definitions:

  validated  = both statuses ok and checksum_match == true
  speedup    = triton_median_ms / cuda_median_ms   (CUDA advantage; < 1 means Triton wins)
  best ratio = min speedup;  wins = speedup < 1;  within 1.1x = speedup < 1.1 (wins included)

Usage: python experiments/h100/tools/summarize_sweep.py CSV [CSV ...] [--label NAME] [--md OUT.md]

Several CSVs (e.g. the fixed-harness sweep's zk_32_128.csv and zk_256.csv) are summarized as one.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

FAMILIES = {
    "Verifiable / Spartan": ["zk_verifiable_asics", "zk_spartan_1", "zk_spartan_2"],
    "Halo2 curve checks": ["zk_witness_non_id", "zk_witness_id_point_1", "zk_witness_id_point_2"],
    "Incomplete addition": ["zk_incomplete_add_1", "zk_incomplete_add_2"],
    "Complete addition": [f"zk_complete_add_{i}" for i in range(1, 13)],
    "HyperPlonk vanilla": ["zk_vanilla_zerocheck_hp", "zk_vanilla_permcheck_hp"],
    "HyperPlonk Jellyfish": ["zk_jellyfish_zerocheck_hp", "zk_jellyfish_permcheck_hp"],
    "Opening check": ["zk_opencheck"],
}
# zkPHIRE ASIC column of the paper's Table 4 (ms), quoted for side-by-side context only.
ASIC_MS = {
    "Verifiable / Spartan": 6.31,
    "Halo2 curve checks": 8.86,
    "Incomplete addition": 23.06,
    "Complete addition": 19.95,
    "HyperPlonk vanilla": 19.81,
    "HyperPlonk Jellyfish": 30.09,
    "Opening check": 21.29,
}


def gmean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else math.nan


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def validated(row: dict[str, str]) -> bool:
    return (
        row.get("triton_status") == "ok"
        and row.get("cuda_status") == "ok"
        and row.get("checksum_match", "").lower() == "true"
    )


def table2(rows: list[dict[str, str]]) -> list[str]:
    by_bw: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if validated(r):
            by_bw[int(r["bit_width"])].append(
                float(r["triton_median_ms"]) / float(r["cuda_median_ms"])
            )
    out = [
        "| Width | Validated | Median | Best ratio | Wins | Within 1.1x |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    allv: list[float] = []
    for bw in sorted(by_bw):
        v = by_bw[bw]
        allv += v
        out.append(
            f"| {bw} | {len(v)} | {statistics.median(v):.2f}x | {min(v):.3f} | "
            f"{sum(x < 1 for x in v)} | {sum(x < 1.1 for x in v)} |"
        )
    if allv:
        out.append(
            f"| All | {len(allv)} | {statistics.median(allv):.2f}x | {min(allv):.3f} | "
            f"{sum(x < 1 for x in allv)} | {sum(x < 1.1 for x in allv)} |"
        )
    return out


def table4(rows: list[dict[str, str]], bit_width: int = 256, rounds: int = 24) -> list[str]:
    sel = {
        r["workload"]: r
        for r in rows
        if validated(r) and int(r["bit_width"]) == bit_width and int(r["rounds"]) == rounds
    }
    out = [
        f"| Family ({bit_width}-bit, r={rounds}) | n | Triton ms | CUDA ms | Speedup | ASIC ms (paper) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    tri_all, cu_all = [], []
    for fam, names in FAMILIES.items():
        got = [sel[n] for n in names if n in sel]
        t = [float(r["triton_median_ms"]) for r in got]
        c = [float(r["cuda_median_ms"]) for r in got]
        tri_all += t
        cu_all += c
        if got:
            out.append(
                f"| {fam} | {len(got)} | {gmean(t):.2f} | {gmean(c):.2f} | "
                f"{gmean(t) / gmean(c):.2f}x | {ASIC_MS[fam]:.2f} |"
            )
        else:
            out.append(f"| {fam} | 0 | - | - | - | {ASIC_MS[fam]:.2f} |")
    if tri_all:
        out.append(
            f"| All matched | {len(tri_all)} | {gmean(tri_all):.2f} | {gmean(cu_all):.2f} | "
            f"{gmean(tri_all) / gmean(cu_all):.2f}x | - |"
        )
    return out


def counts(rows: list[dict[str, str]]) -> list[str]:
    n = len(rows)
    ok = sum(validated(r) for r in rows)
    mismatch = sum(r.get("checksum_match", "").lower() == "false" for r in rows)
    skipped = sum(
        r.get("triton_status") == "skipped" or r.get("cuda_status") == "skipped" for r in rows
    )
    return [f"rows={n} validated={ok} checksum_mismatch={mismatch} skipped={skipped}"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path, nargs="+")
    ap.add_argument("--label", default=None)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()
    rows = [r for p in args.csv for r in load(p)]
    lines = [
        f"## {args.label or ' + '.join(p.name for p in args.csv)}",
        "",
        *counts(rows),
        "",
        "### Table 2",
        "",
        *table2(rows),
        "",
        "### Table 4",
        "",
        *table4(rows),
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.md:
        args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
