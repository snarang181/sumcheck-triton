"""Classify every non-validated case of a resilient-sweep CSV from its per-case log.

The resilient driver records only `returncode=N` or `timeout after Ns`. This reads each
case's log and labels it out-of-memory / compile error / timeout / checksum mismatch /
other, next to the paper's footprint formula 3 x vars x 2^r x limbs x 4 bytes, so the
capacity vs non-capacity split can be stated from evidence rather than from the formula.

Usage: python experiments/h100/tools/classify_skips.py CSV [--device-gb 80] [--md OUT.md]
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

PATTERNS = [
    (
        "out-of-memory",
        re.compile(
            r"OutOfMemoryError|out of memory|cudaErrorMemoryAllocation|"
            r"CUDA_ERROR_OUT_OF_MEMORY",
            re.I,
        ),
    ),
    ("timeout", re.compile(r"killed after \d+s timeout")),
    (
        "compile-error",
        re.compile(
            r"CompilationError|PTXASError|ptxas (fatal|error)|"
            r"LLVM ERROR|MLIR|out of resource",
            re.I,
        ),
    ),
    ("launch-error", re.compile(r"CUDA error:|illegal memory access|too many resources", re.I)),
]


def classify(row: dict[str, str], root: Path) -> tuple[str, str]:
    if row.get("checksum_match", "").lower() == "false":
        return "checksum-mismatch", ""
    log = Path(row.get("log_path") or "")
    if not log.is_absolute():
        log = root / log
    text = log.read_text(errors="replace") if log.exists() else ""
    for label, pat in PATTERNS:
        m = pat.search(text)
        if m:
            line = next((ln for ln in text.splitlines() if pat.search(ln)), m.group(0))
            return label, line.strip()[:160]
    return ("other" if text else "no-log"), (text.strip().splitlines() or [""])[-1][:160]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--root", type=Path, default=Path("."), help="base for relative log paths")
    ap.add_argument("--device-gb", type=float, default=80.0)
    ap.add_argument("--md", type=Path, default=None)
    args = ap.parse_args()
    with args.csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    out = [
        "| Workload | Width | r | Footprint GB | > device? | Class | Evidence |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    tally: Counter[tuple[str, int, bool]] = Counter()
    for r in rows:
        ok = (
            r.get("triton_status") == "ok"
            and r.get("cuda_status") == "ok"
            and r.get("checksum_match", "").lower() == "true"
        )
        if ok:
            continue
        limbs = int(r["bit_width"]) // 32
        gb = 3 * int(r["vars"]) * (1 << int(r["rounds"])) * limbs * 4 / 1e9
        over = gb > args.device_gb
        label, evidence = classify(r, args.root)
        tally[(label, int(r["bit_width"]), over)] += 1
        out.append(
            f"| {r['workload']} | {r['bit_width']} | {r['rounds']} | {gb:.1f} | "
            f"{'yes' if over else 'no'} | {label} | `{evidence}` |"
        )
    summary = ["| Class | Width | Footprint > device | Cases |", "|---|---:|---|---:|"]
    summary += [
        f"| {k[0]} | {k[1]} | {'yes' if k[2] else 'no'} | {v} |" for k, v in sorted(tally.items())
    ]
    text = "\n".join(
        ["### Non-validated cases by class", "", *summary, "", "### Per case", "", *out, ""]
    )
    print(text)
    if args.md:
        args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
