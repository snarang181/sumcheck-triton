"""Check the CUDA field-sweep binary against the canonical registry for all 32 workloads.

Runs build/field_sumcheck_cuda at small sizes and compares its checksums with the
reference checksums computed by validate_vs_registry.py from src/zkduel/workloads.py
(read from experiments/h100/validation/*.jsonl). The binary's checksum covers the first
run plus warmups plus repeats, the same convention the Triton validator uses (warmups=0,
repeats=1 -> 2x a single run), so the numbers are directly comparable.

Usage: python experiments/h100/tools/validate_cuda_vs_registry.py [--point-mode specialized]
       [--binary BIN] [--validation-dir DIR]   (DIR: where validate_vs_registry.py wrote its
       reference JSONL files; default experiments/h100/validation)
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
VAL = ROOT / "experiments" / "h100" / "validation"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", type=Path, default=ROOT / "build" / "field_sumcheck_cuda")
    ap.add_argument("--point-mode", default="specialized")
    ap.add_argument("--layout", default="element")
    ap.add_argument("--workloads", nargs="*", default=None, help="restrict to these workloads")
    ap.add_argument(
        "--validation-dir",
        type=Path,
        default=VAL,
        help="directory of validate_vs_registry.py JSONL files with the reference checksums",
    )
    ap.add_argument(
        "--out", type=Path, default=None, help="default: VALIDATION_DIR/cuda_vs_registry.jsonl"
    )
    args = ap.parse_args()
    args.out = args.out or args.validation_dir / "cuda_vs_registry.jsonl"

    ref: dict[tuple, int] = {}
    for f in args.validation_dir.glob("*.jsonl"):
        if f.name == args.out.name or f.name.startswith("cuda_"):  # earlier CUDA results
            continue
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("reference_checksum") is None:
                continue
            if r.get("point_mode") != args.point_mode or r.get("layout") != args.layout:
                continue
            ref[(r["workload"], r["bit_width"], r["rounds"], r["challenge_mode"])] = r[
                "reference_checksum"
            ]
    workloads = sorted({k[0] for k in ref if not args.workloads or k[0] in args.workloads})
    ref = {k: v for k, v in ref.items() if k[0] in workloads}
    results = []
    for cm in ("fixed", "sha3"):
        rounds = sorted({k[2] for k in ref if k[3] == cm})
        widths = sorted({k[1] for k in ref if k[3] == cm})
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "cuda.csv"
            subprocess.run(
                [
                    str(args.binary),
                    "--workloads",
                    *workloads,
                    "--bit-widths",
                    *map(str, widths),
                    "--rounds",
                    *map(str, rounds),
                    "--warmups",
                    "0",
                    "--repeats",
                    "1",
                    "--challenge-mode",
                    cm,
                    "--point-mode",
                    args.point_mode,
                    "--layout",
                    args.layout,
                    "--out",
                    str(out),
                ],
                check=True,
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
            )
            for row in csv.DictReader(out.open()):
                key = (row["workload"], int(row["bit_width"]), int(row["rounds"]), cm)
                if key in ref:
                    results.append(
                        {
                            "workload": key[0],
                            "bit_width": key[1],
                            "rounds": key[2],
                            "challenge_mode": cm,
                            "point_mode": args.point_mode,
                            "cuda_checksum": int(row["checksum"]),
                            "reference_checksum": ref[key],
                            "match": int(row["checksum"]) == ref[key],
                        }
                    )
    with args.out.open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    bad = [r for r in results if not r["match"]]
    print(
        f"compared {len(results)} cases over {len({r['workload'] for r in results})} workloads; "
        f"mismatches: {len(bad)}"
    )
    for r in bad:
        print("MISMATCH", r)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
