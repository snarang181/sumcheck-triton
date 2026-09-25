from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

POLY_WORKLOADS = [
    "poly_a",
    "poly_ab",
    "poly_ab_plus_c",
    "poly_abc",
    "poly_aabbc",
    "poly_abc_plus_de",
    "poly_abcg_plus_deg",
]


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_by_key(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {(row["workload"], int(row["rounds"])): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run matched device-resident Triton and native CUDA SumCheck ladders."
    )
    parser.add_argument("--workloads", nargs="+", default=POLY_WORKLOADS)
    parser.add_argument("--rounds", nargs="+", type=int, default=[8, 10, 12, 14, 16, 18, 20, 22])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cuda-binary", default="build/u256_sumcheck_cuda")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--triton-out", type=Path, default=Path("results/triton_u256_matched.csv"))
    parser.add_argument("--cuda-out", type=Path, default=Path("results/cuda_u256_matched.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/triton_cuda_u256_compare.csv"))
    args = parser.parse_args()

    unknown = sorted(set(args.workloads) - set(POLY_WORKLOADS))
    if unknown:
        raise SystemExit(f"unknown workloads: {unknown}")

    args.triton_out.parent.mkdir(parents=True, exist_ok=True)
    args.cuda_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        run(["cuda/build/build_u256_sumcheck.sh"])

    run(
        [
            args.python,
            "-u",
            "benchmarks/u256/device_resident_sweep.py",
            "--schedule",
            "manual",
            "--workloads",
            *args.workloads,
            "--rounds",
            *[str(x) for x in args.rounds],
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--table-mode",
            "matched",
            "--challenge-mode",
            "ones",
            "--out",
            str(args.triton_out),
        ]
    )

    run(
        [
            args.cuda_binary,
            "--workloads",
            *args.workloads,
            "--rounds",
            *[str(x) for x in args.rounds],
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--block",
            str(args.block),
            "--out",
            str(args.cuda_out),
        ]
    )

    triton_rows = read_by_key(args.triton_out)
    cuda_rows = read_by_key(args.cuda_out)
    keys = sorted(set(triton_rows) | set(cuda_rows), key=lambda x: (x[0], x[1]))

    fieldnames = [
        "workload",
        "polynomial",
        "rounds",
        "N",
        "vars",
        "degree",
        "terms",
        "warmups",
        "repeats",
        "table_mode",
        "challenge_mode",
        "triton_backend",
        "triton_median_ms",
        "triton_status",
        "cuda_backend",
        "cuda_median_ms",
        "cuda_status",
        "cuda_checksum",
        "cuda_speedup_vs_triton",
    ]

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in keys:
            t = triton_rows.get(key, {})
            c = cuda_rows.get(key, {})
            triton_ms = t.get("median_ms", "")
            cuda_ms = c.get("median_ms", "")
            speedup = ""
            if triton_ms and cuda_ms and t.get("status") == "ok" and c.get("status") == "ok":
                speedup = f"{float(triton_ms) / float(cuda_ms):.6f}"
            row = {
                "workload": key[0],
                "polynomial": t.get("polynomial") or c.get("polynomial", ""),
                "rounds": key[1],
                "N": t.get("N") or c.get("N", ""),
                "vars": t.get("vars") or c.get("vars", ""),
                "degree": t.get("degree") or c.get("degree", ""),
                "terms": t.get("terms") or c.get("terms", ""),
                "warmups": args.warmups,
                "repeats": args.repeats,
                "table_mode": "matched",
                "challenge_mode": "ones",
                "triton_backend": t.get("backend", ""),
                "triton_median_ms": triton_ms,
                "triton_status": t.get("status", "missing"),
                "cuda_backend": c.get("backend", ""),
                "cuda_median_ms": cuda_ms,
                "cuda_status": c.get("status", "missing"),
                "cuda_checksum": c.get("checksum", ""),
                "cuda_speedup_vs_triton": speedup,
            }
            writer.writerow(row)
            print(
                f"{row['workload']:24s} rounds={row['rounds']:<2d} "
                f"triton_ms={triton_ms} cuda_ms={cuda_ms} speedup={speedup}",
                flush=True,
            )

    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
