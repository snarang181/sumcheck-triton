#!/usr/bin/env python3
"""Independent correctness checker for the native CUDA SumCheck ladder.

This script runs the direct `.cu` executable and compares its checksum against a
Python big-int field/SumCheck reference. It is intentionally not a Python
binding: the CUDA path is exercised by launching the native binary.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PRIME_256 = int(
    "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    16,
)
R_MOD = (1 << 256) % PRIME_256
R_INV = pow(R_MOD, -1, PRIME_256)


@dataclass(frozen=True)
class Workload:
    name: str
    polynomial: str
    ident: int
    vars: int
    degree: int
    terms: int


WORKLOADS = [
    Workload("poly_a", "a", 1, 1, 1, 1),
    Workload("poly_ab", "a*b", 2, 2, 2, 1),
    Workload("poly_ab_plus_c", "a*b + c", 3, 3, 2, 2),
    Workload("poly_abc", "a*b*c", 4, 3, 3, 1),
    Workload("poly_aabbc", "a*a*b*b*c", 5, 3, 5, 1),
    Workload("poly_abc_plus_de", "a*b*c + d*e", 6, 5, 3, 2),
    Workload("poly_abcg_plus_deg", "a*b*c*g + d*e*g", 7, 6, 4, 2),
]
WORKLOAD_BY_NAME = {w.name: w for w in WORKLOADS}


def initial_limb0_value(var: int, off: int, seed: int) -> int:
    x = seed & 0xFFFFFFFF
    x = (x + 7919 * (var + 1) + 104729 * (off + 1)) & 0xFFFFFFFF
    return x


def add_mod(a: int, b: int) -> int:
    return (a + b) % PRIME_256


def sub_mod(a: int, b: int) -> int:
    return (a - b) % PRIME_256


def mont_mul(a: int, b: int) -> int:
    return (a * b * R_INV) % PRIME_256


def point_mont(point: int) -> int:
    return (point * R_MOD) % PRIME_256


def eval_workload(workload: Workload, p: list[int]) -> int:
    if workload.ident == 1:
        return p[0]
    if workload.ident == 2:
        return mont_mul(p[0], p[1])
    if workload.ident == 3:
        return add_mod(mont_mul(p[0], p[1]), p[2])
    if workload.ident == 4:
        return mont_mul(mont_mul(p[0], p[1]), p[2])
    if workload.ident == 5:
        aa = mont_mul(p[0], p[0])
        bb = mont_mul(p[1], p[1])
        return mont_mul(mont_mul(aa, bb), p[2])
    if workload.ident == 6:
        abc = mont_mul(mont_mul(p[0], p[1]), p[2])
        de = mont_mul(p[3], p[4])
        return add_mod(abc, de)
    abcg = mont_mul(mont_mul(mont_mul(p[0], p[1]), p[2]), p[3])
    deg = mont_mul(mont_mul(p[4], p[5]), p[3])
    return add_mod(abcg, deg)


def reference_checksum(workload: Workload, rounds: int, seed: int) -> int:
    cur_n = 1 << rounds
    tables = [
        [initial_limb0_value(var, off, seed) for off in range(cur_n)]
        for var in range(workload.vars)
    ]

    checksum = 0
    for round_idx in range(rounds):
        del round_idx
        half = cur_n // 2

        for point in range(workload.degree + 1):
            t = point_mont(point)
            total = 0
            for off in range(half):
                values = [0] * 6
                for var in range(workload.vars):
                    even = tables[var][2 * off]
                    odd = tables[var][2 * off + 1]
                    diff = sub_mod(odd, even)
                    values[var] = add_mod(even, mont_mul(diff, t))
                total = add_mod(total, eval_workload(workload, values))
            checksum += total & 0xFFFFFFFF

        if cur_n > 2:
            r = point_mont(1)
            next_tables: list[list[int]] = []
            for var in range(workload.vars):
                row = []
                for off in range(half):
                    even = tables[var][2 * off]
                    odd = tables[var][2 * off + 1]
                    diff = sub_mod(odd, even)
                    row.append(add_mod(even, mont_mul(diff, r)))
                next_tables.append(row)
            tables = next_tables
            cur_n = half

    return checksum


def run_cuda(args: argparse.Namespace, raw_out: Path) -> None:
    cmd = [
        str(args.binary),
        "--workloads",
        *args.workloads,
        "--rounds",
        *[str(r) for r in args.rounds],
        "--warmups",
        "0",
        "--repeats",
        "1",
        "--seed",
        str(args.seed),
        "--block",
        str(args.block),
        "--out",
        str(raw_out),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=Path("build/u256_sumcheck_cuda"))
    parser.add_argument("--workloads", nargs="+", default=[w.name for w in WORKLOADS])
    parser.add_argument("--rounds", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument(
        "--out", type=Path, default=Path("results/cuda_poly_ladder_correctness.csv")
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=Path("results/cuda_poly_ladder_correctness_raw.csv"),
    )
    args = parser.parse_args()

    unknown = sorted(set(args.workloads) - set(WORKLOAD_BY_NAME))
    if unknown:
        raise SystemExit(f"unknown workloads: {unknown}")
    if not args.binary.exists():
        raise SystemExit(
            f"missing CUDA binary: {args.binary}; run cuda/build/build_u256_sumcheck.sh"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)

    run_cuda(args, args.raw_out)

    cuda_rows = list(csv.DictReader(args.raw_out.open()))
    rows = []
    ok = True
    for row in cuda_rows:
        workload = WORKLOAD_BY_NAME[row["workload"]]
        rounds = int(row["rounds"])
        cuda_checksum = int(row["checksum"])
        ref_checksum = reference_checksum(workload, rounds, args.seed + rounds)
        status = "ok" if cuda_checksum == ref_checksum else "mismatch"
        if status != "ok":
            ok = False
        rows.append(
            {
                "workload": workload.name,
                "polynomial": workload.polynomial,
                "rounds": rounds,
                "N": 1 << rounds,
                "cuda_checksum": cuda_checksum,
                "reference_checksum": ref_checksum,
                "status": status,
            }
        )
        print(
            f"{status:8s} workload={workload.name:20s} rounds={rounds:<2d} "
            f"cuda={cuda_checksum} reference={ref_checksum}"
        )

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "workload",
                "polynomial",
                "rounds",
                "N",
                "cuda_checksum",
                "reference_checksum",
                "status",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(args.out)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
