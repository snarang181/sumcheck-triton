#!/usr/bin/env python3
"""Correctness checker for native CUDA prime-field SumCheck ladder.

The CUDA benchmark keeps all table values and round sums in Montgomery form.
This checker mirrors that convention with Python big ints and compares the
reported limb checksum from the standalone `.cu` executable.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import importlib.util  # noqa: E402

_fields_spec = importlib.util.spec_from_file_location("zkduel_fields", SRC / "zkduel" / "fields.py")
if _fields_spec is None or _fields_spec.loader is None:
    raise RuntimeError("failed to load zkduel.fields")
_fields = importlib.util.module_from_spec(_fields_spec)
sys.modules[_fields_spec.name] = _fields
_fields_spec.loader.exec_module(_fields)
FIELD_SPECS = _fields.FIELD_SPECS
FieldSpec = _fields.FieldSpec
get_field = _fields.get_field


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
    Workload("zk_verifiable_asics", "qadd*a + qadd*b + qmul*a*b", 8, 4, 3, 3),
    Workload("zk_spartan_1", "A*B*fz - C*fz", 9, 4, 3, 2),
    Workload("zk_spartan_2", "ABC*Z", 10, 2, 2, 1),
    Workload("zk_witness_non_id", "q*y*y - q*x*x*x - 5*q", 11, 3, 4, 3),
    Workload(
        "zk_complete_add_1",
        "q*xq*xq*lambda - 2*q*xq*xp*lambda + q*xp*xp*lambda "
        "- q*xq*yq + q*xq*yp + q*xp*yq - q*xp*yp",
        12,
        6,
        4,
        7,
    ),
    Workload("zk_complete_add_7", "q*xr - q*xq - q*xp*xr*beta + q*xp*xq*beta", 13, 5, 4, 4),
    Workload("zk_complete_add_8", "q*yr - q*yq - q*xp*yr*beta + q*xp*yq*beta", 14, 5, 4, 4),
]
WORKLOAD_BY_NAME = {w.name: w for w in WORKLOADS}


def initial_value(var: int, off: int, seed: int, field: FieldSpec) -> int:
    x = (seed & 0xFFFFFFFF) + 7919 * (var + 1) + 104729 * (off + 1)
    if field.bit_width == 32:
        return x % field.modulus
    return x


def limb_checksum(value: int, field: FieldSpec) -> int:
    return sum((i + 1) * ((value >> (32 * i)) & 0xFFFFFFFF) for i in range(field.limbs))


def mont_mul(a: int, b: int, field: FieldSpec) -> int:
    r_inv = pow(field.r_mod, -1, field.modulus)
    return (a * b * r_inv) % field.modulus


def eval_workload(workload: Workload, values: list[int], field: FieldSpec) -> int:
    if workload.ident == 1:
        return values[0]
    if workload.ident == 2:
        return mont_mul(values[0], values[1], field)
    if workload.ident == 3:
        return (mont_mul(values[0], values[1], field) + values[2]) % field.modulus
    if workload.ident == 4:
        return mont_mul(mont_mul(values[0], values[1], field), values[2], field)
    if workload.ident == 5:
        aa = mont_mul(values[0], values[0], field)
        bb = mont_mul(values[1], values[1], field)
        return mont_mul(mont_mul(aa, bb, field), values[2], field)
    if workload.ident == 6:
        abc = mont_mul(mont_mul(values[0], values[1], field), values[2], field)
        de = mont_mul(values[3], values[4], field)
        return (abc + de) % field.modulus
    if workload.ident == 7:
        abcg = mont_mul(
            mont_mul(mont_mul(values[0], values[1], field), values[2], field),
            values[3],
            field,
        )
        deg = mont_mul(mont_mul(values[4], values[5], field), values[3], field)
        return (abcg + deg) % field.modulus
    if workload.ident == 8:
        qadd_a = mont_mul(values[0], values[1], field)
        qadd_b = mont_mul(values[0], values[2], field)
        qmul_ab = mont_mul(mont_mul(values[3], values[1], field), values[2], field)
        return (qadd_a + qadd_b + qmul_ab) % field.modulus
    if workload.ident == 9:
        abfz = mont_mul(mont_mul(values[0], values[1], field), values[2], field)
        cfz = mont_mul(values[3], values[2], field)
        return (abfz - cfz) % field.modulus
    if workload.ident == 10:
        return mont_mul(values[0], values[1], field)
    if workload.ident == 11:
        qyy = mont_mul(mont_mul(values[0], values[1], field), values[1], field)
        qxxx = mont_mul(
            mont_mul(mont_mul(values[0], values[2], field), values[2], field), values[2], field
        )
        return (qyy - qxxx - 5 * values[0]) % field.modulus
    if workload.ident == 12:
        q_xq_xq_l = mont_mul(
            mont_mul(mont_mul(values[0], values[1], field), values[1], field), values[3], field
        )
        q_xq_xp_l = mont_mul(
            mont_mul(mont_mul(values[0], values[1], field), values[2], field), values[3], field
        )
        q_xp_xp_l = mont_mul(
            mont_mul(mont_mul(values[0], values[2], field), values[2], field), values[3], field
        )
        q_xq_yq = mont_mul(mont_mul(values[0], values[1], field), values[4], field)
        q_xq_yp = mont_mul(mont_mul(values[0], values[1], field), values[5], field)
        q_xp_yq = mont_mul(mont_mul(values[0], values[2], field), values[4], field)
        q_xp_yp = mont_mul(mont_mul(values[0], values[2], field), values[5], field)
        return (
            q_xq_xq_l - 2 * q_xq_xp_l + q_xp_xp_l - q_xq_yq + q_xq_yp + q_xp_yq - q_xp_yp
        ) % field.modulus
    if workload.ident == 13:
        q_xr = mont_mul(values[0], values[1], field)
        q_xq = mont_mul(values[0], values[2], field)
        q_xp_xr_beta = mont_mul(
            mont_mul(mont_mul(values[0], values[3], field), values[1], field), values[4], field
        )
        q_xp_xq_beta = mont_mul(
            mont_mul(mont_mul(values[0], values[3], field), values[2], field), values[4], field
        )
        return (q_xr - q_xq - q_xp_xr_beta + q_xp_xq_beta) % field.modulus
    q_yr = mont_mul(values[0], values[1], field)
    q_yq = mont_mul(values[0], values[2], field)
    q_xp_yr_beta = mont_mul(
        mont_mul(mont_mul(values[0], values[3], field), values[1], field), values[4], field
    )
    q_xp_yq_beta = mont_mul(
        mont_mul(mont_mul(values[0], values[3], field), values[2], field), values[4], field
    )
    return (q_yr - q_yq - q_xp_yr_beta + q_xp_yq_beta) % field.modulus


def reference_checksum(workload: Workload, rounds: int, seed: int, field: FieldSpec) -> int:
    cur_n = 1 << rounds
    tables = [
        [field.montgomery_encode(initial_value(var, off, seed, field)) for off in range(cur_n)]
        for var in range(workload.vars)
    ]
    points = [field.montgomery_encode(i) for i in range(workload.degree + 1)]
    r_one = field.montgomery_encode(1)

    checksum = 0
    for round_idx in range(rounds):
        del round_idx
        half = cur_n // 2
        for point in points:
            total = 0
            for off in range(half):
                values = [0] * 6
                for var in range(workload.vars):
                    even = tables[var][2 * off]
                    odd = tables[var][2 * off + 1]
                    diff = (odd - even) % field.modulus
                    values[var] = (even + mont_mul(diff, point, field)) % field.modulus
                total = (total + eval_workload(workload, values, field)) % field.modulus
            checksum += limb_checksum(total, field)

        if cur_n > 2:
            next_tables: list[list[int]] = []
            for var in range(workload.vars):
                row = []
                for off in range(half):
                    even = tables[var][2 * off]
                    odd = tables[var][2 * off + 1]
                    diff = (odd - even) % field.modulus
                    row.append((even + mont_mul(diff, r_one, field)) % field.modulus)
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
        "--bit-widths",
        *[str(b) for b in args.bit_widths],
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
    parser.add_argument("--binary", type=Path, default=Path("build/field_sumcheck_cuda"))
    parser.add_argument("--workloads", nargs="+", default=[w.name for w in WORKLOADS])
    parser.add_argument("--rounds", nargs="+", type=int, default=[2, 3])
    parser.add_argument("--bit-widths", nargs="+", type=int, default=sorted(FIELD_SPECS))
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument(
        "--out", type=Path, default=Path("results/cuda_sumcheck_field_correctness.csv")
    )
    parser.add_argument(
        "--raw-out", type=Path, default=Path("results/cuda_sumcheck_field_correctness_raw.csv")
    )
    args = parser.parse_args()

    unknown = sorted(set(args.workloads) - set(WORKLOAD_BY_NAME))
    if unknown:
        raise SystemExit(f"unknown workloads: {unknown}")
    for bit_width in args.bit_widths:
        get_field(bit_width)
    if not args.binary.exists():
        raise SystemExit(
            f"missing CUDA binary: {args.binary}; run cuda/build/build_field_sumcheck.sh"
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.raw_out.parent.mkdir(parents=True, exist_ok=True)
    run_cuda(args, args.raw_out)

    rows = []
    ok = True
    for row in csv.DictReader(args.raw_out.open()):
        workload = WORKLOAD_BY_NAME[row["workload"]]
        field = get_field(int(row["bit_width"]))
        rounds = int(row["rounds"])
        cuda_checksum = int(row["checksum"])
        runs_reported = 1 + int(row["warmups"]) + int(row["repeats"])
        ref_checksum = (
            reference_checksum(workload, rounds, args.seed + rounds, field) * runs_reported
        )
        status = "ok" if cuda_checksum == ref_checksum else "mismatch"
        ok = ok and status == "ok"
        rows.append(
            {
                "workload": workload.name,
                "polynomial": workload.polynomial,
                "field": field.name,
                "bit_width": field.bit_width,
                "rounds": rounds,
                "N": 1 << rounds,
                "cuda_checksum": cuda_checksum,
                "reference_checksum": ref_checksum,
                "status": status,
            }
        )
        print(
            f"{status:8s} workload={workload.name:20s} field={field.name:28s} "
            f"rounds={rounds:<2d} cuda={cuda_checksum} reference={ref_checksum}"
        )

    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(args.out)
    return 0 if ok else 1


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.exit(main())
