from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path

from workload_specs import (
    WORKLOAD_BY_NAME,
    WORKLOAD_NAMES,
    ZK_WORKLOAD_NAMES,
    canonicalize_workload_names,
    known_workload_names,
)

SUPPORTED_BIT_WIDTHS = [32, 64, 128, 256]


def run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def read_by_key(path: Path) -> dict[tuple[str, int, int], dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return {(row["workload"], int(row["rounds"]), int(row["bit_width"])): row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run matched full-SumCheck prime-field Montgomery sweeps for Triton and CUDA."
    )
    parser.add_argument("--workloads", nargs="+", default=None)
    parser.add_argument(
        "--zk-only",
        action="store_true",
        help="Restrict default workload set to the 25 ZK workloads (no effect if --workloads is given)",
    )
    parser.add_argument("--bit-widths", nargs="+", type=int, default=SUPPORTED_BIT_WIDTHS)
    parser.add_argument("--rounds", nargs="+", type=int, default=[8, 10, 12, 14, 16, 18, 20, 22])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timing-mode", choices=["whole-run", "round-sum"], default="whole-run")
    parser.add_argument("--challenge-mode", choices=["fixed", "sha3"], default="fixed")
    parser.add_argument("--layout", choices=["element", "limb"], default="element")
    parser.add_argument("--point-mode", choices=["generic", "specialized"], default="generic")
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--triton-script", default="benchmarks/field_sweep/triton_sweep.py")
    parser.add_argument("--cuda-binary", default="build/field_sumcheck_cuda")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--triton-out", type=Path, default=Path("results/triton_field_sweep.csv"))
    parser.add_argument("--cuda-out", type=Path, default=Path("results/cuda_field_sweep.csv"))
    parser.add_argument("--out", type=Path, default=Path("results/triton_cuda_field_compare.csv"))
    args = parser.parse_args()

    if args.workloads is None:
        args.workloads = list(ZK_WORKLOAD_NAMES) if args.zk_only else list(WORKLOAD_NAMES)
    args.workloads = canonicalize_workload_names(args.workloads)
    unknown = sorted(set(args.workloads) - set(WORKLOAD_BY_NAME))
    if unknown:
        known = ", ".join(known_workload_names())
        raise SystemExit(f"unknown workloads: {unknown}; known: {known}")
    unsupported = sorted(set(args.bit_widths) - set(SUPPORTED_BIT_WIDTHS))
    if unsupported:
        raise SystemExit(f"unsupported bit widths: {unsupported}")

    args.triton_out.parent.mkdir(parents=True, exist_ok=True)
    args.cuda_out.parent.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        run(["cuda/build/build_field_sumcheck.sh"])

    run(
        [
            args.python,
            "-u",
            args.triton_script,
            "--workloads",
            *args.workloads,
            "--rounds",
            *[str(x) for x in args.rounds],
            "--bit-widths",
            *[str(x) for x in args.bit_widths],
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--timing-mode",
            args.timing_mode,
            "--challenge-mode",
            args.challenge_mode,
            "--layout",
            args.layout,
            "--point-mode",
            args.point_mode,
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
            "--bit-widths",
            *[str(x) for x in args.bit_widths],
            "--warmups",
            str(args.warmups),
            "--repeats",
            str(args.repeats),
            "--seed",
            str(args.seed),
            "--timing-mode",
            args.timing_mode,
            "--challenge-mode",
            args.challenge_mode,
            "--layout",
            args.layout,
            "--point-mode",
            args.point_mode,
            "--block",
            str(args.block),
            "--out",
            str(args.cuda_out),
        ]
    )

    triton_rows = read_by_key(args.triton_out)
    cuda_rows = read_by_key(args.cuda_out)
    keys = sorted(set(triton_rows) | set(cuda_rows), key=lambda x: (x[0], x[1], x[2]))

    fieldnames = [
        "workload",
        "polynomial",
        "arithmetic",
        "layout",
        "point_mode",
        "field",
        "bit_width",
        "limbs",
        "modulus_hex",
        "rounds",
        "N",
        "vars",
        "degree",
        "terms",
        "warmups",
        "repeats",
        "timing_mode",
        "challenge_mode",
        "triton_backend",
        "triton_first_ms",
        "triton_median_ms",
        "triton_eval_median_ms",
        "triton_challenge_median_ms",
        "triton_fold_median_ms",
        "triton_total_median_ms",
        "triton_status",
        "cuda_backend",
        "cuda_first_ms",
        "cuda_median_ms",
        "cuda_eval_median_ms",
        "cuda_challenge_median_ms",
        "cuda_fold_median_ms",
        "cuda_total_median_ms",
        "cuda_status",
        "checksum_match",
        "triton_checksum",
        "cuda_checksum",
        "cuda_speedup_vs_triton",
    ]

    mismatches: list[str] = []
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for key in keys:
            t = triton_rows.get(key, {})
            c = cuda_rows.get(key, {})
            triton_ms = t.get("median_ms", "")
            cuda_ms = c.get("median_ms", "")
            triton_checksum = t.get("checksum", "")
            cuda_checksum = c.get("checksum", "")
            checksum_match = ""
            if triton_checksum and cuda_checksum:
                checksum_match = str(triton_checksum == cuda_checksum).lower()
            # A speedup is reported only for a validated configuration: both backends ok
            # and their checksums agree. The medians are kept either way.
            speedup = ""
            if (
                triton_ms
                and cuda_ms
                and t.get("status") == "ok"
                and c.get("status") == "ok"
                and checksum_match == "true"
            ):
                speedup = f"{float(triton_ms) / float(cuda_ms):.6f}"
            row = {
                "workload": key[0],
                "polynomial": t.get("polynomial") or c.get("polynomial", ""),
                "arithmetic": "prime-field-montgomery",
                "layout": t.get("layout") or c.get("layout", args.layout),
                "point_mode": t.get("point_mode") or "cuda-generic",
                "field": t.get("field") or c.get("field", ""),
                "bit_width": key[2],
                "limbs": t.get("limbs") or c.get("limbs", ""),
                "modulus_hex": t.get("modulus_hex") or c.get("modulus_hex", ""),
                "rounds": key[1],
                "N": t.get("N") or c.get("N", ""),
                "vars": t.get("vars") or c.get("vars", ""),
                "degree": t.get("degree") or c.get("degree", ""),
                "terms": t.get("terms") or c.get("terms", ""),
                "warmups": args.warmups,
                "repeats": args.repeats,
                "timing_mode": args.timing_mode,
                "challenge_mode": args.challenge_mode,
                "triton_backend": t.get("backend", ""),
                "triton_first_ms": t.get("first_ms", ""),
                "triton_median_ms": triton_ms,
                "triton_eval_median_ms": t.get("eval_median_ms", ""),
                "triton_challenge_median_ms": t.get("challenge_median_ms", ""),
                "triton_fold_median_ms": t.get("fold_median_ms", ""),
                "triton_total_median_ms": t.get("total_median_ms", triton_ms),
                "triton_status": t.get("status", "missing"),
                "cuda_backend": c.get("backend", ""),
                "cuda_first_ms": c.get("first_ms", ""),
                "cuda_median_ms": cuda_ms,
                "cuda_eval_median_ms": c.get("eval_median_ms", ""),
                "cuda_challenge_median_ms": c.get("challenge_median_ms", ""),
                "cuda_fold_median_ms": c.get("fold_median_ms", ""),
                "cuda_total_median_ms": c.get("total_median_ms", cuda_ms),
                "cuda_status": c.get("status", "missing"),
                "checksum_match": checksum_match,
                "triton_checksum": triton_checksum,
                "cuda_checksum": cuda_checksum,
                "cuda_speedup_vs_triton": speedup,
            }
            writer.writerow(row)
            print(
                f"{row['workload']:24s} field={row['field']:28s} rounds={key[1]:<2d} "
                f"triton_ms={triton_ms} cuda_ms={cuda_ms} "
                f"checksum_match={checksum_match} speedup={speedup}",
                flush=True,
            )
            if checksum_match == "false":
                mismatches.append(f"{key[0]} {key[2]}-bit r={key[1]}")
                print(
                    f"WARNING: CHECKSUM MISMATCH {key[0]} {key[2]}-bit r={key[1]}: "
                    f"triton={triton_checksum} cuda={cuda_checksum}; not a validated result",
                    flush=True,
                )

    if mismatches:
        print(
            f"WARNING: {len(mismatches)} of {len(keys)} configuration(s) have mismatching "
            f"Triton/CUDA checksums (checksum_match=false): {', '.join(mismatches)}",
            flush=True,
        )
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
