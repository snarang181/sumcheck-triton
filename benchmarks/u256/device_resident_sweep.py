from __future__ import annotations

import argparse
import csv
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch

from zkduel._expr import expression_degree, expression_vars
from zkduel.triton_backend.u256_sumcheck import (
    _fold_static_tables,
    _make_field,
    _round_eval_static_sum,
    _static_workload_id,
)
from zkduel.triton_backend.uint256 import montgomery_encode_values, to_uint256_tensor
from zkduel.workloads import get_workload

PRIME_256 = int(
    "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
    16,
)

POLY_WORKLOADS = [
    ("poly_a", "a"),
    ("poly_ab", "a*b"),
    ("poly_ab_plus_c", "a*b + c"),
    ("poly_abc", "a*b*c"),
    ("poly_aabbc", "a*a*b*b*c"),
    ("poly_abc_plus_de", "a*b*c + d*e"),
    ("poly_abcg_plus_deg", "a*b*c*g + d*e*g"),
]

# Conservative large-GPU schedules. Memory checks skip rows that do not fit.
# One 256-bit element = 32 bytes. Peak memory is roughly input table plus folded output.
CAPACITY_SCHEDULE = {
    "poly_a": [28],
    "poly_ab": [27],
    "poly_ab_plus_c": [26],
    "poly_abc": [26],
    "poly_aabbc": [26],
    "poly_abc_plus_de": [25],
    "poly_abcg_plus_deg": [25],
}

LARGE_SCHEDULE = {workload: [18, 20, 22, 24] for workload, _poly in POLY_WORKLOADS}


def gib(x: int | float) -> float:
    return float(x) / (1024.0**3)


def estimate_input_bytes(num_vars: int, n: int) -> int:
    return num_vars * n * 8 * 4


def estimate_peak_bytes(num_vars: int, n: int) -> int:
    # Current table plus folded table is about 1.5x input.
    # Use 1.7x as a conservative estimate for allocator overhead/partials.
    return int(1.7 * estimate_input_bytes(num_vars, n))


def make_device_tables(
    num_vars: int,
    n: int,
    device: torch.device,
    seed: int,
    table_mode: str,
) -> torch.Tensor:
    """Create valid device-resident 256-bit residues without Python int tables.

    The kernels operate on Montgomery-domain residues. Any residue below q is a
    valid Montgomery-domain field representation of some logical value.

    `matched` mirrors `cuda/src/u256_sumcheck.cu` exactly:

        limb0 = seed + 7919 * (var + 1) + 104729 * (off + 1) mod 2^32

    `constant` preserves the older large-device fill pattern, where each
    variable row is constant.
    """
    tables = torch.zeros((num_vars, n, 8), dtype=torch.uint32, device=device)
    if table_mode == "constant":
        for i in range(num_vars):
            value = (seed + 7919 * (i + 1)) & 0xFFFFFFFF
            tables[i, :, 0].fill_(value)
        return tables.contiguous()

    if table_mode != "matched":
        raise ValueError(f"unknown table mode: {table_mode}")

    # Use an int64 temporary because uint32 arithmetic support is limited in
    # PyTorch CUDA, then explicitly wrap modulo 2^32 to mirror uint32_t casts.
    offsets = torch.arange(1, n + 1, dtype=torch.int64, device=device)
    wrap = 1 << 32
    for i in range(num_vars):
        values = seed + 7919 * (i + 1) + 104729 * offsets
        tables[i, :, 0] = torch.remainder(values, wrap).to(torch.uint32)
    return tables.contiguous()


def make_challenges(num_rounds: int, q: int, seed: int, challenge_mode: str) -> list[int]:
    if challenge_mode == "ones":
        return [1 for _ in range(num_rounds)]
    if challenge_mode == "deterministic":
        # Deterministic small challenges. They are valid field values and avoid
        # Python big-int table generation.
        return [((seed + 104729 * (i + 1)) % q) for i in range(num_rounds)]
    raise ValueError(f"unknown challenge mode: {challenge_mode}")


def run_static_device_tables(
    tables0: torch.Tensor,
    *,
    q: int,
    expression,
    challenges: list[int],
    rounds: int,
    field,
) -> None:
    workload_id = _static_workload_id(expression)
    num_vars = len(expression_vars(expression))
    degree = expression_degree(expression)

    point_monts = to_uint256_tensor(
        montgomery_encode_values(range(degree + 1), q),
        device=field.device,
    ).contiguous()

    r_monts = to_uint256_tensor(
        montgomery_encode_values([int(x) % q for x in challenges], q),
        device=field.device,
    ).contiguous()

    tables = tables0
    cur_n = 1 << rounds

    for round_idx in range(rounds):
        half = cur_n // 2

        for t_idx in range(degree + 1):
            _round_eval_static_sum(
                tables,
                half_n=half,
                point_mont=point_monts[t_idx].contiguous(),
                field=field,
                workload_id=workload_id,
                num_vars=num_vars,
            )

        if round_idx < rounds - 1:
            tables = _fold_static_tables(
                tables,
                half_n=half,
                r_mont=r_monts[round_idx].contiguous(),
                field=field,
                num_vars=num_vars,
            )
            cur_n = half


def time_one(
    workload: str,
    rounds: int,
    warmups: int,
    repeats: int,
    seed: int,
    force: bool,
    table_mode: str,
    challenge_mode: str,
):
    q = PRIME_256
    expression = get_workload(workload)
    var_names = expression_vars(expression)
    num_vars = len(var_names)
    n = 1 << rounds

    free_bytes, total_bytes = torch.cuda.mem_get_info()
    input_bytes = estimate_input_bytes(num_vars, n)
    peak_est = estimate_peak_bytes(num_vars, n)

    if not force and peak_est > int(0.90 * free_bytes):
        return {
            "status": "skipped_estimate",
            "median_ms": "",
            "input_gib": f"{gib(input_bytes):.3f}",
            "peak_est_gib": f"{gib(peak_est):.3f}",
            "peak_allocated_gib": "",
            "gpu_free_gib_before": f"{gib(free_bytes):.3f}",
            "gpu_total_gib": f"{gib(total_bytes):.3f}",
            "note": "estimated peak exceeds 90% of currently free GPU memory",
            "table_mode": table_mode,
            "challenge_mode": challenge_mode,
        }

    field = _make_field(q)
    tables0 = make_device_tables(
        num_vars, n, field.device, seed=seed + rounds, table_mode=table_mode
    )
    challenges = make_challenges(rounds, q, seed=seed + rounds, challenge_mode=challenge_mode)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    try:
        for _ in range(warmups):
            run_static_device_tables(
                tables0,
                q=q,
                expression=expression,
                challenges=challenges,
                rounds=rounds,
                field=field,
            )
            torch.cuda.synchronize()

        samples = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_static_device_tables(
                tables0,
                q=q,
                expression=expression,
                challenges=challenges,
                rounds=rounds,
                field=field,
            )
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)

        peak_alloc = torch.cuda.max_memory_allocated()

        return {
            "status": "ok",
            "median_ms": f"{statistics.median(samples):.3f}",
            "input_gib": f"{gib(input_bytes):.3f}",
            "peak_est_gib": f"{gib(peak_est):.3f}",
            "peak_allocated_gib": f"{gib(peak_alloc):.3f}",
            "gpu_free_gib_before": f"{gib(free_bytes):.3f}",
            "gpu_total_gib": f"{gib(total_bytes):.3f}",
            "note": "",
            "table_mode": table_mode,
            "challenge_mode": challenge_mode,
        }

    except torch.cuda.OutOfMemoryError as e:
        torch.cuda.empty_cache()
        return {
            "status": "oom",
            "median_ms": "",
            "input_gib": f"{gib(input_bytes):.3f}",
            "peak_est_gib": f"{gib(peak_est):.3f}",
            "peak_allocated_gib": "",
            "gpu_free_gib_before": f"{gib(free_bytes):.3f}",
            "gpu_total_gib": f"{gib(total_bytes):.3f}",
            "note": str(e).replace("\n", " ")[:240],
            "table_mode": table_mode,
            "challenge_mode": challenge_mode,
        }


def build_jobs(args):
    if args.schedule == "large":
        return [(w, r) for w, _ in POLY_WORKLOADS for r in LARGE_SCHEDULE[w]]

    if args.schedule == "capacity":
        return [(w, r) for w, _ in POLY_WORKLOADS for r in CAPACITY_SCHEDULE[w]]

    if args.schedule == "manual":
        if not args.rounds:
            raise SystemExit("--rounds is required for --schedule manual")
        workloads = args.workloads or [w for w, _ in POLY_WORKLOADS]
        return [(w, r) for w in workloads for r in args.rounds]

    raise SystemExit(f"unknown schedule: {args.schedule}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=["large", "capacity", "manual"], default="manual")
    ap.add_argument("--workloads", nargs="+", default=None)
    ap.add_argument("--rounds", type=int, nargs="+", default=None)
    ap.add_argument("--warmups", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--table-mode", choices=["matched", "constant"], default="matched")
    ap.add_argument("--challenge-mode", choices=["ones", "deterministic"], default="ones")
    ap.add_argument("--out", default="results/poly_ladder_large_device.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmarks/u256/device_resident_sweep.py")

    workload_to_poly = dict(POLY_WORKLOADS)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "workload",
        "polynomial",
        "backend",
        "bit_width",
        "rounds",
        "N",
        "vars",
        "degree",
        "terms",
        "warmups",
        "repeats",
        "median_ms",
        "status",
        "input_gib",
        "peak_est_gib",
        "peak_allocated_gib",
        "gpu_free_gib_before",
        "gpu_total_gib",
        "note",
        "table_mode",
        "challenge_mode",
    ]

    jobs = build_jobs(args)

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for workload, rounds in jobs:
            expression = get_workload(workload)
            result = time_one(
                workload,
                rounds,
                warmups=args.warmups,
                repeats=args.repeats,
                seed=args.seed,
                force=args.force,
                table_mode=args.table_mode,
                challenge_mode=args.challenge_mode,
            )

            row = {
                "workload": workload,
                "polynomial": workload_to_poly[workload],
                "backend": "u256-static-eval-device-resident",
                "bit_width": 256,
                "rounds": rounds,
                "N": 1 << rounds,
                "vars": len(expression_vars(expression)),
                "degree": expression_degree(expression),
                "terms": len(expression),
                "warmups": args.warmups,
                "repeats": args.repeats,
                **result,
            }
            writer.writerow(row)
            f.flush()

            print(
                f"{workload:24s} rounds={rounds:<2d} N={1 << rounds:<12d} "
                f"status={row['status']:<16s} median_ms={row['median_ms']} "
                f"input_gib={row['input_gib']} peak_alloc_gib={row['peak_allocated_gib']}",
                flush=True,
            )

            torch.cuda.empty_cache()

    print(out_path)


if __name__ == "__main__":
    main()
