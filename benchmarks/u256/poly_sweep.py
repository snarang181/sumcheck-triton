"""Benchmark the small polynomial workload ladder."""

from __future__ import annotations

import argparse
import csv
import random
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from zkduel import sumcheck
from zkduel._expr import expression_degree, expression_vars
from zkduel.workloads import get_workload

PRIME_256 = int("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16)
POLY_WORKLOADS = [
    ("poly_a", "a"),
    ("poly_ab", "a*b"),
    ("poly_ab_plus_c", "a*b + c"),
    ("poly_abc", "a*b*c"),
    ("poly_aabbc", "a*a*b*b*c"),
    ("poly_abc_plus_de", "a*b*c + d*e"),
    ("poly_abcg_plus_deg", "a*b*c*g + d*e*g"),
]


def _rand_tables(expression, rounds: int, q: int, seed: int):
    rng = random.Random(seed)
    n = 1 << rounds
    return {v: [rng.randrange(q) for _ in range(n)] for v in expression_vars(expression)}


def _rand_challenges(rounds: int, q: int, seed: int):
    rng = random.Random(seed + 99)
    return [rng.randrange(q) for _ in range(rounds)]


def _time_backend(tables, q, expression, challenges, rounds, warmups, repeats):
    import torch

    for _ in range(warmups):
        sumcheck(
            tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=rounds,
            bit_width=256,
            backend="u256-static-eval",
        )
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        sumcheck(
            tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=rounds,
            bit_width=256,
            backend="u256-static-eval",
        )
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, nargs="+", default=[12])
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/poly_ladder_triton.csv")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "workload",
                "polynomial",
                "backend",
                "bit_width",
                "rounds",
                "N",
                "vars",
                "degree",
                "terms",
                "median_ms",
            ],
        )
        writer.writeheader()
        for rounds in args.rounds:
            for workload, polynomial in POLY_WORKLOADS:
                expression = get_workload(workload)
                q = PRIME_256
                tables = _rand_tables(expression, rounds, q, args.seed + rounds)
                challenges = _rand_challenges(rounds, q, args.seed + rounds)
                ms = _time_backend(
                    tables,
                    q,
                    expression,
                    challenges,
                    rounds,
                    args.warmups,
                    args.repeats,
                )
                writer.writerow(
                    {
                        "workload": workload,
                        "polynomial": polynomial,
                        "backend": "u256-static-eval",
                        "bit_width": 256,
                        "rounds": rounds,
                        "N": 1 << rounds,
                        "vars": len(expression_vars(expression)),
                        "degree": expression_degree(expression),
                        "terms": len(expression),
                        "median_ms": f"{ms:.3f}",
                    }
                )
    print(out_path)


if __name__ == "__main__":
    main()
