"""Small benchmark for the static 256-bit Triton SumCheck backend."""

from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from zkduel import available_backends, get_workload, sumcheck
from zkduel._expr import expression_vars
from zkduel.workloads import WORKLOADS

PRIME_256 = int("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16)


def _rand_tables(expression, num_rounds: int, q: int, seed: int):
    rng = random.Random(seed)
    n = 1 << num_rounds
    return {v: [rng.randrange(q) for _ in range(n)] for v in expression_vars(expression)}


def _rand_challenges(num_rounds: int, q: int, seed: int):
    rng = random.Random(seed + 99)
    return [rng.randrange(q) for _ in range(num_rounds)]


def _time_one(tables, q, expression, challenges, rounds, warmups, repeats):
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

    times = []
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
        times.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", default="poly_ab", choices=sorted(WORKLOADS))
    ap.add_argument("--rounds", type=int, nargs="+", default=[8, 12, 16])
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--csv", action="store_true")
    args = ap.parse_args()

    available = set(available_backends())
    if "u256-static-eval" not in available:
        raise SystemExit(f"u256-static-eval unavailable; available={sorted(available)}")

    q = PRIME_256
    expression = get_workload(args.workload)

    if args.csv:
        print("backend,workload,bit_width,rounds,N,median_ms")
    else:
        print(f"workload={args.workload} bit_width=256 q={q}")
        print(f"available_backends={sorted(available)}")
        print(f"{'backend':<18}{'rounds':<8}{'N':<12}{'median_ms':<12}")

    for rounds in args.rounds:
        tables = _rand_tables(expression, rounds, q, seed=args.seed + rounds)
        challenges = _rand_challenges(rounds, q, seed=args.seed + rounds)
        ms = _time_one(tables, q, expression, challenges, rounds, args.warmups, args.repeats)
        if args.csv:
            print(f"u256-static-eval,{args.workload},256,{rounds},{1 << rounds},{ms:.3f}")
        else:
            print(f"{'u256-static-eval':<18}{rounds:<8}{1 << rounds:<12}{ms:<12.3f}")


if __name__ == "__main__":
    main()
