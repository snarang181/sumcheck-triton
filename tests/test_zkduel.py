from __future__ import annotations

import random

import numpy as np
import pytest

from zkduel import available_backends, get_workload, sumcheck
from zkduel._expr import expression_vars

PRIME_256 = int("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16)
POLY_WORKLOAD_NAMES = [
    "poly_a",
    "poly_ab",
    "poly_ab_plus_c",
    "poly_abc",
    "poly_aabbc",
    "poly_abc_plus_de",
    "poly_abcg_plus_deg",
]


def _rand_tables(expression, num_rounds: int, q: int, seed: int):
    rng = random.Random(seed)
    n = 1 << num_rounds
    return {v: [rng.randrange(q) for _ in range(n)] for v in expression_vars(expression)}


def _rand_challenges(num_rounds: int, q: int, seed: int):
    rng = random.Random(seed + 99)
    return [rng.randrange(q) for _ in range(num_rounds)]


@pytest.mark.parametrize("workload", POLY_WORKLOAD_NAMES)
def test_polynomial_ladder_reference_claim_consistency(workload):
    q = PRIME_256
    expression = get_workload(workload)
    rounds = 4
    tables = _rand_tables(expression, rounds, q, seed=11)
    challenges = _rand_challenges(rounds, q, seed=11)
    claim, evals = sumcheck(
        tables,
        q=q,
        expression=expression,
        challenges=challenges,
        num_rounds=rounds,
        bit_width=256,
        backend="reference",
    )
    assert (int(evals[0][0]) + int(evals[0][1])) % q == claim


@pytest.mark.parametrize("workload", POLY_WORKLOAD_NAMES)
def test_u256_static_eval_polynomial_ladder_matches_reference(workload):
    backend = "u256-static-eval"
    if backend not in available_backends():
        pytest.skip(f"{backend} unavailable")
    q = PRIME_256
    expression = get_workload(workload)
    rounds = 2
    tables = _rand_tables(expression, rounds, q, seed=256)
    challenges = _rand_challenges(rounds, q, seed=256)
    ref_claim, ref_evals = sumcheck(
        tables,
        q=q,
        expression=expression,
        challenges=challenges,
        num_rounds=rounds,
        bit_width=256,
        backend="reference",
    )
    tri_claim, tri_evals = sumcheck(
        tables,
        q=q,
        expression=expression,
        challenges=challenges,
        num_rounds=rounds,
        bit_width=256,
        backend=backend,
    )
    assert tri_claim == ref_claim
    np.testing.assert_array_equal(tri_evals, ref_evals)
