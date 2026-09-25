"""NumPy / pure-Python reference sumcheck.

This is the ground-truth implementation used by tests. It uses Python ints
throughout, so it works for any prime that fits in a Python int. It is intentionally slow,
but correct.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from ._expr import Expression, expression_degree, expression_vars, normalize_expression


def _eval_term(coeff: int, term: Sequence[str], values: Mapping[str, int], q: int) -> int:
    acc = coeff % q
    for v in term:
        acc = (acc * values[v]) % q
    return acc


def _eval_expression(expression: Expression, values: Mapping[str, int], q: int) -> int:
    terms = normalize_expression(expression)
    return sum(_eval_term(coeff, term, values, q) for coeff, term in terms) % q


def _table_compose(expression: Expression, tables: Mapping[str, list[int]], q: int) -> list[int]:
    n = len(next(iter(tables.values())))
    out = [0] * n
    for i in range(n):
        vals = {v: tables[v][i] for v in tables}
        out[i] = _eval_expression(expression, vals, q)
    return out


def _to_int_list(arr: Iterable[int] | np.ndarray, q: int) -> list[int]:
    if isinstance(arr, np.ndarray):
        return [int(x) % q for x in arr.tolist()]
    return [int(x) % q for x in arr]


def sumcheck_reference(
    eval_tables: Mapping[str, Iterable[int] | np.ndarray],
    *,
    q: int,
    expression: Expression,
    challenges: Sequence[int],
    num_rounds: int,
) -> tuple[int, np.ndarray]:
    """Run the sumcheck prover loop with Python-int arithmetic.

    Args:
        eval_tables: per-variable evaluation tables on {0,1}^num_rounds, length
            2**num_rounds, values in [0, q).
        q: prime modulus.
        expression: polynomial expressed as list of multiplicative terms.
        challenges: verifier challenges, one per round (only the first
            ``num_rounds - 1`` are consumed for folding; the last is the
            verifier's final-point challenge but is not used here).
        num_rounds: number of variables to fold.

    Returns:
        ``(claim0, round_evals)`` where ``claim0`` is the initial sum over the
        hypercube and ``round_evals`` has shape ``(num_rounds, degree + 1)``
        with ``g_round(0), g_round(1), ..., g_round(degree)``.
    """
    var_names = expression_vars(expression)
    degree = expression_degree(expression)

    tables: dict[str, list[int]] = {v: _to_int_list(eval_tables[v], q) for v in var_names}

    expected_len = 1 << num_rounds
    for v, t in tables.items():
        if len(t) != expected_len:
            raise ValueError(f"Table for {v!r} has length {len(t)}, expected {expected_len}")

    composed = _table_compose(expression, tables, q)
    claim0 = sum(composed) % q

    round_evals: list[list[int]] = []

    for round_idx in range(num_rounds):
        evens = {v: tables[v][::2] for v in var_names}
        odds = {v: tables[v][1::2] for v in var_names}
        diffs = {v: [(o - e) % q for o, e in zip(odds[v], evens[v])] for v in var_names}

        # g(0) and g(1)
        g0 = sum(_table_compose(expression, evens, q)) % q
        g1 = sum(_table_compose(expression, odds, q)) % q
        g_points = [g0, g1]

        # g(2), ..., g(degree): start from odds (= evens + 1*diff), then add diff each step
        current = {v: list(odds[v]) for v in var_names}
        for _ in range(2, degree + 1):
            for v in var_names:
                current[v] = [(c + d) % q for c, d in zip(current[v], diffs[v])]
            g_points.append(sum(_table_compose(expression, current, q)) % q)

        round_evals.append(g_points)

        if round_idx < num_rounds - 1:
            r = int(challenges[round_idx]) % q
            new_tables: dict[str, list[int]] = {}
            for v in var_names:
                new_tables[v] = [(e + r * d) % q for e, d in zip(evens[v], diffs[v])]
            tables = new_tables

    return claim0, np.array(round_evals, dtype=object)
