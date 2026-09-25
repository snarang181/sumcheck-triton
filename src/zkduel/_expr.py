"""Expression utilities shared by the GPU backends."""

from __future__ import annotations

from typing import Any, Sequence

Expression = Sequence[Any]
NormalizedExpression = tuple[tuple[int, tuple[str, ...]], ...]


def _is_coeff_term(term: Any) -> bool:
    if not isinstance(term, (list, tuple)) or len(term) != 2:
        return False
    coeff, factors = term
    if isinstance(coeff, str) or isinstance(factors, str):
        return False
    try:
        int(coeff)
    except (TypeError, ValueError):
        return False
    try:
        return all(isinstance(v, str) for v in factors)
    except TypeError:
        return False


def normalize_expression(expression: Expression) -> NormalizedExpression:
    """Normalize expression terms to ``(coefficient, factor_names)``.

    The original syntax, ``[["a", "b"], ["c"]]``, remains valid and implies
    coefficient 1 for every term. Coefficient-aware terms are written as
    ``[(3, ["a", "b"]), (-1, ["c"])]``.
    """
    normalized: list[tuple[int, tuple[str, ...]]] = []
    for term in expression:
        if _is_coeff_term(term):
            coeff, factors = term
            normalized.append((int(coeff), tuple(factors)))
            continue
        if isinstance(term, str):
            raise TypeError("Expression terms must be sequences of variable names")
        try:
            factors = tuple(term)
        except TypeError as exc:
            raise TypeError("Expression terms must be sequences") from exc
        if not all(isinstance(v, str) for v in factors):
            raise TypeError(
                "Expression terms must contain variable names, or be "
                "(coefficient, variable_names) pairs"
            )
        normalized.append((1, factors))
    if not normalized:
        raise ValueError("expression must contain at least one term")
    return tuple(normalized)


def expression_vars(expression: Expression) -> list[str]:
    seen: list[str] = []
    for _coeff, term in normalize_expression(expression):
        for v in term:
            if v not in seen:
                seen.append(v)
    return seen


def expression_degree(expression: Expression) -> int:
    return max(len(term) for _coeff, term in normalize_expression(expression))


def encode_expression(
    expression: Expression,
) -> tuple[list[str], tuple, tuple, tuple]:
    """Return (var_names, term_lens, term_offsets, term_var_indices_flat).

    ``term_lens[i]`` is the number of variables in term i, ``term_offsets[i]``
    is the starting index into ``term_var_indices_flat`` for term i, and
    ``term_var_indices_flat`` is a flat tuple of variable indices (into
    ``var_names``) of total length ``sum(term_lens)``. Tuples are returned so
    the data can flow as ``tl.constexpr`` values to a Triton kernel.
    """
    var_names, _term_coeffs, term_lens, term_offsets, flat = encode_expression_with_coeffs(
        expression
    )
    return var_names, term_lens, term_offsets, flat


def encode_expression_with_coeffs(
    expression: Expression,
) -> tuple[list[str], tuple, tuple, tuple, tuple]:
    """Return expression encoding with per-term integer coefficients."""
    terms = normalize_expression(expression)
    var_names = expression_vars(expression)
    var_to_idx = {v: i for i, v in enumerate(var_names)}
    term_coeffs: list[int] = []
    term_lens: list[int] = []
    term_offsets: list[int] = []
    flat: list[int] = []
    for coeff, term in terms:
        term_offsets.append(len(flat))
        term_coeffs.append(coeff)
        term_lens.append(len(term))
        for v in term:
            flat.append(var_to_idx[v])
    return (
        var_names,
        tuple(term_coeffs),
        tuple(term_lens),
        tuple(term_offsets),
        tuple(flat),
    )
