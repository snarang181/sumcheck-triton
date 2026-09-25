"""Public API for the 256-bit Triton SumCheck package."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

from .reference import sumcheck_reference


def available_backends() -> list[str]:
    """Return usable local backends."""
    backends = ["reference"]
    try:
        import torch
        import triton  # noqa: F401

        if torch.cuda.is_available():
            backends.append("u256-static-eval")
    except ImportError:
        pass
    return backends


def sumcheck(
    eval_tables: Mapping[str, Iterable[int] | np.ndarray],
    *,
    q: int,
    expression: Sequence[Sequence[str]],
    challenges: Sequence[int],
    num_rounds: int,
    bit_width: int = 256,
    backend: str = "u256-static-eval",
) -> tuple[int, np.ndarray]:
    """Run SumCheck with either CPU reference or the static 256-bit Triton path."""
    if bit_width != 256:
        raise ValueError(f"this package currently exposes only bit_width=256, got {bit_width}")
    if q >= (1 << 256):
        raise ValueError(f"q={q} too large for bit_width=256")

    if backend == "reference":
        claim0, evals = sumcheck_reference(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
        )
        return int(claim0), np.array(evals.tolist(), dtype=object)

    if backend == "u256-static-eval":
        from .triton_backend.u256_sumcheck import zkduel_uint256_static_eval

        return zkduel_uint256_static_eval(
            eval_tables,
            q=q,
            expression=expression,
            challenges=challenges,
            num_rounds=num_rounds,
            bit_width=bit_width,
        )

    raise ValueError(f"Unknown backend: {backend!r}; expected 'reference' or 'u256-static-eval'")
