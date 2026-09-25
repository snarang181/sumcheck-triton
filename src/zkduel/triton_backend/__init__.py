"""Triton backend kernels and 256-bit limb helpers."""

from .u256_sumcheck import zkduel_uint256_static_eval

__all__ = ["zkduel_uint256_static_eval"]
