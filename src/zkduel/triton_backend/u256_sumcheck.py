"""Static 256-bit Triton SumCheck backend.

This is the main GPU path for the project. It uses fixed Triton source for the
polynomial ladder workloads and represents field elements as eight
little-endian uint32 limbs in Montgomery form.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from .._expr import Expression, expression_degree, expression_vars, normalize_expression
from ..workloads import WORKLOADS
from .uint256 import (
    _add_limbs,
    _cmp_limbs,
    _load_modulus,
    _mont_mul_limbs,
    _sub_limbs,
    from_uint256_tensor,
    montgomery_decode_values,
    montgomery_encode_values,
    montgomery_nprime,
    to_uint256_tensor,
    uint256_add_mod,
)

_STATIC_BLOCK = 64

_WORKLOAD_POLY_A = 1
_WORKLOAD_POLY_AB = 2
_WORKLOAD_POLY_AB_PLUS_C = 3
_WORKLOAD_POLY_ABC = 4
_WORKLOAD_POLY_AABBC = 5
_WORKLOAD_POLY_ABC_PLUS_DE = 6
_WORKLOAD_POLY_ABCG_PLUS_DEG = 7

_STATIC_WORKLOAD_IDS = {
    "poly_a": _WORKLOAD_POLY_A,
    "poly_ab": _WORKLOAD_POLY_AB,
    "poly_ab_plus_c": _WORKLOAD_POLY_AB_PLUS_C,
    "poly_abc": _WORKLOAD_POLY_ABC,
    "poly_aabbc": _WORKLOAD_POLY_AABBC,
    "poly_abc_plus_de": _WORKLOAD_POLY_ABC_PLUS_DE,
    "poly_abcg_plus_deg": _WORKLOAD_POLY_ABCG_PLUS_DEG,
}
_STATIC_WORKLOAD_BY_EXPR = {
    normalize_expression(WORKLOADS[name]): workload_id
    for name, workload_id in _STATIC_WORKLOAD_IDS.items()
}


@dataclass(frozen=True)
class Uint256Field:
    modulus: int
    modulus_tensor: torch.Tensor
    nprime: int
    device: torch.device


# -----------------------------------------------------------------------------
# Host helpers
# -----------------------------------------------------------------------------


def _as_int_list(arr: Iterable[int] | np.ndarray | torch.Tensor, q: int) -> list[int]:
    if isinstance(arr, torch.Tensor):
        if arr.ndim == 2 and arr.shape[1] == 8 and arr.dtype is torch.uint32:
            return [x % q for x in from_uint256_tensor(arr)]
        return [int(x) % q for x in arr.detach().cpu().reshape(-1).tolist()]
    if isinstance(arr, np.ndarray):
        return [int(x) % q for x in arr.reshape(-1).tolist()]
    return [int(x) % q for x in arr]


def _make_field(q: int) -> Uint256Field:
    if q <= 2 or q >= (1 << 256):
        raise ValueError(f"q must be an odd modulus below 2^256, got {q}")
    if q & 1 == 0:
        raise ValueError("Montgomery modulus must be odd")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device not available; Triton uint256 backend needs a GPU")

    device = torch.device("cuda")
    return Uint256Field(
        modulus=int(q),
        modulus_tensor=to_uint256_tensor([int(q)], device=device)[0].contiguous(),
        nprime=montgomery_nprime(int(q)),
        device=device,
    )


def _to_mont_tensor(values: Iterable[int], field: Uint256Field) -> torch.Tensor:
    return to_uint256_tensor(montgomery_encode_values(values, field.modulus), device=field.device)


def _constant_mont(value: int, n: int, field: Uint256Field) -> torch.Tensor:
    encoded = montgomery_encode_values([value % field.modulus], field.modulus)[0]
    return to_uint256_tensor([encoded] * n, device=field.device)


def _decode_scalar(x: torch.Tensor, field: Uint256Field) -> int:
    return montgomery_decode_values(from_uint256_tensor(x), field.modulus)[0]


def _sum_mod_mont(values: torch.Tensor, field: Uint256Field) -> torch.Tensor:
    cur = values.contiguous()
    while cur.shape[0] > 1:
        pair_n = cur.shape[0] // 2
        left = cur[: pair_n * 2 : 2].contiguous()
        right = cur[1 : pair_n * 2 : 2].contiguous()
        summed = uint256_add_mod(left, right, field.modulus_tensor)
        if cur.shape[0] & 1:
            cur = torch.cat([summed, cur[-1:].contiguous()], dim=0)
        else:
            cur = summed
    return cur


def _static_workload_id(expression: Expression) -> int:
    normalized = normalize_expression(expression)
    try:
        return _STATIC_WORKLOAD_BY_EXPR[normalized]
    except KeyError as exc:
        supported = ", ".join(sorted(_STATIC_WORKLOAD_IDS))
        raise ValueError(
            "static uint256 kernels currently support only the polynomial "
            f"ladder workloads: {supported}"
        ) from exc


def _prepare_static_tables(
    eval_tables: Mapping[str, Iterable[int] | np.ndarray | torch.Tensor],
    *,
    q: int,
    expression: Expression,
    num_rounds: int,
    field: Uint256Field,
) -> tuple[list[str], torch.Tensor]:
    var_names = expression_vars(expression)
    n = 1 << num_rounds
    rows = []
    for name in var_names:
        values = _as_int_list(eval_tables[name], q)
        if len(values) != n:
            raise ValueError(f"Table for {name!r} has length {len(values)}, expected {n}")
        rows.append(_to_mont_tensor(values, field))
    return var_names, torch.stack(rows, dim=0).contiguous()


# -----------------------------------------------------------------------------
# Static Triton uint256 building blocks
# -----------------------------------------------------------------------------


@triton.jit
def _zero_u256(seed):
    z = (seed * 0).to(tl.uint64)
    return (z, z, z, z, z, z, z, z)


@triton.jit
def _load_masked_u256(ptr, offs, mask):
    return (
        tl.load(ptr + offs * 8 + 0, mask=mask, other=0).to(tl.uint64),
        tl.load(ptr + offs * 8 + 1, mask=mask, other=0).to(tl.uint64),
        tl.load(ptr + offs * 8 + 2, mask=mask, other=0).to(tl.uint64),
        tl.load(ptr + offs * 8 + 3, mask=mask, other=0).to(tl.uint64),
        tl.load(ptr + offs * 8 + 4, mask=mask, other=0).to(tl.uint64),
        tl.load(ptr + offs * 8 + 5, mask=mask, other=0).to(tl.uint64),
        tl.load(ptr + offs * 8 + 6, mask=mask, other=0).to(tl.uint64),
        tl.load(ptr + offs * 8 + 7, mask=mask, other=0).to(tl.uint64),
    )


@triton.jit
def _store_masked_u256(ptr, offs, value, mask):
    tl.store(ptr + offs * 8 + 0, value[0].to(tl.uint32), mask=mask)
    tl.store(ptr + offs * 8 + 1, value[1].to(tl.uint32), mask=mask)
    tl.store(ptr + offs * 8 + 2, value[2].to(tl.uint32), mask=mask)
    tl.store(ptr + offs * 8 + 3, value[3].to(tl.uint32), mask=mask)
    tl.store(ptr + offs * 8 + 4, value[4].to(tl.uint32), mask=mask)
    tl.store(ptr + offs * 8 + 5, value[5].to(tl.uint32), mask=mask)
    tl.store(ptr + offs * 8 + 6, value[6].to(tl.uint32), mask=mask)
    tl.store(ptr + offs * 8 + 7, value[7].to(tl.uint32), mask=mask)


@triton.jit
def _add_mod_u256(a, b, m):
    s, carry = _add_limbs(a, b)
    gt, eq, _lt = _cmp_limbs(s, m)
    should_sub = (carry != 0) | gt | eq
    reduced, _borrow = _sub_limbs(s, m)
    return (
        tl.where(should_sub, reduced[0], s[0]),
        tl.where(should_sub, reduced[1], s[1]),
        tl.where(should_sub, reduced[2], s[2]),
        tl.where(should_sub, reduced[3], s[3]),
        tl.where(should_sub, reduced[4], s[4]),
        tl.where(should_sub, reduced[5], s[5]),
        tl.where(should_sub, reduced[6], s[6]),
        tl.where(should_sub, reduced[7], s[7]),
    )


@triton.jit
def _sub_mod_u256(a, b, m):
    d, borrow = _sub_limbs(a, b)
    corrected, _carry = _add_limbs(d, m)
    should_add = borrow != 0
    return (
        tl.where(should_add, corrected[0], d[0]),
        tl.where(should_add, corrected[1], d[1]),
        tl.where(should_add, corrected[2], d[2]),
        tl.where(should_add, corrected[3], d[3]),
        tl.where(should_add, corrected[4], d[4]),
        tl.where(should_add, corrected[5], d[5]),
        tl.where(should_add, corrected[6], d[6]),
        tl.where(should_add, corrected[7], d[7]),
    )


@triton.jit
def _reduce_step_u256(cur, m, HALF: tl.constexpr):
    r0 = tl.reshape(cur[0], (HALF, 2))
    r1 = tl.reshape(cur[1], (HALF, 2))
    r2 = tl.reshape(cur[2], (HALF, 2))
    r3 = tl.reshape(cur[3], (HALF, 2))
    r4 = tl.reshape(cur[4], (HALF, 2))
    r5 = tl.reshape(cur[5], (HALF, 2))
    r6 = tl.reshape(cur[6], (HALF, 2))
    r7 = tl.reshape(cur[7], (HALF, 2))
    a0, b0 = tl.split(r0)
    a1, b1 = tl.split(r1)
    a2, b2 = tl.split(r2)
    a3, b3 = tl.split(r3)
    a4, b4 = tl.split(r4)
    a5, b5 = tl.split(r5)
    a6, b6 = tl.split(r6)
    a7, b7 = tl.split(r7)
    return _add_mod_u256((a0, a1, a2, a3, a4, a5, a6, a7), (b0, b1, b2, b3, b4, b5, b6, b7), m)


@triton.jit
def _reduce64_u256(value, m, mask):
    cur = (
        tl.where(mask, value[0], 0),
        tl.where(mask, value[1], 0),
        tl.where(mask, value[2], 0),
        tl.where(mask, value[3], 0),
        tl.where(mask, value[4], 0),
        tl.where(mask, value[5], 0),
        tl.where(mask, value[6], 0),
        tl.where(mask, value[7], 0),
    )
    cur = _reduce_step_u256(cur, m, HALF=32)
    cur = _reduce_step_u256(cur, m, HALF=16)
    cur = _reduce_step_u256(cur, m, HALF=8)
    cur = _reduce_step_u256(cur, m, HALF=4)
    cur = _reduce_step_u256(cur, m, HALF=2)
    cur = _reduce_step_u256(cur, m, HALF=1)
    return cur


@triton.jit
def _store_block_partial_u256(out_ptr, partial_idx, value):
    tl.store(out_ptr + partial_idx * 8 + 0, tl.sum(value[0]).to(tl.uint32))
    tl.store(out_ptr + partial_idx * 8 + 1, tl.sum(value[1]).to(tl.uint32))
    tl.store(out_ptr + partial_idx * 8 + 2, tl.sum(value[2]).to(tl.uint32))
    tl.store(out_ptr + partial_idx * 8 + 3, tl.sum(value[3]).to(tl.uint32))
    tl.store(out_ptr + partial_idx * 8 + 4, tl.sum(value[4]).to(tl.uint32))
    tl.store(out_ptr + partial_idx * 8 + 5, tl.sum(value[5]).to(tl.uint32))
    tl.store(out_ptr + partial_idx * 8 + 6, tl.sum(value[6]).to(tl.uint32))
    tl.store(out_ptr + partial_idx * 8 + 7, tl.sum(value[7]).to(tl.uint32))


@triton.jit
def _fold_one_var_u256(e, d, r_mont, m, nprime):
    prod = _mont_mul_limbs(d, r_mont, m, nprime)
    return _add_mod_u256(e, prod, m)


@triton.jit
def _point_one_var_u256(e, d, point_mont, m, nprime):
    prod = _mont_mul_limbs(d, point_mont, m, nprime)
    return _add_mod_u256(e, prod, m)


@triton.jit
def _round_eval_poly_a_u256_kernel(
    tables_ptr,
    half_n,
    modulus_ptr,
    nprime,
    point_mont_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_n
    row_stride = 2 * half_n
    twice_offs = 2 * offs
    m = _load_modulus(modulus_ptr)
    point_mont = _load_masked_u256(point_mont_ptr, offs * 0, offs == offs)

    base0 = tables_ptr + 0 * row_stride * 8
    e0 = _load_masked_u256(base0, twice_offs, mask)
    o0 = _load_masked_u256(base0, twice_offs + 1, mask)
    d0 = _sub_mod_u256(o0, e0, m)
    p0 = _point_one_var_u256(e0, d0, point_mont, m, nprime)

    partial = _reduce64_u256(p0, m, mask)
    _store_block_partial_u256(out_ptr, pid, partial)


@triton.jit
def _round_eval_poly_ab_u256_kernel(
    tables_ptr,
    half_n,
    modulus_ptr,
    nprime,
    point_mont_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_n
    row_stride = 2 * half_n
    twice_offs = 2 * offs
    m = _load_modulus(modulus_ptr)
    point_mont = _load_masked_u256(point_mont_ptr, offs * 0, offs == offs)

    base0 = tables_ptr + 0 * row_stride * 8
    base1 = tables_ptr + 1 * row_stride * 8
    e0 = _load_masked_u256(base0, twice_offs, mask)
    o0 = _load_masked_u256(base0, twice_offs + 1, mask)
    e1 = _load_masked_u256(base1, twice_offs, mask)
    o1 = _load_masked_u256(base1, twice_offs + 1, mask)

    d0 = _sub_mod_u256(o0, e0, m)
    d1 = _sub_mod_u256(o1, e1, m)
    p0 = _point_one_var_u256(e0, d0, point_mont, m, nprime)
    p1 = _point_one_var_u256(e1, d1, point_mont, m, nprime)
    total = _mont_mul_limbs(p0, p1, m, nprime)

    partial = _reduce64_u256(total, m, mask)
    _store_block_partial_u256(out_ptr, pid, partial)


@triton.jit
def _round_eval_poly_ab_plus_c_u256_kernel(
    tables_ptr,
    half_n,
    modulus_ptr,
    nprime,
    point_mont_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_n
    row_stride = 2 * half_n
    twice_offs = 2 * offs
    m = _load_modulus(modulus_ptr)
    point_mont = _load_masked_u256(point_mont_ptr, offs * 0, offs == offs)

    base0 = tables_ptr + 0 * row_stride * 8
    base1 = tables_ptr + 1 * row_stride * 8
    base2 = tables_ptr + 2 * row_stride * 8
    e0 = _load_masked_u256(base0, twice_offs, mask)
    o0 = _load_masked_u256(base0, twice_offs + 1, mask)
    e1 = _load_masked_u256(base1, twice_offs, mask)
    o1 = _load_masked_u256(base1, twice_offs + 1, mask)
    e2 = _load_masked_u256(base2, twice_offs, mask)
    o2 = _load_masked_u256(base2, twice_offs + 1, mask)

    d0 = _sub_mod_u256(o0, e0, m)
    d1 = _sub_mod_u256(o1, e1, m)
    d2 = _sub_mod_u256(o2, e2, m)
    p0 = _point_one_var_u256(e0, d0, point_mont, m, nprime)
    p1 = _point_one_var_u256(e1, d1, point_mont, m, nprime)
    p2 = _point_one_var_u256(e2, d2, point_mont, m, nprime)
    ab = _mont_mul_limbs(p0, p1, m, nprime)
    total = _add_mod_u256(ab, p2, m)

    partial = _reduce64_u256(total, m, mask)
    _store_block_partial_u256(out_ptr, pid, partial)


@triton.jit
def _round_eval_poly_abc_u256_kernel(
    tables_ptr,
    half_n,
    modulus_ptr,
    nprime,
    point_mont_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_n
    row_stride = 2 * half_n
    twice_offs = 2 * offs
    m = _load_modulus(modulus_ptr)
    point_mont = _load_masked_u256(point_mont_ptr, offs * 0, offs == offs)

    base0 = tables_ptr + 0 * row_stride * 8
    base1 = tables_ptr + 1 * row_stride * 8
    base2 = tables_ptr + 2 * row_stride * 8
    e0 = _load_masked_u256(base0, twice_offs, mask)
    o0 = _load_masked_u256(base0, twice_offs + 1, mask)
    e1 = _load_masked_u256(base1, twice_offs, mask)
    o1 = _load_masked_u256(base1, twice_offs + 1, mask)
    e2 = _load_masked_u256(base2, twice_offs, mask)
    o2 = _load_masked_u256(base2, twice_offs + 1, mask)

    d0 = _sub_mod_u256(o0, e0, m)
    d1 = _sub_mod_u256(o1, e1, m)
    d2 = _sub_mod_u256(o2, e2, m)
    p0 = _point_one_var_u256(e0, d0, point_mont, m, nprime)
    p1 = _point_one_var_u256(e1, d1, point_mont, m, nprime)
    p2 = _point_one_var_u256(e2, d2, point_mont, m, nprime)
    ab = _mont_mul_limbs(p0, p1, m, nprime)
    total = _mont_mul_limbs(ab, p2, m, nprime)

    partial = _reduce64_u256(total, m, mask)
    _store_block_partial_u256(out_ptr, pid, partial)


@triton.jit
def _round_eval_poly_aabbc_u256_kernel(
    tables_ptr,
    half_n,
    modulus_ptr,
    nprime,
    point_mont_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_n
    row_stride = 2 * half_n
    twice_offs = 2 * offs
    m = _load_modulus(modulus_ptr)
    point_mont = _load_masked_u256(point_mont_ptr, offs * 0, offs == offs)

    base0 = tables_ptr + 0 * row_stride * 8
    base1 = tables_ptr + 1 * row_stride * 8
    base2 = tables_ptr + 2 * row_stride * 8
    e0 = _load_masked_u256(base0, twice_offs, mask)
    o0 = _load_masked_u256(base0, twice_offs + 1, mask)
    e1 = _load_masked_u256(base1, twice_offs, mask)
    o1 = _load_masked_u256(base1, twice_offs + 1, mask)
    e2 = _load_masked_u256(base2, twice_offs, mask)
    o2 = _load_masked_u256(base2, twice_offs + 1, mask)

    d0 = _sub_mod_u256(o0, e0, m)
    d1 = _sub_mod_u256(o1, e1, m)
    d2 = _sub_mod_u256(o2, e2, m)
    p0 = _point_one_var_u256(e0, d0, point_mont, m, nprime)
    p1 = _point_one_var_u256(e1, d1, point_mont, m, nprime)
    p2 = _point_one_var_u256(e2, d2, point_mont, m, nprime)
    aa = _mont_mul_limbs(p0, p0, m, nprime)
    bb = _mont_mul_limbs(p1, p1, m, nprime)
    aabb = _mont_mul_limbs(aa, bb, m, nprime)
    total = _mont_mul_limbs(aabb, p2, m, nprime)

    partial = _reduce64_u256(total, m, mask)
    _store_block_partial_u256(out_ptr, pid, partial)


@triton.jit
def _round_eval_poly_abc_plus_de_u256_kernel(
    tables_ptr,
    half_n,
    modulus_ptr,
    nprime,
    point_mont_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_n
    row_stride = 2 * half_n
    twice_offs = 2 * offs
    m = _load_modulus(modulus_ptr)
    point_mont = _load_masked_u256(point_mont_ptr, offs * 0, offs == offs)

    base0 = tables_ptr + 0 * row_stride * 8
    base1 = tables_ptr + 1 * row_stride * 8
    base2 = tables_ptr + 2 * row_stride * 8
    base3 = tables_ptr + 3 * row_stride * 8
    base4 = tables_ptr + 4 * row_stride * 8
    e0 = _load_masked_u256(base0, twice_offs, mask)
    o0 = _load_masked_u256(base0, twice_offs + 1, mask)
    e1 = _load_masked_u256(base1, twice_offs, mask)
    o1 = _load_masked_u256(base1, twice_offs + 1, mask)
    e2 = _load_masked_u256(base2, twice_offs, mask)
    o2 = _load_masked_u256(base2, twice_offs + 1, mask)
    e3 = _load_masked_u256(base3, twice_offs, mask)
    o3 = _load_masked_u256(base3, twice_offs + 1, mask)
    e4 = _load_masked_u256(base4, twice_offs, mask)
    o4 = _load_masked_u256(base4, twice_offs + 1, mask)

    d0 = _sub_mod_u256(o0, e0, m)
    d1 = _sub_mod_u256(o1, e1, m)
    d2 = _sub_mod_u256(o2, e2, m)
    d3 = _sub_mod_u256(o3, e3, m)
    d4 = _sub_mod_u256(o4, e4, m)
    p0 = _point_one_var_u256(e0, d0, point_mont, m, nprime)
    p1 = _point_one_var_u256(e1, d1, point_mont, m, nprime)
    p2 = _point_one_var_u256(e2, d2, point_mont, m, nprime)
    p3 = _point_one_var_u256(e3, d3, point_mont, m, nprime)
    p4 = _point_one_var_u256(e4, d4, point_mont, m, nprime)
    ab = _mont_mul_limbs(p0, p1, m, nprime)
    abc = _mont_mul_limbs(ab, p2, m, nprime)
    de = _mont_mul_limbs(p3, p4, m, nprime)
    total = _add_mod_u256(abc, de, m)

    partial = _reduce64_u256(total, m, mask)
    _store_block_partial_u256(out_ptr, pid, partial)


@triton.jit
def _round_eval_poly_abcg_plus_deg_u256_kernel(
    tables_ptr,
    half_n,
    modulus_ptr,
    nprime,
    point_mont_ptr,
    out_ptr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_n
    row_stride = 2 * half_n
    twice_offs = 2 * offs
    m = _load_modulus(modulus_ptr)
    point_mont = _load_masked_u256(point_mont_ptr, offs * 0, offs == offs)

    base0 = tables_ptr + 0 * row_stride * 8
    base1 = tables_ptr + 1 * row_stride * 8
    base2 = tables_ptr + 2 * row_stride * 8
    base3 = tables_ptr + 3 * row_stride * 8
    base4 = tables_ptr + 4 * row_stride * 8
    base5 = tables_ptr + 5 * row_stride * 8
    e0 = _load_masked_u256(base0, twice_offs, mask)
    o0 = _load_masked_u256(base0, twice_offs + 1, mask)
    e1 = _load_masked_u256(base1, twice_offs, mask)
    o1 = _load_masked_u256(base1, twice_offs + 1, mask)
    e2 = _load_masked_u256(base2, twice_offs, mask)
    o2 = _load_masked_u256(base2, twice_offs + 1, mask)
    e3 = _load_masked_u256(base3, twice_offs, mask)
    o3 = _load_masked_u256(base3, twice_offs + 1, mask)
    e4 = _load_masked_u256(base4, twice_offs, mask)
    o4 = _load_masked_u256(base4, twice_offs + 1, mask)
    e5 = _load_masked_u256(base5, twice_offs, mask)
    o5 = _load_masked_u256(base5, twice_offs + 1, mask)

    d0 = _sub_mod_u256(o0, e0, m)
    d1 = _sub_mod_u256(o1, e1, m)
    d2 = _sub_mod_u256(o2, e2, m)
    d3 = _sub_mod_u256(o3, e3, m)
    d4 = _sub_mod_u256(o4, e4, m)
    d5 = _sub_mod_u256(o5, e5, m)
    p0 = _point_one_var_u256(e0, d0, point_mont, m, nprime)
    p1 = _point_one_var_u256(e1, d1, point_mont, m, nprime)
    p2 = _point_one_var_u256(e2, d2, point_mont, m, nprime)
    p3 = _point_one_var_u256(e3, d3, point_mont, m, nprime)
    p4 = _point_one_var_u256(e4, d4, point_mont, m, nprime)
    p5 = _point_one_var_u256(e5, d5, point_mont, m, nprime)
    ab = _mont_mul_limbs(p0, p1, m, nprime)
    abc = _mont_mul_limbs(ab, p2, m, nprime)
    abcg = _mont_mul_limbs(abc, p3, m, nprime)
    de = _mont_mul_limbs(p4, p5, m, nprime)
    deg = _mont_mul_limbs(de, p3, m, nprime)
    total = _add_mod_u256(abcg, deg, m)

    partial = _reduce64_u256(total, m, mask)
    _store_block_partial_u256(out_ptr, pid, partial)


@triton.jit
def _fold_static_u256_kernel(
    tables_ptr,
    fold_out_ptr,
    half_n,
    modulus_ptr,
    nprime,
    r_mont_ptr,
    BLOCK: tl.constexpr,
):
    block_pid = tl.program_id(0)
    var_idx = tl.program_id(1)
    offs = block_pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < half_n
    row_stride = 2 * half_n
    twice_offs = 2 * offs
    m = _load_modulus(modulus_ptr)
    r_mont = _load_masked_u256(r_mont_ptr, offs * 0, offs == offs)
    base = tables_ptr + var_idx * row_stride * 8
    e = _load_masked_u256(base, twice_offs, mask)
    o = _load_masked_u256(base, twice_offs + 1, mask)
    d = _sub_mod_u256(o, e, m)
    f = _fold_one_var_u256(e, d, r_mont, m, nprime)
    _store_masked_u256(fold_out_ptr + var_idx * half_n * 8, offs, f, mask)


def _round_eval_static_sum(
    tables: torch.Tensor,
    *,
    half_n: int,
    point_mont: torch.Tensor,
    field: Uint256Field,
    workload_id: int,
    num_vars: int,
) -> int:
    num_blocks = (half_n + _STATIC_BLOCK - 1) // _STATIC_BLOCK
    partials = torch.empty((num_blocks, 8), dtype=torch.uint32, device=field.device)
    kernel = None
    if workload_id == _WORKLOAD_POLY_A:
        kernel = _round_eval_poly_a_u256_kernel
    elif workload_id == _WORKLOAD_POLY_AB:
        kernel = _round_eval_poly_ab_u256_kernel
    elif workload_id == _WORKLOAD_POLY_AB_PLUS_C:
        kernel = _round_eval_poly_ab_plus_c_u256_kernel
    elif workload_id == _WORKLOAD_POLY_ABC:
        kernel = _round_eval_poly_abc_u256_kernel
    elif workload_id == _WORKLOAD_POLY_AABBC:
        kernel = _round_eval_poly_aabbc_u256_kernel
    elif workload_id == _WORKLOAD_POLY_ABC_PLUS_DE:
        kernel = _round_eval_poly_abc_plus_de_u256_kernel
    elif workload_id == _WORKLOAD_POLY_ABCG_PLUS_DEG:
        kernel = _round_eval_poly_abcg_plus_deg_u256_kernel

    if kernel is None:
        raise ValueError(f"unsupported static uint256 workload id: {workload_id}")
    kernel[(num_blocks,)](
        tables,
        half_n,
        field.modulus_tensor,
        field.nprime,
        point_mont,
        partials,
        BLOCK=_STATIC_BLOCK,
    )
    return _decode_scalar(_sum_mod_mont(partials, field), field)


def _fold_static_tables(
    tables: torch.Tensor,
    *,
    half_n: int,
    r_mont: torch.Tensor,
    field: Uint256Field,
    num_vars: int,
) -> torch.Tensor:
    num_blocks = (half_n + _STATIC_BLOCK - 1) // _STATIC_BLOCK
    fold_out = torch.empty((num_vars, half_n, 8), dtype=torch.uint32, device=field.device)
    _fold_static_u256_kernel[(num_blocks, num_vars)](
        tables,
        fold_out,
        half_n,
        field.modulus_tensor,
        field.nprime,
        r_mont,
        BLOCK=_STATIC_BLOCK,
    )
    return fold_out


def zkduel_uint256_static_eval(
    eval_tables: Mapping[str, Iterable[int] | np.ndarray | torch.Tensor],
    *,
    q: int,
    expression: Sequence[Sequence[str]],
    challenges: Sequence[int],
    num_rounds: int,
    bit_width: int = 256,
) -> tuple[int, np.ndarray]:
    """Run the split static-eval 256-bit variant.

    This is the first performance-oriented baseline. Each g(t) point is handled
    by one static Triton kernel that computes evens/odds, diffs, interpolation at
    t, expression evaluation, and block reduction. Folding is one separate
    static kernel over all variables.
    """
    if bit_width != 256:
        raise ValueError(f"u256-static-eval requires bit_width=256, got {bit_width}")

    field = _make_field(int(q))
    workload_id = _static_workload_id(expression)
    var_names, tables = _prepare_static_tables(
        eval_tables, q=int(q), expression=expression, num_rounds=num_rounds, field=field
    )
    num_vars = len(var_names)
    degree = expression_degree(expression)
    point_monts = to_uint256_tensor(
        montgomery_encode_values(range(degree + 1), int(q)), device=field.device
    ).contiguous()

    round_evals: list[list[int]] = []
    cur_n = 1 << num_rounds
    for round_idx in range(num_rounds):
        half = cur_n // 2
        g_points: list[int] = []
        for t_idx in range(degree + 1):
            point_mont = point_monts[t_idx].contiguous()
            g_points.append(
                _round_eval_static_sum(
                    tables,
                    half_n=half,
                    point_mont=point_mont,
                    field=field,
                    workload_id=workload_id,
                    num_vars=num_vars,
                )
            )
        round_evals.append(g_points)

        if round_idx < num_rounds - 1:
            r_value = int(challenges[round_idx]) % int(q)
            r_mont = to_uint256_tensor(
                montgomery_encode_values([r_value], int(q)), device=field.device
            ).contiguous()
            tables = _fold_static_tables(
                tables, half_n=half, r_mont=r_mont, field=field, num_vars=num_vars
            )
            cur_n = half

    claim0 = (round_evals[0][0] + round_evals[0][1]) % int(q)
    return int(claim0), np.array(round_evals, dtype=object)
