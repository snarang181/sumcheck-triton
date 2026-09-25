"""Full SumCheck prime-field Montgomery sweep using Triton kernels.

This benchmark runs the same SumCheck round/eval/fold shape over actual prime
fields at 32, 64, 128, and 256-bit storage widths. Unlike
`benchmarks/wraparound_sweep/triton_sweep.py`, arithmetic here is Montgomery multiplication and
modular add/sub over a prime field for each width.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TRITON_CACHE_DIR", str(_ROOT / ".triton-cache-sumcheck-field"))
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import triton
import triton.language as tl
from kernels.poly import eval_poly_workload
from kernels.zk import eval_zk_workload
from workload_specs import (
    WORKLOAD_BY_NAME,
    WORKLOAD_NAMES,
    ZK_WORKLOAD_NAMES,
    WorkloadSpec,
    canonicalize_workload_names,
    known_workload_names,
)

from zkduel.fields import FIELD_SPECS, FieldSpec, get_field

BLOCK = 128


@triton.jit
def _zero_tuple(offs, LIMBS: tl.constexpr):
    z = (offs * 0).to(tl.uint64)
    out = ()
    for _ in tl.static_range(0, LIMBS):
        out += (z,)
    return out


@triton.jit
def _load_elem(ptr, idxs, mask, LIMBS: tl.constexpr):
    out = ()
    base = idxs.to(tl.int64) * LIMBS
    for i in tl.static_range(0, LIMBS):
        out += (tl.load(ptr + base + i, mask=mask, other=0).to(tl.uint64),)
    return out


@triton.jit
def _load_const(ptr, LIMBS: tl.constexpr):
    out = ()
    for i in tl.static_range(0, LIMBS):
        out += (tl.load(ptr + i).to(tl.uint64),)
    return out


@triton.jit
def _store_elem(ptr, idxs, value, mask, LIMBS: tl.constexpr):
    base = idxs.to(tl.int64) * LIMBS
    for i in tl.static_range(0, LIMBS):
        tl.store(ptr + base + i, value[i].to(tl.uint32), mask=mask)


@triton.jit
def _load_table_elem(ptr, idxs, row_stride, mask, LIMBS: tl.constexpr, LAYOUT: tl.constexpr):
    out = ()
    idxs = idxs.to(tl.int64)
    row_stride = row_stride.to(tl.int64)
    for i in tl.static_range(0, LIMBS):
        if LAYOUT == 0:
            addr = ptr + idxs * LIMBS + i
        else:
            addr = ptr + i * row_stride + idxs
        out += (tl.load(addr, mask=mask, other=0).to(tl.uint64),)
    return out


@triton.jit
def _store_table_elem(
    ptr, idxs, row_stride, value, mask, LIMBS: tl.constexpr, LAYOUT: tl.constexpr
):
    idxs = idxs.to(tl.int64)
    row_stride = row_stride.to(tl.int64)
    for i in tl.static_range(0, LIMBS):
        if LAYOUT == 0:
            addr = ptr + idxs * LIMBS + i
        else:
            addr = ptr + i * row_stride + idxs
        tl.store(addr, value[i].to(tl.uint32), mask=mask)


@triton.jit
def _select_tuple(cond, a, b, LIMBS: tl.constexpr):
    out = ()
    for i in tl.static_range(0, LIMBS):
        out += (tl.where(cond, a[i], b[i]),)
    return out


@triton.jit
def _ge_tuple(a, b, LIMBS: tl.constexpr):
    gt = a[0] == (a[0] + 1)
    eq = a[0] == a[0]
    for i in tl.static_range(LIMBS - 1, -1, -1):
        gt = gt | (eq & (a[i] > b[i]))
        eq = eq & (a[i] == b[i])
    return gt | eq


@triton.jit
def _add_raw(a, b, LIMBS: tl.constexpr):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    carry = a[0] * 0
    out = ()
    for i in tl.static_range(0, LIMBS):
        uv = a[i] + b[i] + carry
        out += (uv & mask32,)
        carry = uv >> 32
    return out, carry


@triton.jit
def _sub_raw(a, b, LIMBS: tl.constexpr):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    borrow = a[0] * 0
    out = ()
    for i in tl.static_range(0, LIMBS):
        bi = b[i] + borrow
        ai = a[i]
        out += ((ai - bi) & mask32,)
        borrow = tl.where(ai < bi, 1, 0).to(tl.uint64)
    return out, borrow


@triton.jit
def _add_mod(a, b, m, LIMBS: tl.constexpr):
    s, carry = _add_raw(a, b, LIMBS)
    reduced, _ = _sub_raw(s, m, LIMBS)
    need_reduce = (carry != 0) | _ge_tuple(s, m, LIMBS)
    return _select_tuple(need_reduce, reduced, s, LIMBS)


@triton.jit
def _sub_mod(a, b, m, LIMBS: tl.constexpr):
    d, borrow = _sub_raw(a, b, LIMBS)
    corrected, _ = _add_raw(d, m, LIMBS)
    return _select_tuple(borrow != 0, corrected, d, LIMBS)


@triton.jit
def _double_mod(a, m, LIMBS: tl.constexpr):
    return _add_mod(a, a, m, LIMBS)


@triton.jit
def _mul5_mod(a, m, LIMBS: tl.constexpr):
    two = _double_mod(a, m, LIMBS)
    four = _double_mod(two, m, LIMBS)
    return _add_mod(four, a, m, LIMBS)


@triton.jit
def _mont_mul(a, b, m, nprime, LIMBS: tl.constexpr):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    t = ()
    for _ in tl.static_range(0, LIMBS + 1):
        t += (a[0] * 0,)

    for i in tl.static_range(0, LIMBS):
        carry = a[0] * 0
        prod = ()
        for j in tl.static_range(0, LIMBS):
            uv = t[j] + a[j] * b[i] + carry
            prod += (uv & mask32,)
            carry = uv >> 32
        uvn = t[LIMBS] + carry
        t = prod + (uvn & mask32,)

        q = (t[0] * nprime) & mask32
        carry = a[0] * 0
        red = ()
        for j in tl.static_range(0, LIMBS):
            uv = t[j] + q * m[j] + carry
            low = uv & mask32
            carry = uv >> 32
            if j > 0:
                red += (low,)
        last = t[LIMBS] + carry
        red += (last & mask32,)
        t = red + (last >> 32,)

    res = ()
    for i in tl.static_range(0, LIMBS):
        res += (t[i],)
    reduced, _ = _sub_raw(res, m, LIMBS)
    need_reduce = (t[LIMBS] != 0) | _ge_tuple(res, m, LIMBS)
    return _select_tuple(need_reduce, reduced, res, LIMBS)


@triton.jit
def _reduce_step_mod(cur, m, HALF: tl.constexpr, LIMBS: tl.constexpr):
    lhs = ()
    rhs = ()
    for i in tl.static_range(0, LIMBS):
        pair = tl.reshape(cur[i], (HALF, 2))
        a, b = tl.split(pair)
        lhs += (a,)
        rhs += (b,)
    return _add_mod(lhs, rhs, m, LIMBS)


@triton.jit
def _sum_block_mod(value, mask, m, LIMBS: tl.constexpr):
    cur = ()
    for i in tl.static_range(0, LIMBS):
        cur += (tl.where(mask, value[i], 0),)
    cur = _reduce_step_mod(cur, m, HALF=64, LIMBS=LIMBS)
    cur = _reduce_step_mod(cur, m, HALF=32, LIMBS=LIMBS)
    cur = _reduce_step_mod(cur, m, HALF=16, LIMBS=LIMBS)
    cur = _reduce_step_mod(cur, m, HALF=8, LIMBS=LIMBS)
    cur = _reduce_step_mod(cur, m, HALF=4, LIMBS=LIMBS)
    cur = _reduce_step_mod(cur, m, HALF=2, LIMBS=LIMBS)
    return _reduce_step_mod(cur, m, HALF=1, LIMBS=LIMBS)


@triton.jit
def _store_eval_partial(partials_ptr, pid, total, mask, m, LIMBS: tl.constexpr):
    partial = _sum_block_mod(total, mask, m, LIMBS)
    for i in tl.static_range(0, LIMBS):
        tl.store(partials_ptr + pid * LIMBS + i, tl.sum(partial[i], axis=0).to(tl.uint32))


@triton.jit
def _load_interpolated_var(
    tables_ptr,
    twice,
    row_stride,
    mask,
    m,
    point,
    nprime,
    VAR_IDX: tl.constexpr,
    POINT_IDX: tl.constexpr,
    POINT_MODE: tl.constexpr,
    LIMBS: tl.constexpr,
    LAYOUT: tl.constexpr,
):
    base = tables_ptr + VAR_IDX * row_stride * LIMBS
    even = _load_table_elem(base, twice, row_stride, mask, LIMBS, LAYOUT)
    odd = _load_table_elem(base, twice + 1, row_stride, mask, LIMBS, LAYOUT)
    if POINT_MODE == 1 and POINT_IDX == 0:
        return even
    if POINT_MODE == 1 and POINT_IDX == 1:
        return odd

    diff = _sub_mod(odd, even, m, LIMBS)
    if POINT_MODE == 1 and POINT_IDX <= 15:
        scaled = diff
        for _ in tl.static_range(1, POINT_IDX):
            scaled = _add_mod(scaled, diff, m, LIMBS)
    else:
        scaled = _mont_mul(diff, point, m, nprime, LIMBS)
    return _add_mod(even, scaled, m, LIMBS)


@triton.jit(
    do_not_specialize=["half_n"], do_not_specialize_on_alignment=["tables_ptr", "partials_ptr"]
)
def _round_eval_specialized_kernel(
    tables_ptr,
    partials_ptr,
    half_n,
    modulus_ptr,
    point_monts_ptr,
    nprime,
    WORKLOAD: tl.constexpr,
    NUM_VARS: tl.constexpr,
    POINT_IDX: tl.constexpr,
    POINT_MODE: tl.constexpr,
    LIMBS: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = (pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)).to(tl.int64)
    mask = offs < half_n
    row_stride = half_n.to(tl.int64) * 2
    twice = 2 * offs
    m = _load_const(modulus_ptr, LIMBS)
    point = _load_const(point_monts_ptr + POINT_IDX * LIMBS, LIMBS)
    p = ()
    for var in tl.static_range(0, 22):
        value = _zero_tuple(offs, LIMBS)
        if var < NUM_VARS:
            value = _load_interpolated_var(
                tables_ptr,
                twice,
                row_stride,
                mask,
                m,
                point,
                nprime,
                var,
                POINT_IDX,
                POINT_MODE,
                LIMBS,
                LAYOUT,
            )
        p += (value,)
    if WORKLOAD <= 7:
        total = eval_poly_workload(WORKLOAD, p, m, nprime, LIMBS)
    else:
        total = eval_zk_workload(WORKLOAD, p, m, nprime, LIMBS)
    _store_eval_partial(partials_ptr, pid, total, mask, m, LIMBS)


@triton.jit(
    do_not_specialize=["n"], do_not_specialize_on_alignment=["partials_in_ptr", "partials_out_ptr"]
)
def _reduce_pairs_kernel(
    partials_in_ptr, partials_out_ptr, n, modulus_ptr, LIMBS: tl.constexpr, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_n = (n + 1) // 2
    mask = offs < out_n
    lhs = 2 * offs
    rhs = lhs + 1
    m = _load_const(modulus_ptr, LIMBS)
    a = _load_elem(partials_in_ptr, lhs, mask, LIMBS)
    b = _load_elem(partials_in_ptr, rhs, mask & (rhs < n), LIMBS)
    out = _add_mod(a, b, m, LIMBS)
    _store_elem(partials_out_ptr, offs, out, mask, LIMBS)


@triton.jit(do_not_specialize=["half_n"], do_not_specialize_on_alignment=["tables_ptr", "out_ptr"])
def _fold_kernel(
    tables_ptr,
    out_ptr,
    half_n,
    modulus_ptr,
    r_mont_ptr,
    nprime,
    NUM_VARS: tl.constexpr,
    LIMBS: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Widen before multiplying: pid * BLOCK_SIZE in int32 wraps once the grid covers more
    # than 2^31 elements (e.g. rounds=28 with 9 or more tables).
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    total = NUM_VARS * half_n.to(tl.int64)
    mask = offs < total
    var = offs // half_n
    local = offs - var * half_n
    row_stride = half_n.to(tl.int64) * 2
    m = _load_const(modulus_ptr, LIMBS)
    r_mont = _load_const(r_mont_ptr, LIMBS)
    base = tables_ptr + var * row_stride * LIMBS
    even = _load_table_elem(base, 2 * local, row_stride, mask, LIMBS, LAYOUT)
    odd = _load_table_elem(base, 2 * local + 1, row_stride, mask, LIMBS, LAYOUT)
    diff = _sub_mod(odd, even, m, LIMBS)
    folded = _add_mod(even, _mont_mul(diff, r_mont, m, nprime, LIMBS), m, LIMBS)
    out_var = out_ptr + var * half_n.to(tl.int64) * LIMBS
    _store_table_elem(out_var, local, half_n, folded, mask, LIMBS, LAYOUT)


@triton.jit(do_not_specialize=["total"], do_not_specialize_on_alignment=["tables_ptr"])
def _encode_tables_kernel(
    tables_ptr,
    total,
    n,
    modulus_ptr,
    r2_ptr,
    nprime,
    LIMBS: tl.constexpr,
    LAYOUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Widen before multiplying: pid * BLOCK_SIZE in int32 wraps once the grid covers more
    # than 2^31 elements (e.g. rounds=28 with 9 or more tables).
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE).to(tl.int64)
    mask = offs < total
    row_stride = n.to(tl.int64)
    var = offs // row_stride
    local = offs - var * row_stride
    base = tables_ptr + var * row_stride * LIMBS
    m = _load_const(modulus_ptr, LIMBS)
    r2 = _load_const(r2_ptr, LIMBS)
    raw = _load_table_elem(base, local, row_stride, mask, LIMBS, LAYOUT)
    encoded = _mont_mul(raw, r2, m, nprime, LIMBS)
    _store_table_elem(base, local, row_stride, encoded, mask, LIMBS, LAYOUT)


def _grid(n: int) -> tuple[int]:
    return (triton.cdiv(n, BLOCK),)


def _raw_limbs(value: int, limbs: int) -> tuple[int, ...]:
    return tuple((int(value) >> (32 * i)) & 0xFFFFFFFF for i in range(limbs))


def _to_limb_tensor(
    values: list[tuple[int, ...]], field: FieldSpec, device: torch.device
) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.uint32, device=device).contiguous()


def _make_tables(
    num_vars: int, n: int, field: FieldSpec, device: torch.device, seed: int, layout: str
) -> torch.Tensor:
    if layout == "element":
        tables = torch.zeros((num_vars, n, field.limbs), dtype=torch.uint32, device=device)
    else:
        tables = torch.zeros((num_vars, field.limbs, n), dtype=torch.uint32, device=device)
    offsets = torch.arange(1, n + 1, dtype=torch.int64, device=device)
    # Values stay within signed int64 for the configured sweeps. For 64/128/256-bit
    # fields they are also below the modulus; the 32-bit prime needs a setup-time
    # reduction before Montgomery encoding.
    for var in range(num_vars):
        values = seed + 7919 * (var + 1) + 104729 * offsets
        if field.bit_width == 32:
            values = torch.remainder(values, field.modulus)
        if layout == "element":
            tables[var, :, 0] = values.to(torch.uint32)
            if field.limbs > 1:
                tables[var, :, 1] = torch.div(values, 1 << 32, rounding_mode="floor").to(
                    torch.uint32
                )
        else:
            tables[var, 0, :] = values.to(torch.uint32)
            if field.limbs > 1:
                tables[var, 1, :] = torch.div(values, 1 << 32, rounding_mode="floor").to(
                    torch.uint32
                )
    return tables.contiguous()


def _reduce_partials(
    partials: torch.Tensor,
    partials_scratch: torch.Tensor,
    n: int,
    field: FieldSpec,
    modulus: torch.Tensor,
) -> torch.Tensor:
    src = partials
    dst = partials_scratch
    while n > 1:
        out_n = (n + 1) // 2
        _reduce_pairs_kernel[_grid(out_n)](
            src, dst, n, modulus, LIMBS=field.limbs, BLOCK_SIZE=BLOCK
        )
        src, dst = dst, src
        n = out_n
    return src


def _sha3_challenge_limbs(
    round_idx: int, coeffs: list[list[int]], field: FieldSpec
) -> tuple[int, ...]:
    hasher = hashlib.sha3_256()
    hasher.update(b"zkduel:field-sweep:v1")
    hasher.update(int(round_idx).to_bytes(4, "little", signed=False))
    for coeff in coeffs:
        for limb in coeff:
            hasher.update(int(limb).to_bytes(4, "little", signed=False))
    challenge = int.from_bytes(hasher.digest(), "little") % field.modulus
    return field.montgomery_limbs(challenge)


def _median(xs: list[float]) -> float:
    return statistics.median(xs) if xs else 0.0


def _launch_round_eval(
    workload: WorkloadSpec,
    grid: tuple[int],
    cur: torch.Tensor,
    partials: torch.Tensor,
    half: int,
    modulus: torch.Tensor,
    point_monts: torch.Tensor,
    field: FieldSpec,
    point_idx: int,
    point_mode: str,
    layout: str,
) -> None:
    _round_eval_specialized_kernel[grid](
        cur,
        partials,
        half,
        modulus,
        point_monts,
        field.nprime,
        WORKLOAD=workload.ident,
        NUM_VARS=workload.num_vars,
        POINT_IDX=point_idx,
        POINT_MODE=1 if point_mode == "specialized" else 0,
        LIMBS=field.limbs,
        LAYOUT=0 if layout == "element" else 1,
        BLOCK_SIZE=BLOCK,
    )


def _run_once(
    workload: WorkloadSpec,
    tables0: torch.Tensor,
    scratch_a: torch.Tensor,
    scratch_b: torch.Tensor,
    partials: torch.Tensor,
    partials_scratch: torch.Tensor,
    field: FieldSpec,
    modulus: torch.Tensor,
    point_monts: torch.Tensor,
    rounds: int,
    *,
    timing_mode: str,
    challenge_mode: str,
    layout: str,
    point_mode: str,
) -> dict[str, float | int]:
    num_vars = workload.num_vars
    degree = workload.degree
    cur = tables0
    write_to_a = True
    cur_n = 1 << rounds
    checksum = 0
    eval_ms = 0.0
    challenge_ms = 0.0
    fold_ms = 0.0

    for round_idx in range(rounds):
        half = cur_n // 2
        num_blocks = triton.cdiv(half, BLOCK)
        coeffs: list[list[int]] = []

        if timing_mode == "round-sum":
            torch.cuda.synchronize()
            eval_t0 = time.perf_counter()

        for point_idx in range(degree + 1):
            _launch_round_eval(
                workload,
                (num_blocks,),
                cur,
                partials,
                half,
                modulus,
                point_monts,
                field,
                point_idx,
                point_mode,
                layout,
            )
            reduced = _reduce_partials(partials, partials_scratch, num_blocks, field, modulus)
            host_limbs = reduced[0, : field.limbs].detach().cpu().tolist()
            coeffs.append([int(v) for v in host_limbs])
            checksum += sum((i + 1) * int(v) for i, v in enumerate(host_limbs))

        if timing_mode == "round-sum":
            torch.cuda.synchronize()
            eval_ms += (time.perf_counter() - eval_t0) * 1000.0

        if round_idx + 1 < rounds:
            if timing_mode == "round-sum":
                challenge_t0 = time.perf_counter()
            if challenge_mode == "sha3":
                challenge_limbs = _sha3_challenge_limbs(round_idx, coeffs, field)
                r_mont = _to_limb_tensor([challenge_limbs], field, modulus.device).reshape(
                    field.limbs
                )
            else:
                r_mont = point_monts[1].contiguous()
            if timing_mode == "round-sum":
                challenge_ms += (time.perf_counter() - challenge_t0) * 1000.0

            total = num_vars * half
            nxt = scratch_a if write_to_a else scratch_b
            if timing_mode == "round-sum":
                torch.cuda.synchronize()
                fold_t0 = time.perf_counter()
            _fold_kernel[_grid(total)](
                cur,
                nxt,
                half,
                modulus,
                r_mont,
                field.nprime,
                NUM_VARS=num_vars,
                LIMBS=field.limbs,
                LAYOUT=0 if layout == "element" else 1,
                BLOCK_SIZE=BLOCK,
            )
            if timing_mode == "round-sum":
                torch.cuda.synchronize()
                fold_ms += (time.perf_counter() - fold_t0) * 1000.0
            cur = nxt
            write_to_a = not write_to_a
            cur_n = half

    return {
        "checksum": checksum,
        "eval_ms": eval_ms,
        "challenge_ms": challenge_ms,
        "fold_ms": fold_ms,
        "total_ms": eval_ms + challenge_ms + fold_ms,
    }


def _time_one(
    workload: str,
    bit_width: int,
    rounds: int,
    warmups: int,
    repeats: int,
    seed: int,
    timing_mode: str,
    challenge_mode: str,
    layout: str,
    point_mode: str,
):
    field = get_field(bit_width)
    spec = WORKLOAD_BY_NAME[workload]
    num_vars = spec.num_vars
    degree = spec.degree
    n = 1 << rounds
    device = torch.device("cuda")
    tables = _make_tables(num_vars, n, field, device, seed + rounds, layout)
    scratch_a = torch.empty_like(tables)
    scratch_b = torch.empty_like(tables)
    max_partials = triton.cdiv(n // 2, BLOCK)
    partials = torch.empty((max_partials, field.limbs), dtype=torch.uint32, device=device)
    partials_scratch = torch.empty_like(partials)
    modulus = _to_limb_tensor([_raw_limbs(field.modulus, field.limbs)], field, device).reshape(
        field.limbs
    )
    point_monts = _to_limb_tensor(
        [field.montgomery_limbs(i) for i in range(degree + 1)], field, device
    )
    r2 = _to_limb_tensor([field.to_limbs(field.r_mod * field.r_mod)], field, device).reshape(
        field.limbs
    )
    total_entries = num_vars * n
    _encode_tables_kernel[_grid(total_entries)](
        tables.reshape(-1, field.limbs),
        total_entries,
        n,
        modulus,
        r2,
        field.nprime,
        LIMBS=field.limbs,
        LAYOUT=0 if layout == "element" else 1,
        BLOCK_SIZE=BLOCK,
    )

    torch.cuda.synchronize()
    if timing_mode == "whole-run":
        t0 = time.perf_counter()
        first_run = _run_once(
            spec,
            tables,
            scratch_a,
            scratch_b,
            partials,
            partials_scratch,
            field,
            modulus,
            point_monts,
            rounds,
            timing_mode=timing_mode,
            challenge_mode=challenge_mode,
            layout=layout,
            point_mode=point_mode,
        )
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - t0) * 1000.0
    else:
        first_run = _run_once(
            spec,
            tables,
            scratch_a,
            scratch_b,
            partials,
            partials_scratch,
            field,
            modulus,
            point_monts,
            rounds,
            timing_mode=timing_mode,
            challenge_mode=challenge_mode,
            layout=layout,
            point_mode=point_mode,
        )
        first_ms = float(first_run["total_ms"])

    checksum = int(first_run["checksum"])
    for _ in range(warmups):
        warm_run = _run_once(
            spec,
            tables,
            scratch_a,
            scratch_b,
            partials,
            partials_scratch,
            field,
            modulus,
            point_monts,
            rounds,
            timing_mode=timing_mode,
            challenge_mode=challenge_mode,
            layout=layout,
            point_mode=point_mode,
        )
        checksum += int(warm_run["checksum"])
    torch.cuda.synchronize()

    samples = []
    eval_samples = []
    challenge_samples = []
    fold_samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        if timing_mode == "whole-run":
            t0 = time.perf_counter()
            run = _run_once(
                spec,
                tables,
                scratch_a,
                scratch_b,
                partials,
                partials_scratch,
                field,
                modulus,
                point_monts,
                rounds,
                timing_mode=timing_mode,
                challenge_mode=challenge_mode,
                layout=layout,
                point_mode=point_mode,
            )
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - t0) * 1000.0)
        else:
            run = _run_once(
                spec,
                tables,
                scratch_a,
                scratch_b,
                partials,
                partials_scratch,
                field,
                modulus,
                point_monts,
                rounds,
                timing_mode=timing_mode,
                challenge_mode=challenge_mode,
                layout=layout,
                point_mode=point_mode,
            )
            samples.append(float(run["total_ms"]))
            eval_samples.append(float(run["eval_ms"]))
            challenge_samples.append(float(run["challenge_ms"]))
            fold_samples.append(float(run["fold_ms"]))
        checksum += int(run["checksum"])

    median_ms = statistics.median(samples)
    eval_median_ms = _median(eval_samples)
    challenge_median_ms = _median(challenge_samples)
    fold_median_ms = _median(fold_samples)
    input_bytes = num_vars * n * field.limbs * 4
    return {
        "workload": workload,
        "polynomial": spec.polynomial,
        "backend": f"triton-sumcheck-prime-field-montgomery-{layout}-layout-{point_mode}-points",
        "field": field.name,
        "arithmetic": "montgomery",
        "layout": layout,
        "point_mode": point_mode,
        "bit_width": bit_width,
        "limbs": field.limbs,
        "modulus_hex": hex(field.modulus),
        "rounds": rounds,
        "N": n,
        "vars": num_vars,
        "degree": degree,
        "terms": spec.terms,
        "warmups": warmups,
        "repeats": repeats,
        "timing_mode": timing_mode,
        "challenge_mode": challenge_mode,
        "first_ms": f"{first_ms:.3f}",
        "median_ms": f"{median_ms:.3f}",
        "eval_median_ms": f"{eval_median_ms:.3f}",
        "challenge_median_ms": f"{challenge_median_ms:.6f}",
        "fold_median_ms": f"{fold_median_ms:.3f}",
        "total_median_ms": f"{median_ms:.3f}",
        "input_mib": f"{input_bytes / (1024**2):.6f}",
        "checksum": checksum,
        "status": "ok",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit-widths", type=int, nargs="+", default=sorted(FIELD_SPECS))
    ap.add_argument("--workloads", nargs="+", default=None)
    ap.add_argument(
        "--zk-only",
        action="store_true",
        help="Restrict default workload set to the 25 ZK workloads "
        "(no effect if --workloads is given)",
    )
    ap.add_argument("--rounds", type=int, nargs="+", default=[8, 10, 12, 14, 16])
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timing-mode", choices=["whole-run", "round-sum"], default="whole-run")
    ap.add_argument("--challenge-mode", choices=["fixed", "sha3"], default="fixed")
    ap.add_argument("--layout", choices=["element", "limb"], default="element")
    ap.add_argument("--point-mode", choices=["generic", "specialized"], default="generic")
    ap.add_argument("--out", default="results/triton_field_sweep.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmarks/field_sweep/triton_sweep.py")

    if args.workloads is None:
        args.workloads = list(ZK_WORKLOAD_NAMES) if args.zk_only else list(WORKLOAD_NAMES)
    args.workloads = canonicalize_workload_names(args.workloads)
    unknown = sorted(set(args.workloads) - set(WORKLOAD_BY_NAME))
    if unknown:
        known = ", ".join(known_workload_names())
        raise SystemExit(f"unknown workloads: {unknown}; known: {known}")
    for bit_width in args.bit_widths:
        get_field(bit_width)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload",
        "polynomial",
        "backend",
        "field",
        "arithmetic",
        "layout",
        "point_mode",
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
        "first_ms",
        "median_ms",
        "eval_median_ms",
        "challenge_median_ms",
        "fold_median_ms",
        "total_median_ms",
        "input_mib",
        "checksum",
        "status",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for workload in args.workloads:
            for rounds in args.rounds:
                for bit_width in args.bit_widths:
                    row = _time_one(
                        workload,
                        bit_width,
                        rounds,
                        warmups=args.warmups,
                        repeats=args.repeats,
                        seed=args.seed,
                        timing_mode=args.timing_mode,
                        challenge_mode=args.challenge_mode,
                        layout=args.layout,
                        point_mode=args.point_mode,
                    )
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"{workload:24s} field={row['field']:28s} rounds={rounds:<2d} "
                        f"first_ms={row['first_ms']} median_ms={row['median_ms']} "
                        f"checksum={row['checksum']}",
                        flush=True,
                    )

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
