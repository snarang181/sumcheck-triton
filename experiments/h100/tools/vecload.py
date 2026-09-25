"""Kernel-side remedy for the TritonGPUCoalesce compile cost (T12, T17): fewer, wider loads.

In the element-major layout, a table's even and odd elements for one thread are adjacent, so their
2 x LIMBS limbs are one contiguous run of 32-bit words. This variant loads that run with one 2-D
tl.load of shape [BLOCK, 2 * LIMBS] and splits it into limbs with tl.split, instead of issuing
2 x LIMBS scalar loads. The eval kernel's memory-op count M drops by a factor of 2 x LIMBS, and the
Coalesce pass, O(M N^2), should get cheaper by about that factor. Results are bit-identical.

install() routes triton_sweep's eval kernel through it (element layout only; limb-major is
unchanged).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks" / "field_sweep"), str(ROOT / "src")]

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton_sweep as ts  # noqa: E402
from triton_sweep import _add_mod, _load_table_elem, _mont_mul, _sub_mod  # noqa: E402,F401


@triton.jit
def _cols2(x):
    a, b = tl.split(x)
    return (a, b)


@triton.jit
def _interleave(ca, cb, HALF: tl.constexpr):
    out = ()
    for j in tl.static_range(0, HALF):
        out += (ca[j], cb[j])
    return out


@triton.jit
def _cols4(x):
    a, b = tl.split(tl.reshape(x, [x.shape[0], 2, 2]))
    return _interleave(_cols2(a), _cols2(b), 2)


@triton.jit
def _cols8(x):
    a, b = tl.split(tl.reshape(x, [x.shape[0], 4, 2]))
    return _interleave(_cols4(a), _cols4(b), 4)


@triton.jit
def _cols16(x):
    a, b = tl.split(tl.reshape(x, [x.shape[0], 8, 2]))
    return _interleave(_cols8(a), _cols8(b), 8)


@triton.jit
def _load_pair_vec(base, twice, mask, LIMBS: tl.constexpr):
    """Even and odd elements (2 * offs, 2 * offs + 1) of an element-major table, one 2-D load."""
    cols = tl.arange(0, 2 * LIMBS)
    addr = base + (twice.to(tl.int64) * LIMBS)[:, None] + cols[None, :]
    x = tl.load(addr, mask=mask[:, None], other=0)
    if LIMBS == 1:
        parts = _cols2(x)
    elif LIMBS == 2:
        parts = _cols4(x)
    elif LIMBS == 4:
        parts = _cols8(x)
    else:
        parts = _cols16(x)
    even = ()
    odd = ()
    for i in tl.static_range(0, LIMBS):
        even += (parts[i].to(tl.uint64),)
        odd += (parts[LIMBS + i].to(tl.uint64),)
    return even, odd


@triton.jit
def load_interpolated_var_vec(
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
    if LAYOUT == 0:
        even, odd = _load_pair_vec(base, twice, mask, LIMBS)
    else:
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


def install() -> None:
    """Route triton_sweep's eval kernel through the wide-load table reads."""
    ts._load_interpolated_var = load_interpolated_var_vec
    original = ts._time_one

    def _time_one(*args, **kwargs):
        row = original(*args, **kwargs)
        row["backend"] = row["backend"] + "-vecload"
        return row

    ts._time_one = _time_one
