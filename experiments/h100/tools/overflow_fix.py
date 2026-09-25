"""Fixed copies of triton_sweep's _encode_tables_kernel and _fold_kernel (int64 offsets).

Both kernels compute `(pid * BLOCK_SIZE + tl.arange(...)).to(tl.int64)`: the multiply
happens in int32 and wraps once the grid covers more than 2^31 elements. The encode grid
spans all vars x 2^rounds elements, so at rounds=28 any workload with 9 or more tables
(vars x 2^28 > 2^31) reads and writes before the start of the table tensor: an illegal
memory access on H100, and crashes or silently wrong checksums in the A100 sweep
(zk_vanilla_zerocheck_hp at 32/64 bits, r=28). The fold grid spans vars x 2^(rounds-1)
elements, which overflows only for 17+ tables at r=28 (zk_jellyfish_zerocheck_hp).

The fix widens pid before the multiply. The eval kernel is unaffected (its grid covers
half_n < 2^31 elements) and is not replaced, so its compiled kernels stay valid.

Usage: import this module and call install() before the first kernel launch.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for p in (ROOT / "benchmarks" / "field_sweep", ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import triton  # noqa: E402
import triton.language as tl  # noqa: E402
import triton_sweep as ts  # noqa: E402
from triton_sweep import (  # noqa: E402
    _add_mod,
    _load_const,
    _load_table_elem,
    _mont_mul,
    _store_table_elem,
    _sub_mod,
)


@triton.jit(do_not_specialize=["half_n"], do_not_specialize_on_alignment=["tables_ptr", "out_ptr"])
def fold_kernel_fixed(
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
def encode_tables_kernel_fixed(
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


def install() -> None:
    ts._encode_tables_kernel = encode_tables_kernel_fixed
    ts._fold_kernel = fold_kernel_fixed
