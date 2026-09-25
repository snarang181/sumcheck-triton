"""Ablation: replace the field sweep's block reduction with tl.reduce.

triton_sweep._store_eval_partial reduces each 128-lane block with a reshape/split tree.
Triton lowers that tree to layouts with sizePerThread=[64, 2], [32, 2], ..., so every
thread holds the whole block and performs all 127 modular additions itself. This module
provides a drop-in _store_eval_partial that uses tl.reduce with a modular-add combine
function instead; the result is bit-identical (modular addition is associative and
commutative), only the reduction strategy changes.

tl.reduce combine functions take only the reduced values, so the modulus is baked in as
constexpr limbs. Each limb count in zkduel.fields.FIELD_SPECS belongs to exactly one field
(1: p32, 2: Goldilocks, 4: 2^127-1, 8: BLS12-381 r), which selects the constants.

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
from triton_sweep import _add_raw, _ge_tuple, _select_tuple, _sub_raw  # noqa: E402

from zkduel.fields import FIELD_SPECS  # noqa: E402


def _modulus_limbs(bit_width: int) -> list[tl.constexpr]:
    f = FIELD_SPECS[bit_width]
    return [tl.constexpr((f.modulus >> (32 * i)) & 0xFFFFFFFF) for i in range(f.limbs)]


(M1_0,) = _modulus_limbs(32)
M2_0, M2_1 = _modulus_limbs(64)
M4_0, M4_1, M4_2, M4_3 = _modulus_limbs(128)
M8_0, M8_1, M8_2, M8_3, M8_4, M8_5, M8_6, M8_7 = _modulus_limbs(256)


@triton.jit
def _addmod(a, b, m, LIMBS: tl.constexpr):
    s, carry = _add_raw(a, b, LIMBS)
    reduced, _ = _sub_raw(s, m, LIMBS)
    need = (carry != 0) | _ge_tuple(s, m, LIMBS)
    return _select_tuple(need, reduced, s, LIMBS)


@triton.jit
def _comb1(a0, b0):
    z = a0 * 0
    return _addmod((a0,), (b0,), (z + M1_0,), 1)


@triton.jit
def _comb2(a0, a1, b0, b1):
    z = a0 * 0
    return _addmod((a0, a1), (b0, b1), (z + M2_0, z + M2_1), 2)


@triton.jit
def _comb4(a0, a1, a2, a3, b0, b1, b2, b3):
    z = a0 * 0
    m = (z + M4_0, z + M4_1, z + M4_2, z + M4_3)
    return _addmod((a0, a1, a2, a3), (b0, b1, b2, b3), m, 4)


@triton.jit
def _comb8(a0, a1, a2, a3, a4, a5, a6, a7, b0, b1, b2, b3, b4, b5, b6, b7):
    z = a0 * 0
    m = (z + M8_0, z + M8_1, z + M8_2, z + M8_3, z + M8_4, z + M8_5, z + M8_6, z + M8_7)
    return _addmod((a0, a1, a2, a3, a4, a5, a6, a7), (b0, b1, b2, b3, b4, b5, b6, b7), m, 8)


@triton.jit
def store_eval_partial_treduce(partials_ptr, pid, total, mask, m, LIMBS: tl.constexpr):
    vals = ()
    for i in tl.static_range(0, LIMBS):
        vals += (tl.where(mask, total[i], 0),)
    if LIMBS == 1:
        red = tl.reduce(vals, 0, _comb1)
    elif LIMBS == 2:
        red = tl.reduce(vals, 0, _comb2)
    elif LIMBS == 4:
        red = tl.reduce(vals, 0, _comb4)
    else:
        red = tl.reduce(vals, 0, _comb8)
    for i in tl.static_range(0, LIMBS):
        tl.store(partials_ptr + pid * LIMBS + i, red[i].to(tl.uint32))


def install() -> None:
    """Route triton_sweep's eval kernel through the tl.reduce block reduction."""
    ts._store_eval_partial = store_eval_partial_treduce
    original = ts._time_one

    def _time_one(*args, **kwargs):
        row = original(*args, **kwargs)
        row["backend"] = row["backend"] + "-treduce"
        return row

    ts._time_one = _time_one
