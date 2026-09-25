"""Triton uint256 limb kernels.

Representation: tensors have shape ``(n, 8)``, dtype ``torch.uint32``, and store
little-endian 32-bit limbs. These kernels are a foundation for experimenting
with multiprecision arithmetic in Triton. They are intentionally standalone and
not yet wired into SumCheck field arithmetic.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import triton
import triton.language as tl

LIMBS = 8
LIMB_BITS = 32
LIMB_BASE = 1 << LIMB_BITS
LIMB_MASK = LIMB_BASE - 1
UINT256_MASK = (1 << 256) - 1
_BLOCK = 256


@triton.jit
def _load_u256(ptr, offs):
    return (
        tl.load(ptr + offs * 8 + 0).to(tl.uint64),
        tl.load(ptr + offs * 8 + 1).to(tl.uint64),
        tl.load(ptr + offs * 8 + 2).to(tl.uint64),
        tl.load(ptr + offs * 8 + 3).to(tl.uint64),
        tl.load(ptr + offs * 8 + 4).to(tl.uint64),
        tl.load(ptr + offs * 8 + 5).to(tl.uint64),
        tl.load(ptr + offs * 8 + 6).to(tl.uint64),
        tl.load(ptr + offs * 8 + 7).to(tl.uint64),
    )


@triton.jit
def _load_modulus(ptr):
    return (
        tl.load(ptr + 0).to(tl.uint64),
        tl.load(ptr + 1).to(tl.uint64),
        tl.load(ptr + 2).to(tl.uint64),
        tl.load(ptr + 3).to(tl.uint64),
        tl.load(ptr + 4).to(tl.uint64),
        tl.load(ptr + 5).to(tl.uint64),
        tl.load(ptr + 6).to(tl.uint64),
        tl.load(ptr + 7).to(tl.uint64),
    )


@triton.jit
def _store_u256(ptr, offs, limbs, mask):
    tl.store(ptr + offs * 8 + 0, limbs[0], mask=mask)
    tl.store(ptr + offs * 8 + 1, limbs[1], mask=mask)
    tl.store(ptr + offs * 8 + 2, limbs[2], mask=mask)
    tl.store(ptr + offs * 8 + 3, limbs[3], mask=mask)
    tl.store(ptr + offs * 8 + 4, limbs[4], mask=mask)
    tl.store(ptr + offs * 8 + 5, limbs[5], mask=mask)
    tl.store(ptr + offs * 8 + 6, limbs[6], mask=mask)
    tl.store(ptr + offs * 8 + 7, limbs[7], mask=mask)


@triton.jit
def _cmp_limbs(a, b):
    eq = a[0] == a[0]
    gt = a[0] != a[0]
    lt = a[0] != a[0]
    for i in tl.static_range(7, -1, -1):
        ai = a[i]
        bi = b[i]
        gt = gt | (eq & (ai > bi))
        lt = lt | (eq & (ai < bi))
        eq = eq & (ai == bi)
    return gt, eq, lt


@triton.jit
def _add_limbs(a, b):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    carry = a[0] * 0
    s0 = a[0] + b[0] + carry
    o0 = s0 & mask32
    carry = s0 >> 32
    s1 = a[1] + b[1] + carry
    o1 = s1 & mask32
    carry = s1 >> 32
    s2 = a[2] + b[2] + carry
    o2 = s2 & mask32
    carry = s2 >> 32
    s3 = a[3] + b[3] + carry
    o3 = s3 & mask32
    carry = s3 >> 32
    s4 = a[4] + b[4] + carry
    o4 = s4 & mask32
    carry = s4 >> 32
    s5 = a[5] + b[5] + carry
    o5 = s5 & mask32
    carry = s5 >> 32
    s6 = a[6] + b[6] + carry
    o6 = s6 & mask32
    carry = s6 >> 32
    s7 = a[7] + b[7] + carry
    o7 = s7 & mask32
    carry = s7 >> 32
    return (o0, o1, o2, o3, o4, o5, o6, o7), carry


@triton.jit
def _sub_limbs(a, b):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    borrow = a[0] * 0
    subtrahend0 = b[0] + borrow
    next_borrow0 = subtrahend0 > a[0]
    d0 = a[0] - subtrahend0
    o0 = d0 & mask32
    borrow = next_borrow0.to(tl.uint64)
    subtrahend1 = b[1] + borrow
    next_borrow1 = subtrahend1 > a[1]
    d1 = a[1] - subtrahend1
    o1 = d1 & mask32
    borrow = next_borrow1.to(tl.uint64)
    subtrahend2 = b[2] + borrow
    next_borrow2 = subtrahend2 > a[2]
    d2 = a[2] - subtrahend2
    o2 = d2 & mask32
    borrow = next_borrow2.to(tl.uint64)
    subtrahend3 = b[3] + borrow
    next_borrow3 = subtrahend3 > a[3]
    d3 = a[3] - subtrahend3
    o3 = d3 & mask32
    borrow = next_borrow3.to(tl.uint64)
    subtrahend4 = b[4] + borrow
    next_borrow4 = subtrahend4 > a[4]
    d4 = a[4] - subtrahend4
    o4 = d4 & mask32
    borrow = next_borrow4.to(tl.uint64)
    subtrahend5 = b[5] + borrow
    next_borrow5 = subtrahend5 > a[5]
    d5 = a[5] - subtrahend5
    o5 = d5 & mask32
    borrow = next_borrow5.to(tl.uint64)
    subtrahend6 = b[6] + borrow
    next_borrow6 = subtrahend6 > a[6]
    d6 = a[6] - subtrahend6
    o6 = d6 & mask32
    borrow = next_borrow6.to(tl.uint64)
    subtrahend7 = b[7] + borrow
    next_borrow7 = subtrahend7 > a[7]
    d7 = a[7] - subtrahend7
    o7 = d7 & mask32
    borrow = next_borrow7.to(tl.uint64)
    return (o0, o1, o2, o3, o4, o5, o6, o7), borrow


@triton.jit
def _mul_lo_limbs(a, b):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    r0 = a[0] * 0
    r1 = a[0] * 0
    r2 = a[0] * 0
    r3 = a[0] * 0
    r4 = a[0] * 0
    r5 = a[0] * 0
    r6 = a[0] * 0
    r7 = a[0] * 0
    carry = a[0] * 0
    ai0 = a[0]
    uv_0_0 = r0 + ai0 * b[0] + carry
    r0 = uv_0_0 & mask32
    carry = uv_0_0 >> 32
    uv_0_1 = r1 + ai0 * b[1] + carry
    r1 = uv_0_1 & mask32
    carry = uv_0_1 >> 32
    uv_0_2 = r2 + ai0 * b[2] + carry
    r2 = uv_0_2 & mask32
    carry = uv_0_2 >> 32
    uv_0_3 = r3 + ai0 * b[3] + carry
    r3 = uv_0_3 & mask32
    carry = uv_0_3 >> 32
    uv_0_4 = r4 + ai0 * b[4] + carry
    r4 = uv_0_4 & mask32
    carry = uv_0_4 >> 32
    uv_0_5 = r5 + ai0 * b[5] + carry
    r5 = uv_0_5 & mask32
    carry = uv_0_5 >> 32
    uv_0_6 = r6 + ai0 * b[6] + carry
    r6 = uv_0_6 & mask32
    carry = uv_0_6 >> 32
    uv_0_7 = r7 + ai0 * b[7] + carry
    r7 = uv_0_7 & mask32
    carry = uv_0_7 >> 32
    carry = a[0] * 0
    ai1 = a[1]
    uv_1_0 = r1 + ai1 * b[0] + carry
    r1 = uv_1_0 & mask32
    carry = uv_1_0 >> 32
    uv_1_1 = r2 + ai1 * b[1] + carry
    r2 = uv_1_1 & mask32
    carry = uv_1_1 >> 32
    uv_1_2 = r3 + ai1 * b[2] + carry
    r3 = uv_1_2 & mask32
    carry = uv_1_2 >> 32
    uv_1_3 = r4 + ai1 * b[3] + carry
    r4 = uv_1_3 & mask32
    carry = uv_1_3 >> 32
    uv_1_4 = r5 + ai1 * b[4] + carry
    r5 = uv_1_4 & mask32
    carry = uv_1_4 >> 32
    uv_1_5 = r6 + ai1 * b[5] + carry
    r6 = uv_1_5 & mask32
    carry = uv_1_5 >> 32
    uv_1_6 = r7 + ai1 * b[6] + carry
    r7 = uv_1_6 & mask32
    carry = uv_1_6 >> 32
    carry = a[0] * 0
    ai2 = a[2]
    uv_2_0 = r2 + ai2 * b[0] + carry
    r2 = uv_2_0 & mask32
    carry = uv_2_0 >> 32
    uv_2_1 = r3 + ai2 * b[1] + carry
    r3 = uv_2_1 & mask32
    carry = uv_2_1 >> 32
    uv_2_2 = r4 + ai2 * b[2] + carry
    r4 = uv_2_2 & mask32
    carry = uv_2_2 >> 32
    uv_2_3 = r5 + ai2 * b[3] + carry
    r5 = uv_2_3 & mask32
    carry = uv_2_3 >> 32
    uv_2_4 = r6 + ai2 * b[4] + carry
    r6 = uv_2_4 & mask32
    carry = uv_2_4 >> 32
    uv_2_5 = r7 + ai2 * b[5] + carry
    r7 = uv_2_5 & mask32
    carry = uv_2_5 >> 32
    carry = a[0] * 0
    ai3 = a[3]
    uv_3_0 = r3 + ai3 * b[0] + carry
    r3 = uv_3_0 & mask32
    carry = uv_3_0 >> 32
    uv_3_1 = r4 + ai3 * b[1] + carry
    r4 = uv_3_1 & mask32
    carry = uv_3_1 >> 32
    uv_3_2 = r5 + ai3 * b[2] + carry
    r5 = uv_3_2 & mask32
    carry = uv_3_2 >> 32
    uv_3_3 = r6 + ai3 * b[3] + carry
    r6 = uv_3_3 & mask32
    carry = uv_3_3 >> 32
    uv_3_4 = r7 + ai3 * b[4] + carry
    r7 = uv_3_4 & mask32
    carry = uv_3_4 >> 32
    carry = a[0] * 0
    ai4 = a[4]
    uv_4_0 = r4 + ai4 * b[0] + carry
    r4 = uv_4_0 & mask32
    carry = uv_4_0 >> 32
    uv_4_1 = r5 + ai4 * b[1] + carry
    r5 = uv_4_1 & mask32
    carry = uv_4_1 >> 32
    uv_4_2 = r6 + ai4 * b[2] + carry
    r6 = uv_4_2 & mask32
    carry = uv_4_2 >> 32
    uv_4_3 = r7 + ai4 * b[3] + carry
    r7 = uv_4_3 & mask32
    carry = uv_4_3 >> 32
    carry = a[0] * 0
    ai5 = a[5]
    uv_5_0 = r5 + ai5 * b[0] + carry
    r5 = uv_5_0 & mask32
    carry = uv_5_0 >> 32
    uv_5_1 = r6 + ai5 * b[1] + carry
    r6 = uv_5_1 & mask32
    carry = uv_5_1 >> 32
    uv_5_2 = r7 + ai5 * b[2] + carry
    r7 = uv_5_2 & mask32
    carry = uv_5_2 >> 32
    carry = a[0] * 0
    ai6 = a[6]
    uv_6_0 = r6 + ai6 * b[0] + carry
    r6 = uv_6_0 & mask32
    carry = uv_6_0 >> 32
    uv_6_1 = r7 + ai6 * b[1] + carry
    r7 = uv_6_1 & mask32
    carry = uv_6_1 >> 32
    carry = a[0] * 0
    ai7 = a[7]
    uv_7_0 = r7 + ai7 * b[0] + carry
    r7 = uv_7_0 & mask32
    carry = uv_7_0 >> 32
    return (r0, r1, r2, r3, r4, r5, r6, r7)


@triton.jit
def _mont_mul_limbs(a, b, m, nprime):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    t0 = a[0] * 0
    t1 = a[0] * 0
    t2 = a[0] * 0
    t3 = a[0] * 0
    t4 = a[0] * 0
    t5 = a[0] * 0
    t6 = a[0] * 0
    t7 = a[0] * 0
    t8 = a[0] * 0
    t9 = a[0] * 0
    t10 = a[0] * 0
    t11 = a[0] * 0
    t12 = a[0] * 0
    t13 = a[0] * 0
    t14 = a[0] * 0
    t15 = a[0] * 0
    t16 = a[0] * 0
    # Full 8x8 -> 16-limb product, with one extra carry limb for reduction.
    carry = a[0] * 0
    ai0 = a[0]
    prod_uv_0_0 = t0 + ai0 * b[0] + carry
    t0 = prod_uv_0_0 & mask32
    carry = prod_uv_0_0 >> 32
    prod_uv_0_1 = t1 + ai0 * b[1] + carry
    t1 = prod_uv_0_1 & mask32
    carry = prod_uv_0_1 >> 32
    prod_uv_0_2 = t2 + ai0 * b[2] + carry
    t2 = prod_uv_0_2 & mask32
    carry = prod_uv_0_2 >> 32
    prod_uv_0_3 = t3 + ai0 * b[3] + carry
    t3 = prod_uv_0_3 & mask32
    carry = prod_uv_0_3 >> 32
    prod_uv_0_4 = t4 + ai0 * b[4] + carry
    t4 = prod_uv_0_4 & mask32
    carry = prod_uv_0_4 >> 32
    prod_uv_0_5 = t5 + ai0 * b[5] + carry
    t5 = prod_uv_0_5 & mask32
    carry = prod_uv_0_5 >> 32
    prod_uv_0_6 = t6 + ai0 * b[6] + carry
    t6 = prod_uv_0_6 & mask32
    carry = prod_uv_0_6 >> 32
    prod_uv_0_7 = t7 + ai0 * b[7] + carry
    t7 = prod_uv_0_7 & mask32
    carry = prod_uv_0_7 >> 32
    prod_carry_0_8 = t8 + carry
    t8 = prod_carry_0_8 & mask32
    carry = prod_carry_0_8 >> 32
    prod_carry_0_9 = t9 + carry
    t9 = prod_carry_0_9 & mask32
    carry = prod_carry_0_9 >> 32
    prod_carry_0_10 = t10 + carry
    t10 = prod_carry_0_10 & mask32
    carry = prod_carry_0_10 >> 32
    prod_carry_0_11 = t11 + carry
    t11 = prod_carry_0_11 & mask32
    carry = prod_carry_0_11 >> 32
    prod_carry_0_12 = t12 + carry
    t12 = prod_carry_0_12 & mask32
    carry = prod_carry_0_12 >> 32
    prod_carry_0_13 = t13 + carry
    t13 = prod_carry_0_13 & mask32
    carry = prod_carry_0_13 >> 32
    prod_carry_0_14 = t14 + carry
    t14 = prod_carry_0_14 & mask32
    carry = prod_carry_0_14 >> 32
    prod_carry_0_15 = t15 + carry
    t15 = prod_carry_0_15 & mask32
    carry = prod_carry_0_15 >> 32
    prod_carry_0_16 = t16 + carry
    t16 = prod_carry_0_16 & mask32
    carry = prod_carry_0_16 >> 32
    carry = a[0] * 0
    ai1 = a[1]
    prod_uv_1_0 = t1 + ai1 * b[0] + carry
    t1 = prod_uv_1_0 & mask32
    carry = prod_uv_1_0 >> 32
    prod_uv_1_1 = t2 + ai1 * b[1] + carry
    t2 = prod_uv_1_1 & mask32
    carry = prod_uv_1_1 >> 32
    prod_uv_1_2 = t3 + ai1 * b[2] + carry
    t3 = prod_uv_1_2 & mask32
    carry = prod_uv_1_2 >> 32
    prod_uv_1_3 = t4 + ai1 * b[3] + carry
    t4 = prod_uv_1_3 & mask32
    carry = prod_uv_1_3 >> 32
    prod_uv_1_4 = t5 + ai1 * b[4] + carry
    t5 = prod_uv_1_4 & mask32
    carry = prod_uv_1_4 >> 32
    prod_uv_1_5 = t6 + ai1 * b[5] + carry
    t6 = prod_uv_1_5 & mask32
    carry = prod_uv_1_5 >> 32
    prod_uv_1_6 = t7 + ai1 * b[6] + carry
    t7 = prod_uv_1_6 & mask32
    carry = prod_uv_1_6 >> 32
    prod_uv_1_7 = t8 + ai1 * b[7] + carry
    t8 = prod_uv_1_7 & mask32
    carry = prod_uv_1_7 >> 32
    prod_carry_1_9 = t9 + carry
    t9 = prod_carry_1_9 & mask32
    carry = prod_carry_1_9 >> 32
    prod_carry_1_10 = t10 + carry
    t10 = prod_carry_1_10 & mask32
    carry = prod_carry_1_10 >> 32
    prod_carry_1_11 = t11 + carry
    t11 = prod_carry_1_11 & mask32
    carry = prod_carry_1_11 >> 32
    prod_carry_1_12 = t12 + carry
    t12 = prod_carry_1_12 & mask32
    carry = prod_carry_1_12 >> 32
    prod_carry_1_13 = t13 + carry
    t13 = prod_carry_1_13 & mask32
    carry = prod_carry_1_13 >> 32
    prod_carry_1_14 = t14 + carry
    t14 = prod_carry_1_14 & mask32
    carry = prod_carry_1_14 >> 32
    prod_carry_1_15 = t15 + carry
    t15 = prod_carry_1_15 & mask32
    carry = prod_carry_1_15 >> 32
    prod_carry_1_16 = t16 + carry
    t16 = prod_carry_1_16 & mask32
    carry = prod_carry_1_16 >> 32
    carry = a[0] * 0
    ai2 = a[2]
    prod_uv_2_0 = t2 + ai2 * b[0] + carry
    t2 = prod_uv_2_0 & mask32
    carry = prod_uv_2_0 >> 32
    prod_uv_2_1 = t3 + ai2 * b[1] + carry
    t3 = prod_uv_2_1 & mask32
    carry = prod_uv_2_1 >> 32
    prod_uv_2_2 = t4 + ai2 * b[2] + carry
    t4 = prod_uv_2_2 & mask32
    carry = prod_uv_2_2 >> 32
    prod_uv_2_3 = t5 + ai2 * b[3] + carry
    t5 = prod_uv_2_3 & mask32
    carry = prod_uv_2_3 >> 32
    prod_uv_2_4 = t6 + ai2 * b[4] + carry
    t6 = prod_uv_2_4 & mask32
    carry = prod_uv_2_4 >> 32
    prod_uv_2_5 = t7 + ai2 * b[5] + carry
    t7 = prod_uv_2_5 & mask32
    carry = prod_uv_2_5 >> 32
    prod_uv_2_6 = t8 + ai2 * b[6] + carry
    t8 = prod_uv_2_6 & mask32
    carry = prod_uv_2_6 >> 32
    prod_uv_2_7 = t9 + ai2 * b[7] + carry
    t9 = prod_uv_2_7 & mask32
    carry = prod_uv_2_7 >> 32
    prod_carry_2_10 = t10 + carry
    t10 = prod_carry_2_10 & mask32
    carry = prod_carry_2_10 >> 32
    prod_carry_2_11 = t11 + carry
    t11 = prod_carry_2_11 & mask32
    carry = prod_carry_2_11 >> 32
    prod_carry_2_12 = t12 + carry
    t12 = prod_carry_2_12 & mask32
    carry = prod_carry_2_12 >> 32
    prod_carry_2_13 = t13 + carry
    t13 = prod_carry_2_13 & mask32
    carry = prod_carry_2_13 >> 32
    prod_carry_2_14 = t14 + carry
    t14 = prod_carry_2_14 & mask32
    carry = prod_carry_2_14 >> 32
    prod_carry_2_15 = t15 + carry
    t15 = prod_carry_2_15 & mask32
    carry = prod_carry_2_15 >> 32
    prod_carry_2_16 = t16 + carry
    t16 = prod_carry_2_16 & mask32
    carry = prod_carry_2_16 >> 32
    carry = a[0] * 0
    ai3 = a[3]
    prod_uv_3_0 = t3 + ai3 * b[0] + carry
    t3 = prod_uv_3_0 & mask32
    carry = prod_uv_3_0 >> 32
    prod_uv_3_1 = t4 + ai3 * b[1] + carry
    t4 = prod_uv_3_1 & mask32
    carry = prod_uv_3_1 >> 32
    prod_uv_3_2 = t5 + ai3 * b[2] + carry
    t5 = prod_uv_3_2 & mask32
    carry = prod_uv_3_2 >> 32
    prod_uv_3_3 = t6 + ai3 * b[3] + carry
    t6 = prod_uv_3_3 & mask32
    carry = prod_uv_3_3 >> 32
    prod_uv_3_4 = t7 + ai3 * b[4] + carry
    t7 = prod_uv_3_4 & mask32
    carry = prod_uv_3_4 >> 32
    prod_uv_3_5 = t8 + ai3 * b[5] + carry
    t8 = prod_uv_3_5 & mask32
    carry = prod_uv_3_5 >> 32
    prod_uv_3_6 = t9 + ai3 * b[6] + carry
    t9 = prod_uv_3_6 & mask32
    carry = prod_uv_3_6 >> 32
    prod_uv_3_7 = t10 + ai3 * b[7] + carry
    t10 = prod_uv_3_7 & mask32
    carry = prod_uv_3_7 >> 32
    prod_carry_3_11 = t11 + carry
    t11 = prod_carry_3_11 & mask32
    carry = prod_carry_3_11 >> 32
    prod_carry_3_12 = t12 + carry
    t12 = prod_carry_3_12 & mask32
    carry = prod_carry_3_12 >> 32
    prod_carry_3_13 = t13 + carry
    t13 = prod_carry_3_13 & mask32
    carry = prod_carry_3_13 >> 32
    prod_carry_3_14 = t14 + carry
    t14 = prod_carry_3_14 & mask32
    carry = prod_carry_3_14 >> 32
    prod_carry_3_15 = t15 + carry
    t15 = prod_carry_3_15 & mask32
    carry = prod_carry_3_15 >> 32
    prod_carry_3_16 = t16 + carry
    t16 = prod_carry_3_16 & mask32
    carry = prod_carry_3_16 >> 32
    carry = a[0] * 0
    ai4 = a[4]
    prod_uv_4_0 = t4 + ai4 * b[0] + carry
    t4 = prod_uv_4_0 & mask32
    carry = prod_uv_4_0 >> 32
    prod_uv_4_1 = t5 + ai4 * b[1] + carry
    t5 = prod_uv_4_1 & mask32
    carry = prod_uv_4_1 >> 32
    prod_uv_4_2 = t6 + ai4 * b[2] + carry
    t6 = prod_uv_4_2 & mask32
    carry = prod_uv_4_2 >> 32
    prod_uv_4_3 = t7 + ai4 * b[3] + carry
    t7 = prod_uv_4_3 & mask32
    carry = prod_uv_4_3 >> 32
    prod_uv_4_4 = t8 + ai4 * b[4] + carry
    t8 = prod_uv_4_4 & mask32
    carry = prod_uv_4_4 >> 32
    prod_uv_4_5 = t9 + ai4 * b[5] + carry
    t9 = prod_uv_4_5 & mask32
    carry = prod_uv_4_5 >> 32
    prod_uv_4_6 = t10 + ai4 * b[6] + carry
    t10 = prod_uv_4_6 & mask32
    carry = prod_uv_4_6 >> 32
    prod_uv_4_7 = t11 + ai4 * b[7] + carry
    t11 = prod_uv_4_7 & mask32
    carry = prod_uv_4_7 >> 32
    prod_carry_4_12 = t12 + carry
    t12 = prod_carry_4_12 & mask32
    carry = prod_carry_4_12 >> 32
    prod_carry_4_13 = t13 + carry
    t13 = prod_carry_4_13 & mask32
    carry = prod_carry_4_13 >> 32
    prod_carry_4_14 = t14 + carry
    t14 = prod_carry_4_14 & mask32
    carry = prod_carry_4_14 >> 32
    prod_carry_4_15 = t15 + carry
    t15 = prod_carry_4_15 & mask32
    carry = prod_carry_4_15 >> 32
    prod_carry_4_16 = t16 + carry
    t16 = prod_carry_4_16 & mask32
    carry = prod_carry_4_16 >> 32
    carry = a[0] * 0
    ai5 = a[5]
    prod_uv_5_0 = t5 + ai5 * b[0] + carry
    t5 = prod_uv_5_0 & mask32
    carry = prod_uv_5_0 >> 32
    prod_uv_5_1 = t6 + ai5 * b[1] + carry
    t6 = prod_uv_5_1 & mask32
    carry = prod_uv_5_1 >> 32
    prod_uv_5_2 = t7 + ai5 * b[2] + carry
    t7 = prod_uv_5_2 & mask32
    carry = prod_uv_5_2 >> 32
    prod_uv_5_3 = t8 + ai5 * b[3] + carry
    t8 = prod_uv_5_3 & mask32
    carry = prod_uv_5_3 >> 32
    prod_uv_5_4 = t9 + ai5 * b[4] + carry
    t9 = prod_uv_5_4 & mask32
    carry = prod_uv_5_4 >> 32
    prod_uv_5_5 = t10 + ai5 * b[5] + carry
    t10 = prod_uv_5_5 & mask32
    carry = prod_uv_5_5 >> 32
    prod_uv_5_6 = t11 + ai5 * b[6] + carry
    t11 = prod_uv_5_6 & mask32
    carry = prod_uv_5_6 >> 32
    prod_uv_5_7 = t12 + ai5 * b[7] + carry
    t12 = prod_uv_5_7 & mask32
    carry = prod_uv_5_7 >> 32
    prod_carry_5_13 = t13 + carry
    t13 = prod_carry_5_13 & mask32
    carry = prod_carry_5_13 >> 32
    prod_carry_5_14 = t14 + carry
    t14 = prod_carry_5_14 & mask32
    carry = prod_carry_5_14 >> 32
    prod_carry_5_15 = t15 + carry
    t15 = prod_carry_5_15 & mask32
    carry = prod_carry_5_15 >> 32
    prod_carry_5_16 = t16 + carry
    t16 = prod_carry_5_16 & mask32
    carry = prod_carry_5_16 >> 32
    carry = a[0] * 0
    ai6 = a[6]
    prod_uv_6_0 = t6 + ai6 * b[0] + carry
    t6 = prod_uv_6_0 & mask32
    carry = prod_uv_6_0 >> 32
    prod_uv_6_1 = t7 + ai6 * b[1] + carry
    t7 = prod_uv_6_1 & mask32
    carry = prod_uv_6_1 >> 32
    prod_uv_6_2 = t8 + ai6 * b[2] + carry
    t8 = prod_uv_6_2 & mask32
    carry = prod_uv_6_2 >> 32
    prod_uv_6_3 = t9 + ai6 * b[3] + carry
    t9 = prod_uv_6_3 & mask32
    carry = prod_uv_6_3 >> 32
    prod_uv_6_4 = t10 + ai6 * b[4] + carry
    t10 = prod_uv_6_4 & mask32
    carry = prod_uv_6_4 >> 32
    prod_uv_6_5 = t11 + ai6 * b[5] + carry
    t11 = prod_uv_6_5 & mask32
    carry = prod_uv_6_5 >> 32
    prod_uv_6_6 = t12 + ai6 * b[6] + carry
    t12 = prod_uv_6_6 & mask32
    carry = prod_uv_6_6 >> 32
    prod_uv_6_7 = t13 + ai6 * b[7] + carry
    t13 = prod_uv_6_7 & mask32
    carry = prod_uv_6_7 >> 32
    prod_carry_6_14 = t14 + carry
    t14 = prod_carry_6_14 & mask32
    carry = prod_carry_6_14 >> 32
    prod_carry_6_15 = t15 + carry
    t15 = prod_carry_6_15 & mask32
    carry = prod_carry_6_15 >> 32
    prod_carry_6_16 = t16 + carry
    t16 = prod_carry_6_16 & mask32
    carry = prod_carry_6_16 >> 32
    carry = a[0] * 0
    ai7 = a[7]
    prod_uv_7_0 = t7 + ai7 * b[0] + carry
    t7 = prod_uv_7_0 & mask32
    carry = prod_uv_7_0 >> 32
    prod_uv_7_1 = t8 + ai7 * b[1] + carry
    t8 = prod_uv_7_1 & mask32
    carry = prod_uv_7_1 >> 32
    prod_uv_7_2 = t9 + ai7 * b[2] + carry
    t9 = prod_uv_7_2 & mask32
    carry = prod_uv_7_2 >> 32
    prod_uv_7_3 = t10 + ai7 * b[3] + carry
    t10 = prod_uv_7_3 & mask32
    carry = prod_uv_7_3 >> 32
    prod_uv_7_4 = t11 + ai7 * b[4] + carry
    t11 = prod_uv_7_4 & mask32
    carry = prod_uv_7_4 >> 32
    prod_uv_7_5 = t12 + ai7 * b[5] + carry
    t12 = prod_uv_7_5 & mask32
    carry = prod_uv_7_5 >> 32
    prod_uv_7_6 = t13 + ai7 * b[6] + carry
    t13 = prod_uv_7_6 & mask32
    carry = prod_uv_7_6 >> 32
    prod_uv_7_7 = t14 + ai7 * b[7] + carry
    t14 = prod_uv_7_7 & mask32
    carry = prod_uv_7_7 >> 32
    prod_carry_7_15 = t15 + carry
    t15 = prod_carry_7_15 & mask32
    carry = prod_carry_7_15 >> 32
    prod_carry_7_16 = t16 + carry
    t16 = prod_carry_7_16 & mask32
    carry = prod_carry_7_16 >> 32
    # Montgomery REDC, base 2^32, R = 2^256.
    q0 = (t0 * nprime) & mask32
    carry = a[0] * 0
    red_uv_0_0 = t0 + q0 * m[0] + carry
    t0 = red_uv_0_0 & mask32
    carry = red_uv_0_0 >> 32
    red_uv_0_1 = t1 + q0 * m[1] + carry
    t1 = red_uv_0_1 & mask32
    carry = red_uv_0_1 >> 32
    red_uv_0_2 = t2 + q0 * m[2] + carry
    t2 = red_uv_0_2 & mask32
    carry = red_uv_0_2 >> 32
    red_uv_0_3 = t3 + q0 * m[3] + carry
    t3 = red_uv_0_3 & mask32
    carry = red_uv_0_3 >> 32
    red_uv_0_4 = t4 + q0 * m[4] + carry
    t4 = red_uv_0_4 & mask32
    carry = red_uv_0_4 >> 32
    red_uv_0_5 = t5 + q0 * m[5] + carry
    t5 = red_uv_0_5 & mask32
    carry = red_uv_0_5 >> 32
    red_uv_0_6 = t6 + q0 * m[6] + carry
    t6 = red_uv_0_6 & mask32
    carry = red_uv_0_6 >> 32
    red_uv_0_7 = t7 + q0 * m[7] + carry
    t7 = red_uv_0_7 & mask32
    carry = red_uv_0_7 >> 32
    red_carry_0_8 = t8 + carry
    t8 = red_carry_0_8 & mask32
    carry = red_carry_0_8 >> 32
    red_carry_0_9 = t9 + carry
    t9 = red_carry_0_9 & mask32
    carry = red_carry_0_9 >> 32
    red_carry_0_10 = t10 + carry
    t10 = red_carry_0_10 & mask32
    carry = red_carry_0_10 >> 32
    red_carry_0_11 = t11 + carry
    t11 = red_carry_0_11 & mask32
    carry = red_carry_0_11 >> 32
    red_carry_0_12 = t12 + carry
    t12 = red_carry_0_12 & mask32
    carry = red_carry_0_12 >> 32
    red_carry_0_13 = t13 + carry
    t13 = red_carry_0_13 & mask32
    carry = red_carry_0_13 >> 32
    red_carry_0_14 = t14 + carry
    t14 = red_carry_0_14 & mask32
    carry = red_carry_0_14 >> 32
    red_carry_0_15 = t15 + carry
    t15 = red_carry_0_15 & mask32
    carry = red_carry_0_15 >> 32
    red_carry_0_16 = t16 + carry
    t16 = red_carry_0_16 & mask32
    carry = red_carry_0_16 >> 32
    q1 = (t1 * nprime) & mask32
    carry = a[0] * 0
    red_uv_1_0 = t1 + q1 * m[0] + carry
    t1 = red_uv_1_0 & mask32
    carry = red_uv_1_0 >> 32
    red_uv_1_1 = t2 + q1 * m[1] + carry
    t2 = red_uv_1_1 & mask32
    carry = red_uv_1_1 >> 32
    red_uv_1_2 = t3 + q1 * m[2] + carry
    t3 = red_uv_1_2 & mask32
    carry = red_uv_1_2 >> 32
    red_uv_1_3 = t4 + q1 * m[3] + carry
    t4 = red_uv_1_3 & mask32
    carry = red_uv_1_3 >> 32
    red_uv_1_4 = t5 + q1 * m[4] + carry
    t5 = red_uv_1_4 & mask32
    carry = red_uv_1_4 >> 32
    red_uv_1_5 = t6 + q1 * m[5] + carry
    t6 = red_uv_1_5 & mask32
    carry = red_uv_1_5 >> 32
    red_uv_1_6 = t7 + q1 * m[6] + carry
    t7 = red_uv_1_6 & mask32
    carry = red_uv_1_6 >> 32
    red_uv_1_7 = t8 + q1 * m[7] + carry
    t8 = red_uv_1_7 & mask32
    carry = red_uv_1_7 >> 32
    red_carry_1_9 = t9 + carry
    t9 = red_carry_1_9 & mask32
    carry = red_carry_1_9 >> 32
    red_carry_1_10 = t10 + carry
    t10 = red_carry_1_10 & mask32
    carry = red_carry_1_10 >> 32
    red_carry_1_11 = t11 + carry
    t11 = red_carry_1_11 & mask32
    carry = red_carry_1_11 >> 32
    red_carry_1_12 = t12 + carry
    t12 = red_carry_1_12 & mask32
    carry = red_carry_1_12 >> 32
    red_carry_1_13 = t13 + carry
    t13 = red_carry_1_13 & mask32
    carry = red_carry_1_13 >> 32
    red_carry_1_14 = t14 + carry
    t14 = red_carry_1_14 & mask32
    carry = red_carry_1_14 >> 32
    red_carry_1_15 = t15 + carry
    t15 = red_carry_1_15 & mask32
    carry = red_carry_1_15 >> 32
    red_carry_1_16 = t16 + carry
    t16 = red_carry_1_16 & mask32
    carry = red_carry_1_16 >> 32
    q2 = (t2 * nprime) & mask32
    carry = a[0] * 0
    red_uv_2_0 = t2 + q2 * m[0] + carry
    t2 = red_uv_2_0 & mask32
    carry = red_uv_2_0 >> 32
    red_uv_2_1 = t3 + q2 * m[1] + carry
    t3 = red_uv_2_1 & mask32
    carry = red_uv_2_1 >> 32
    red_uv_2_2 = t4 + q2 * m[2] + carry
    t4 = red_uv_2_2 & mask32
    carry = red_uv_2_2 >> 32
    red_uv_2_3 = t5 + q2 * m[3] + carry
    t5 = red_uv_2_3 & mask32
    carry = red_uv_2_3 >> 32
    red_uv_2_4 = t6 + q2 * m[4] + carry
    t6 = red_uv_2_4 & mask32
    carry = red_uv_2_4 >> 32
    red_uv_2_5 = t7 + q2 * m[5] + carry
    t7 = red_uv_2_5 & mask32
    carry = red_uv_2_5 >> 32
    red_uv_2_6 = t8 + q2 * m[6] + carry
    t8 = red_uv_2_6 & mask32
    carry = red_uv_2_6 >> 32
    red_uv_2_7 = t9 + q2 * m[7] + carry
    t9 = red_uv_2_7 & mask32
    carry = red_uv_2_7 >> 32
    red_carry_2_10 = t10 + carry
    t10 = red_carry_2_10 & mask32
    carry = red_carry_2_10 >> 32
    red_carry_2_11 = t11 + carry
    t11 = red_carry_2_11 & mask32
    carry = red_carry_2_11 >> 32
    red_carry_2_12 = t12 + carry
    t12 = red_carry_2_12 & mask32
    carry = red_carry_2_12 >> 32
    red_carry_2_13 = t13 + carry
    t13 = red_carry_2_13 & mask32
    carry = red_carry_2_13 >> 32
    red_carry_2_14 = t14 + carry
    t14 = red_carry_2_14 & mask32
    carry = red_carry_2_14 >> 32
    red_carry_2_15 = t15 + carry
    t15 = red_carry_2_15 & mask32
    carry = red_carry_2_15 >> 32
    red_carry_2_16 = t16 + carry
    t16 = red_carry_2_16 & mask32
    carry = red_carry_2_16 >> 32
    q3 = (t3 * nprime) & mask32
    carry = a[0] * 0
    red_uv_3_0 = t3 + q3 * m[0] + carry
    t3 = red_uv_3_0 & mask32
    carry = red_uv_3_0 >> 32
    red_uv_3_1 = t4 + q3 * m[1] + carry
    t4 = red_uv_3_1 & mask32
    carry = red_uv_3_1 >> 32
    red_uv_3_2 = t5 + q3 * m[2] + carry
    t5 = red_uv_3_2 & mask32
    carry = red_uv_3_2 >> 32
    red_uv_3_3 = t6 + q3 * m[3] + carry
    t6 = red_uv_3_3 & mask32
    carry = red_uv_3_3 >> 32
    red_uv_3_4 = t7 + q3 * m[4] + carry
    t7 = red_uv_3_4 & mask32
    carry = red_uv_3_4 >> 32
    red_uv_3_5 = t8 + q3 * m[5] + carry
    t8 = red_uv_3_5 & mask32
    carry = red_uv_3_5 >> 32
    red_uv_3_6 = t9 + q3 * m[6] + carry
    t9 = red_uv_3_6 & mask32
    carry = red_uv_3_6 >> 32
    red_uv_3_7 = t10 + q3 * m[7] + carry
    t10 = red_uv_3_7 & mask32
    carry = red_uv_3_7 >> 32
    red_carry_3_11 = t11 + carry
    t11 = red_carry_3_11 & mask32
    carry = red_carry_3_11 >> 32
    red_carry_3_12 = t12 + carry
    t12 = red_carry_3_12 & mask32
    carry = red_carry_3_12 >> 32
    red_carry_3_13 = t13 + carry
    t13 = red_carry_3_13 & mask32
    carry = red_carry_3_13 >> 32
    red_carry_3_14 = t14 + carry
    t14 = red_carry_3_14 & mask32
    carry = red_carry_3_14 >> 32
    red_carry_3_15 = t15 + carry
    t15 = red_carry_3_15 & mask32
    carry = red_carry_3_15 >> 32
    red_carry_3_16 = t16 + carry
    t16 = red_carry_3_16 & mask32
    carry = red_carry_3_16 >> 32
    q4 = (t4 * nprime) & mask32
    carry = a[0] * 0
    red_uv_4_0 = t4 + q4 * m[0] + carry
    t4 = red_uv_4_0 & mask32
    carry = red_uv_4_0 >> 32
    red_uv_4_1 = t5 + q4 * m[1] + carry
    t5 = red_uv_4_1 & mask32
    carry = red_uv_4_1 >> 32
    red_uv_4_2 = t6 + q4 * m[2] + carry
    t6 = red_uv_4_2 & mask32
    carry = red_uv_4_2 >> 32
    red_uv_4_3 = t7 + q4 * m[3] + carry
    t7 = red_uv_4_3 & mask32
    carry = red_uv_4_3 >> 32
    red_uv_4_4 = t8 + q4 * m[4] + carry
    t8 = red_uv_4_4 & mask32
    carry = red_uv_4_4 >> 32
    red_uv_4_5 = t9 + q4 * m[5] + carry
    t9 = red_uv_4_5 & mask32
    carry = red_uv_4_5 >> 32
    red_uv_4_6 = t10 + q4 * m[6] + carry
    t10 = red_uv_4_6 & mask32
    carry = red_uv_4_6 >> 32
    red_uv_4_7 = t11 + q4 * m[7] + carry
    t11 = red_uv_4_7 & mask32
    carry = red_uv_4_7 >> 32
    red_carry_4_12 = t12 + carry
    t12 = red_carry_4_12 & mask32
    carry = red_carry_4_12 >> 32
    red_carry_4_13 = t13 + carry
    t13 = red_carry_4_13 & mask32
    carry = red_carry_4_13 >> 32
    red_carry_4_14 = t14 + carry
    t14 = red_carry_4_14 & mask32
    carry = red_carry_4_14 >> 32
    red_carry_4_15 = t15 + carry
    t15 = red_carry_4_15 & mask32
    carry = red_carry_4_15 >> 32
    red_carry_4_16 = t16 + carry
    t16 = red_carry_4_16 & mask32
    carry = red_carry_4_16 >> 32
    q5 = (t5 * nprime) & mask32
    carry = a[0] * 0
    red_uv_5_0 = t5 + q5 * m[0] + carry
    t5 = red_uv_5_0 & mask32
    carry = red_uv_5_0 >> 32
    red_uv_5_1 = t6 + q5 * m[1] + carry
    t6 = red_uv_5_1 & mask32
    carry = red_uv_5_1 >> 32
    red_uv_5_2 = t7 + q5 * m[2] + carry
    t7 = red_uv_5_2 & mask32
    carry = red_uv_5_2 >> 32
    red_uv_5_3 = t8 + q5 * m[3] + carry
    t8 = red_uv_5_3 & mask32
    carry = red_uv_5_3 >> 32
    red_uv_5_4 = t9 + q5 * m[4] + carry
    t9 = red_uv_5_4 & mask32
    carry = red_uv_5_4 >> 32
    red_uv_5_5 = t10 + q5 * m[5] + carry
    t10 = red_uv_5_5 & mask32
    carry = red_uv_5_5 >> 32
    red_uv_5_6 = t11 + q5 * m[6] + carry
    t11 = red_uv_5_6 & mask32
    carry = red_uv_5_6 >> 32
    red_uv_5_7 = t12 + q5 * m[7] + carry
    t12 = red_uv_5_7 & mask32
    carry = red_uv_5_7 >> 32
    red_carry_5_13 = t13 + carry
    t13 = red_carry_5_13 & mask32
    carry = red_carry_5_13 >> 32
    red_carry_5_14 = t14 + carry
    t14 = red_carry_5_14 & mask32
    carry = red_carry_5_14 >> 32
    red_carry_5_15 = t15 + carry
    t15 = red_carry_5_15 & mask32
    carry = red_carry_5_15 >> 32
    red_carry_5_16 = t16 + carry
    t16 = red_carry_5_16 & mask32
    carry = red_carry_5_16 >> 32
    q6 = (t6 * nprime) & mask32
    carry = a[0] * 0
    red_uv_6_0 = t6 + q6 * m[0] + carry
    t6 = red_uv_6_0 & mask32
    carry = red_uv_6_0 >> 32
    red_uv_6_1 = t7 + q6 * m[1] + carry
    t7 = red_uv_6_1 & mask32
    carry = red_uv_6_1 >> 32
    red_uv_6_2 = t8 + q6 * m[2] + carry
    t8 = red_uv_6_2 & mask32
    carry = red_uv_6_2 >> 32
    red_uv_6_3 = t9 + q6 * m[3] + carry
    t9 = red_uv_6_3 & mask32
    carry = red_uv_6_3 >> 32
    red_uv_6_4 = t10 + q6 * m[4] + carry
    t10 = red_uv_6_4 & mask32
    carry = red_uv_6_4 >> 32
    red_uv_6_5 = t11 + q6 * m[5] + carry
    t11 = red_uv_6_5 & mask32
    carry = red_uv_6_5 >> 32
    red_uv_6_6 = t12 + q6 * m[6] + carry
    t12 = red_uv_6_6 & mask32
    carry = red_uv_6_6 >> 32
    red_uv_6_7 = t13 + q6 * m[7] + carry
    t13 = red_uv_6_7 & mask32
    carry = red_uv_6_7 >> 32
    red_carry_6_14 = t14 + carry
    t14 = red_carry_6_14 & mask32
    carry = red_carry_6_14 >> 32
    red_carry_6_15 = t15 + carry
    t15 = red_carry_6_15 & mask32
    carry = red_carry_6_15 >> 32
    red_carry_6_16 = t16 + carry
    t16 = red_carry_6_16 & mask32
    carry = red_carry_6_16 >> 32
    q7 = (t7 * nprime) & mask32
    carry = a[0] * 0
    red_uv_7_0 = t7 + q7 * m[0] + carry
    t7 = red_uv_7_0 & mask32
    carry = red_uv_7_0 >> 32
    red_uv_7_1 = t8 + q7 * m[1] + carry
    t8 = red_uv_7_1 & mask32
    carry = red_uv_7_1 >> 32
    red_uv_7_2 = t9 + q7 * m[2] + carry
    t9 = red_uv_7_2 & mask32
    carry = red_uv_7_2 >> 32
    red_uv_7_3 = t10 + q7 * m[3] + carry
    t10 = red_uv_7_3 & mask32
    carry = red_uv_7_3 >> 32
    red_uv_7_4 = t11 + q7 * m[4] + carry
    t11 = red_uv_7_4 & mask32
    carry = red_uv_7_4 >> 32
    red_uv_7_5 = t12 + q7 * m[5] + carry
    t12 = red_uv_7_5 & mask32
    carry = red_uv_7_5 >> 32
    red_uv_7_6 = t13 + q7 * m[6] + carry
    t13 = red_uv_7_6 & mask32
    carry = red_uv_7_6 >> 32
    red_uv_7_7 = t14 + q7 * m[7] + carry
    t14 = red_uv_7_7 & mask32
    carry = red_uv_7_7 >> 32
    red_carry_7_15 = t15 + carry
    t15 = red_carry_7_15 & mask32
    carry = red_carry_7_15 >> 32
    red_carry_7_16 = t16 + carry
    t16 = red_carry_7_16 & mask32
    carry = red_carry_7_16 >> 32
    res = (t8, t9, t10, t11, t12, t13, t14, t15)
    gt, eq, _lt = _cmp_limbs(res, m)
    should_sub = (t16 != 0) | gt | eq
    reduced, _borrow = _sub_limbs(res, m)
    return (
        tl.where(should_sub, reduced[0], res[0]),
        tl.where(should_sub, reduced[1], res[1]),
        tl.where(should_sub, reduced[2], res[2]),
        tl.where(should_sub, reduced[3], res[3]),
        tl.where(should_sub, reduced[4], res[4]),
        tl.where(should_sub, reduced[5], res[5]),
        tl.where(should_sub, reduced[6], res[6]),
        tl.where(should_sub, reduced[7], res[7]),
    )


@triton.jit
def _mont_mul_kernel(a_ptr, b_ptr, modulus_ptr, nprime, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = _load_u256(a_ptr, offs)
    b = _load_u256(b_ptr, offs)
    m = _load_modulus(modulus_ptr)
    out = _mont_mul_limbs(a, b, m, nprime)
    _store_u256(out_ptr, offs, out, mask)


@triton.jit
def _add_kernel(a_ptr, b_ptr, out_ptr, carry_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = _load_u256(a_ptr, offs)
    b = _load_u256(b_ptr, offs)
    out, carry = _add_limbs(a, b)
    _store_u256(out_ptr, offs, out, mask)
    tl.store(carry_ptr + offs, carry, mask=mask)


@triton.jit
def _sub_kernel(a_ptr, b_ptr, out_ptr, borrow_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = _load_u256(a_ptr, offs)
    b = _load_u256(b_ptr, offs)
    out, borrow = _sub_limbs(a, b)
    _store_u256(out_ptr, offs, out, mask)
    tl.store(borrow_ptr + offs, borrow, mask=mask)


@triton.jit
def _cmp_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = _load_u256(a_ptr, offs)
    b = _load_u256(b_ptr, offs)
    gt, eq, lt = _cmp_limbs(a, b)
    result = tl.where(gt, 1, tl.where(lt, -1, 0))
    tl.store(out_ptr + offs, result, mask=mask)


@triton.jit
def _mul_lo_kernel(a_ptr, b_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = _load_u256(a_ptr, offs)
    b = _load_u256(b_ptr, offs)
    out = _mul_lo_limbs(a, b)
    _store_u256(out_ptr, offs, out, mask)


@triton.jit
def _add_mod_kernel(a_ptr, b_ptr, modulus_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = _load_u256(a_ptr, offs)
    b = _load_u256(b_ptr, offs)
    m = _load_modulus(modulus_ptr)
    s, carry = _add_limbs(a, b)
    gt, eq, _lt = _cmp_limbs(s, m)
    should_sub = (carry != 0) | gt | eq
    reduced, _borrow = _sub_limbs(s, m)
    out = (
        tl.where(should_sub, reduced[0], s[0]),
        tl.where(should_sub, reduced[1], s[1]),
        tl.where(should_sub, reduced[2], s[2]),
        tl.where(should_sub, reduced[3], s[3]),
        tl.where(should_sub, reduced[4], s[4]),
        tl.where(should_sub, reduced[5], s[5]),
        tl.where(should_sub, reduced[6], s[6]),
        tl.where(should_sub, reduced[7], s[7]),
    )
    _store_u256(out_ptr, offs, out, mask)


@triton.jit
def _sub_mod_kernel(a_ptr, b_ptr, modulus_ptr, out_ptr, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = _load_u256(a_ptr, offs)
    b = _load_u256(b_ptr, offs)
    m = _load_modulus(modulus_ptr)
    d, borrow = _sub_limbs(a, b)
    corrected, _carry = _add_limbs(d, m)
    should_add = borrow != 0
    out = (
        tl.where(should_add, corrected[0], d[0]),
        tl.where(should_add, corrected[1], d[1]),
        tl.where(should_add, corrected[2], d[2]),
        tl.where(should_add, corrected[3], d[3]),
        tl.where(should_add, corrected[4], d[4]),
        tl.where(should_add, corrected[5], d[5]),
        tl.where(should_add, corrected[6], d[6]),
        tl.where(should_add, corrected[7], d[7]),
    )
    _store_u256(out_ptr, offs, out, mask)


def _require_tensor(x: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if x.dtype is not torch.uint32:
        raise TypeError(f"{name} must have dtype torch.uint32, got {x.dtype}")
    if x.ndim != 2 or x.shape[1] != LIMBS:
        raise ValueError(f"{name} must have shape (n, {LIMBS}), got {tuple(x.shape)}")
    if not x.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    return x.contiguous()


def _require_modulus(
    modulus: torch.Tensor | int | Iterable[int], device: torch.device
) -> torch.Tensor:
    if isinstance(modulus, int):
        return to_uint256_tensor([modulus], device=device)[0].contiguous()
    if isinstance(modulus, torch.Tensor):
        if modulus.dtype is not torch.uint32:
            raise TypeError(f"modulus must have dtype torch.uint32, got {modulus.dtype}")
        if modulus.shape == (1, LIMBS):
            modulus = modulus[0]
        if modulus.shape != (LIMBS,):
            raise ValueError(f"modulus must have shape ({LIMBS},), got {tuple(modulus.shape)}")
        return modulus.to(device=device).contiguous()
    return to_uint256_tensor([limbs_to_int(modulus)], device=device)[0].contiguous()


def to_uint256_tensor(
    values: Iterable[int], *, device: torch.device | str | None = None
) -> torch.Tensor:
    rows: list[list[int]] = []
    for value in values:
        x = int(value)
        if x < 0 or x > UINT256_MASK:
            raise ValueError(f"uint256 value out of range: {x}")
        rows.append([(x >> (LIMB_BITS * i)) & LIMB_MASK for i in range(LIMBS)])
    return torch.tensor(rows, dtype=torch.uint32, device=device)


def limbs_to_int(limbs: Iterable[int]) -> int:
    out = 0
    for i, limb in enumerate(limbs):
        if i >= LIMBS:
            raise ValueError(f"too many limbs for uint256: expected {LIMBS}")
        x = int(limb)
        if x < 0 or x > LIMB_MASK:
            raise ValueError(f"limb out of range: {x}")
        out |= x << (LIMB_BITS * i)
    return out


def from_uint256_tensor(x: torch.Tensor) -> list[int]:
    if x.ndim != 2 or x.shape[1] != LIMBS:
        raise ValueError(f"x must have shape (n, {LIMBS}), got {tuple(x.shape)}")
    cpu = x.detach().to(device="cpu", dtype=torch.uint32).tolist()
    return [limbs_to_int(row) for row in cpu]


def montgomery_nprime(modulus: int) -> int:
    """Return ``-modulus^{-1} mod 2^32`` for an odd modulus."""
    m0 = int(modulus) & LIMB_MASK
    if m0 & 1 == 0:
        raise ValueError("Montgomery modulus must be odd")
    return (-pow(m0, -1, LIMB_BASE)) & LIMB_MASK


def montgomery_r_mod(modulus: int) -> int:
    """Return R mod modulus for R = 2^256."""
    return (1 << 256) % int(modulus)


def montgomery_encode_values(values: Iterable[int], modulus: int) -> list[int]:
    """Encode Python integers as Montgomery residues ``x * R mod modulus``."""
    m = int(modulus)
    r = montgomery_r_mod(m)
    return [(int(v) % m) * r % m for v in values]


def montgomery_decode_values(values: Iterable[int], modulus: int) -> list[int]:
    """Decode Montgomery residues back to standard representation."""
    m = int(modulus)
    r_inv = pow(1 << 256, -1, m)
    return [(int(v) % m) * r_inv % m for v in values]


def _launch_1d(n: int) -> tuple[int]:
    return ((n + _BLOCK - 1) // _BLOCK,)


def uint256_add(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = _require_tensor(a, "a")
    b = _require_tensor(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"a and b shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    n = a.shape[0]
    out = torch.empty_like(a)
    carry = torch.empty(n, dtype=torch.uint32, device=a.device)
    _add_kernel[_launch_1d(n)](a, b, out, carry, n, BLOCK=_BLOCK)
    return out, carry


def uint256_sub(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    a = _require_tensor(a, "a")
    b = _require_tensor(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"a and b shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    n = a.shape[0]
    out = torch.empty_like(a)
    borrow = torch.empty(n, dtype=torch.uint32, device=a.device)
    _sub_kernel[_launch_1d(n)](a, b, out, borrow, n, BLOCK=_BLOCK)
    return out, borrow


def uint256_cmp(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _require_tensor(a, "a")
    b = _require_tensor(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"a and b shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    n = a.shape[0]
    out = torch.empty(n, dtype=torch.int32, device=a.device)
    _cmp_kernel[_launch_1d(n)](a, b, out, n, BLOCK=_BLOCK)
    return out


def uint256_mul_lo(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = _require_tensor(a, "a")
    b = _require_tensor(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"a and b shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    n = a.shape[0]
    out = torch.empty_like(a)
    _mul_lo_kernel[_launch_1d(n)](a, b, out, n, BLOCK=_BLOCK)
    return out


def uint256_mont_mul(
    a: torch.Tensor,
    b: torch.Tensor,
    modulus: torch.Tensor | int | Iterable[int],
    *,
    nprime: int | None = None,
) -> torch.Tensor:
    """Montgomery multiply two uint256 tensors modulo ``modulus``.

    The operation computes ``a * b * R^{-1} mod modulus`` with ``R = 2^256``.
    Inputs and outputs are raw limb tensors; callers decide whether those limbs
    represent Montgomery residues. For field multiplication, pass Montgomery-
    encoded operands and the result remains Montgomery-encoded.
    """
    a = _require_tensor(a, "a")
    b = _require_tensor(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"a and b shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    if nprime is None:
        if not isinstance(modulus, int):
            raise ValueError("nprime must be provided when modulus is not an int")
        nprime = montgomery_nprime(modulus)
    if nprime < 0 or nprime > LIMB_MASK:
        raise ValueError(f"nprime out of 32-bit range: {nprime}")
    m = _require_modulus(modulus, a.device)
    n = a.shape[0]
    out = torch.empty_like(a)
    _mont_mul_kernel[_launch_1d(n)](a, b, m, int(nprime), out, n, BLOCK=_BLOCK)
    return out


def uint256_add_mod(
    a: torch.Tensor, b: torch.Tensor, modulus: torch.Tensor | int | Iterable[int]
) -> torch.Tensor:
    a = _require_tensor(a, "a")
    b = _require_tensor(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"a and b shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    m = _require_modulus(modulus, a.device)
    n = a.shape[0]
    out = torch.empty_like(a)
    _add_mod_kernel[_launch_1d(n)](a, b, m, out, n, BLOCK=_BLOCK)
    return out


def uint256_sub_mod(
    a: torch.Tensor, b: torch.Tensor, modulus: torch.Tensor | int | Iterable[int]
) -> torch.Tensor:
    a = _require_tensor(a, "a")
    b = _require_tensor(b, "b")
    if a.shape != b.shape:
        raise ValueError(f"a and b shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    m = _require_modulus(modulus, a.device)
    n = a.shape[0]
    out = torch.empty_like(a)
    _sub_mod_kernel[_launch_1d(n)](a, b, m, out, n, BLOCK=_BLOCK)
    return out
