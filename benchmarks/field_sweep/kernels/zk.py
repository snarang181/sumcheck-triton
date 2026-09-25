from __future__ import annotations

import triton
import triton.language as tl

from kernels.common import (
    add_mod,
    double_mod,
    mul3_mod,
    mul5_mod,
    mul7_mod,
    prod2,
    prod3,
    prod4,
    prod5,
    prod6,
    prod7,
    sub_mod,
)


@triton.jit
def _eval_zk_verifiable_asics(p, m, nprime, LIMBS: tl.constexpr):
    qadd_a = prod2(p[0], p[1], m, nprime, LIMBS)
    qadd_b = prod2(p[0], p[2], m, nprime, LIMBS)
    qmul_ab = prod3(p[3], p[1], p[2], m, nprime, LIMBS)
    return add_mod(add_mod(qadd_a, qadd_b, m, LIMBS), qmul_ab, m, LIMBS)


@triton.jit
def _eval_zk_spartan_1(p, m, nprime, LIMBS: tl.constexpr):
    abfz = prod3(p[0], p[1], p[2], m, nprime, LIMBS)
    cfz = prod2(p[3], p[2], m, nprime, LIMBS)
    return sub_mod(abfz, cfz, m, LIMBS)


@triton.jit
def _eval_zk_spartan_2(p, m, nprime, LIMBS: tl.constexpr):
    return prod2(p[0], p[1], m, nprime, LIMBS)


@triton.jit
def _eval_zk_witness_non_id(p, m, nprime, LIMBS: tl.constexpr):
    qyy = prod3(p[0], p[1], p[1], m, nprime, LIMBS)
    qxxx = prod4(p[0], p[2], p[2], p[2], m, nprime, LIMBS)
    return sub_mod(sub_mod(qyy, qxxx, m, LIMBS), mul5_mod(p[0], m, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_witness_id_point_1(p, m, nprime, LIMBS: tl.constexpr):
    qxyy = prod4(p[0], p[1], p[2], p[2], m, nprime, LIMBS)
    qxxxx = prod5(p[0], p[1], p[1], p[1], p[1], m, nprime, LIMBS)
    qx = prod2(p[0], p[1], m, nprime, LIMBS)
    return sub_mod(sub_mod(qxyy, qxxxx, m, LIMBS), mul5_mod(qx, m, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_witness_id_point_2(p, m, nprime, LIMBS: tl.constexpr):
    qyyy = prod4(p[0], p[1], p[1], p[1], m, nprime, LIMBS)
    qyxxx = prod5(p[0], p[1], p[2], p[2], p[2], m, nprime, LIMBS)
    qy = prod2(p[0], p[1], m, nprime, LIMBS)
    return sub_mod(sub_mod(qyyy, qyxxx, m, LIMBS), mul5_mod(qy, m, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_incomplete_add_1(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod4(p[0], p[1], p[2], p[2], m, nprime, LIMBS)
    acc = sub_mod(
        acc, double_mod(prod4(p[0], p[1], p[2], p[3], m, nprime, LIMBS), m, LIMBS), m, LIMBS
    )
    acc = add_mod(acc, prod4(p[0], p[1], p[3], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod4(p[0], p[3], p[2], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(
        acc, double_mod(prod4(p[0], p[3], p[3], p[2], m, nprime, LIMBS), m, LIMBS), m, LIMBS
    )
    acc = add_mod(acc, prod4(p[0], p[3], p[3], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod4(p[0], p[2], p[2], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(
        acc, double_mod(prod4(p[0], p[2], p[2], p[3], m, nprime, LIMBS), m, LIMBS), m, LIMBS
    )
    acc = add_mod(acc, prod4(p[0], p[2], p[3], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod3(p[0], p[4], p[4], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, double_mod(prod3(p[0], p[4], p[5], m, nprime, LIMBS), m, LIMBS), m, LIMBS)
    return sub_mod(acc, prod3(p[0], p[5], p[5], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_incomplete_add_2(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod3(p[0], p[1], p[2], m, nprime, LIMBS)
    acc = sub_mod(acc, prod3(p[0], p[1], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod3(p[0], p[4], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod3(p[0], p[4], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod3(p[0], p[5], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod3(p[0], p[5], p[6], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod3(p[0], p[4], p[6], m, nprime, LIMBS), m, LIMBS)
    return add_mod(acc, prod3(p[0], p[4], p[3], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_complete_add_1(p, m, nprime, LIMBS: tl.constexpr):
    q_xq_xq_l = prod4(p[0], p[1], p[1], p[3], m, nprime, LIMBS)
    q_xq_xp_l = prod4(p[0], p[1], p[2], p[3], m, nprime, LIMBS)
    q_xp_xp_l = prod4(p[0], p[2], p[2], p[3], m, nprime, LIMBS)
    q_xq_yq = prod3(p[0], p[1], p[4], m, nprime, LIMBS)
    q_xq_yp = prod3(p[0], p[1], p[5], m, nprime, LIMBS)
    q_xp_yq = prod3(p[0], p[2], p[4], m, nprime, LIMBS)
    q_xp_yp = prod3(p[0], p[2], p[5], m, nprime, LIMBS)
    acc = add_mod(q_xq_xq_l, q_xp_xp_l, m, LIMBS)
    acc = sub_mod(acc, double_mod(q_xq_xp_l, m, LIMBS), m, LIMBS)
    acc = sub_mod(acc, q_xq_yq, m, LIMBS)
    acc = add_mod(acc, q_xq_yp, m, LIMBS)
    acc = add_mod(acc, q_xp_yq, m, LIMBS)
    return sub_mod(acc, q_xp_yp, m, LIMBS)


@triton.jit
def _eval_zk_complete_add_2(p, m, nprime, LIMBS: tl.constexpr):
    acc = double_mod(prod3(p[0], p[1], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, mul3_mod(prod3(p[0], p[3], p[3], m, nprime, LIMBS), m, LIMBS), m, LIMBS)
    acc = sub_mod(
        acc, double_mod(prod5(p[0], p[4], p[1], p[2], p[5], m, nprime, LIMBS), m, LIMBS), m, LIMBS
    )
    acc = add_mod(
        acc, mul3_mod(prod5(p[0], p[4], p[3], p[3], p[5], m, nprime, LIMBS), m, LIMBS), m, LIMBS
    )
    acc = add_mod(
        acc, double_mod(prod5(p[0], p[3], p[1], p[2], p[5], m, nprime, LIMBS), m, LIMBS), m, LIMBS
    )
    return sub_mod(
        acc, mul3_mod(prod5(p[0], p[3], p[3], p[3], p[5], m, nprime, LIMBS), m, LIMBS), m, LIMBS
    )


@triton.jit
def _eval_zk_complete_add_3(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod6(p[0], p[1], p[2], p[2], p[3], p[3], m, nprime, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[4], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[1], p[2], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod6(p[0], p[1], p[1], p[2], p[3], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod5(p[0], p[1], p[1], p[1], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod5(p[0], p[1], p[1], p[2], p[4], m, nprime, LIMBS), m, LIMBS)
    return add_mod(acc, prod5(p[0], p[1], p[1], p[2], p[2], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_complete_add_4(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod6(p[0], p[1], p[1], p[2], p[2], p[3], m, nprime, LIMBS)
    acc = sub_mod(acc, prod6(p[0], p[1], p[2], p[2], p[4], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[5], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[6], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod6(p[0], p[1], p[1], p[1], p[2], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod6(p[0], p[1], p[1], p[2], p[4], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod5(p[0], p[1], p[1], p[2], p[5], m, nprime, LIMBS), m, LIMBS)
    return add_mod(acc, prod5(p[0], p[1], p[1], p[2], p[6], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_complete_add_5(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod6(p[0], p[1], p[2], p[3], p[4], p[4], m, nprime, LIMBS)
    acc = add_mod(acc, prod6(p[0], p[1], p[2], p[5], p[4], p[4], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[1], p[2], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[1], p[2], p[5], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[2], p[5], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[3], p[6], m, nprime, LIMBS), m, LIMBS)
    return sub_mod(acc, prod5(p[0], p[1], p[2], p[5], p[6], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_complete_add_6(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod6(p[0], p[1], p[1], p[2], p[3], p[4], m, nprime, LIMBS)
    acc = add_mod(acc, prod6(p[0], p[1], p[1], p[2], p[5], p[4], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod6(p[0], p[1], p[2], p[5], p[6], p[4], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod6(p[0], p[1], p[2], p[3], p[6], p[4], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[5], p[5], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[5], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod5(p[0], p[1], p[2], p[5], p[7], m, nprime, LIMBS), m, LIMBS)
    return sub_mod(acc, prod5(p[0], p[1], p[2], p[3], p[7], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_complete_add_7(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod2(p[0], p[1], m, nprime, LIMBS)
    acc = sub_mod(acc, prod2(p[0], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod4(p[0], p[3], p[1], p[4], m, nprime, LIMBS), m, LIMBS)
    return add_mod(acc, prod4(p[0], p[3], p[2], p[4], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_complete_add_8(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod2(p[0], p[1], m, nprime, LIMBS)
    acc = sub_mod(acc, prod2(p[0], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod4(p[0], p[3], p[1], p[4], m, nprime, LIMBS), m, LIMBS)
    return add_mod(acc, prod4(p[0], p[3], p[2], p[4], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_complete_add_9(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod2(p[0], p[1], m, nprime, LIMBS)
    acc = sub_mod(acc, prod2(p[0], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod4(p[0], p[3], p[1], p[4], m, nprime, LIMBS), m, LIMBS)
    return add_mod(acc, prod4(p[0], p[2], p[3], p[4], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_complete_add_10(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod2(p[0], p[1], m, nprime, LIMBS)
    acc = sub_mod(acc, prod2(p[0], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod4(p[0], p[3], p[1], p[4], m, nprime, LIMBS), m, LIMBS)
    return add_mod(acc, prod4(p[0], p[3], p[2], p[4], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_complete_add_11(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod2(p[0], p[1], m, nprime, LIMBS)
    acc = sub_mod(acc, prod4(p[0], p[2], p[1], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod4(p[0], p[4], p[1], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod4(p[0], p[5], p[1], p[6], m, nprime, LIMBS), m, LIMBS)
    return sub_mod(acc, prod4(p[0], p[7], p[1], p[6], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_complete_add_12(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod2(p[0], p[1], m, nprime, LIMBS)
    acc = sub_mod(acc, prod4(p[0], p[2], p[1], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod4(p[0], p[4], p[1], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod4(p[0], p[5], p[1], p[6], m, nprime, LIMBS), m, LIMBS)
    return sub_mod(acc, prod4(p[0], p[7], p[1], p[6], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_vanilla_zerocheck_hp(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod3(p[0], p[1], p[2], m, nprime, LIMBS)
    acc = add_mod(acc, prod3(p[3], p[4], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod4(p[5], p[1], p[4], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod3(p[6], p[7], p[2], m, nprime, LIMBS), m, LIMBS)
    return add_mod(acc, prod2(p[8], p[2], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_vanilla_permcheck_hp(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod2(p[0], p[1], m, nprime, LIMBS)
    acc = sub_mod(acc, prod3(p[2], p[3], p[1], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(
        acc, mul7_mod(prod5(p[4], p[5], p[6], p[7], p[1], m, nprime, LIMBS), m, LIMBS), m, LIMBS
    )
    return sub_mod(
        acc, mul7_mod(prod4(p[8], p[9], p[10], p[1], m, nprime, LIMBS), m, LIMBS), m, LIMBS
    )


@triton.jit
def _eval_zk_jellyfish_zerocheck_hp(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod3(p[0], p[1], p[2], m, nprime, LIMBS)
    acc = add_mod(acc, prod3(p[3], p[4], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod4(p[5], p[1], p[4], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod3(p[6], p[7], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod3(p[8], p[1], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod3(p[9], p[4], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod3(p[10], p[7], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod3(p[11], p[12], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = sub_mod(acc, prod3(p[6], p[13], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod4(p[14], p[1], p[4], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod4(p[15], p[7], p[12], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod7(p[16], p[1], p[1], p[1], p[1], p[1], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod7(p[17], p[4], p[4], p[4], p[4], p[4], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod7(p[18], p[7], p[7], p[7], p[7], p[7], p[2], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(
        acc, prod7(p[19], p[12], p[12], p[12], p[12], p[12], p[2], m, nprime, LIMBS), m, LIMBS
    )
    acc = add_mod(acc, prod6(p[20], p[1], p[4], p[7], p[12], p[2], m, nprime, LIMBS), m, LIMBS)
    return add_mod(acc, prod2(p[21], p[2], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def _eval_zk_jellyfish_permcheck_hp(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod2(p[0], p[1], m, nprime, LIMBS)
    acc = sub_mod(acc, prod3(p[2], p[3], p[1], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(
        acc,
        mul7_mod(prod7(p[4], p[5], p[6], p[7], p[8], p[9], p[1], m, nprime, LIMBS), m, LIMBS),
        m,
        LIMBS,
    )
    return sub_mod(
        acc,
        mul7_mod(prod6(p[10], p[11], p[12], p[13], p[14], p[1], m, nprime, LIMBS), m, LIMBS),
        m,
        LIMBS,
    )


@triton.jit
def _eval_zk_opencheck(p, m, nprime, LIMBS: tl.constexpr):
    acc = prod2(p[0], p[1], m, nprime, LIMBS)
    acc = add_mod(acc, prod2(p[2], p[3], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod2(p[4], p[5], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod2(p[6], p[7], m, nprime, LIMBS), m, LIMBS)
    acc = add_mod(acc, prod2(p[8], p[9], m, nprime, LIMBS), m, LIMBS)
    return add_mod(acc, prod2(p[10], p[11], m, nprime, LIMBS), m, LIMBS)


@triton.jit
def eval_zk_workload(WORKLOAD: tl.constexpr, p, m, nprime, LIMBS: tl.constexpr):
    if WORKLOAD == 8:
        return _eval_zk_verifiable_asics(p, m, nprime, LIMBS)
    if WORKLOAD == 9:
        return _eval_zk_spartan_1(p, m, nprime, LIMBS)
    if WORKLOAD == 10:
        return _eval_zk_spartan_2(p, m, nprime, LIMBS)
    if WORKLOAD == 11:
        return _eval_zk_witness_non_id(p, m, nprime, LIMBS)
    if WORKLOAD == 12:
        return _eval_zk_complete_add_1(p, m, nprime, LIMBS)
    if WORKLOAD == 13:
        return _eval_zk_complete_add_7(p, m, nprime, LIMBS)
    if WORKLOAD == 14:
        return _eval_zk_complete_add_8(p, m, nprime, LIMBS)
    if WORKLOAD == 15:
        return _eval_zk_witness_id_point_1(p, m, nprime, LIMBS)
    if WORKLOAD == 16:
        return _eval_zk_witness_id_point_2(p, m, nprime, LIMBS)
    if WORKLOAD == 17:
        return _eval_zk_incomplete_add_1(p, m, nprime, LIMBS)
    if WORKLOAD == 18:
        return _eval_zk_incomplete_add_2(p, m, nprime, LIMBS)
    if WORKLOAD == 19:
        return _eval_zk_complete_add_2(p, m, nprime, LIMBS)
    if WORKLOAD == 20:
        return _eval_zk_complete_add_3(p, m, nprime, LIMBS)
    if WORKLOAD == 21:
        return _eval_zk_complete_add_4(p, m, nprime, LIMBS)
    if WORKLOAD == 22:
        return _eval_zk_complete_add_5(p, m, nprime, LIMBS)
    if WORKLOAD == 23:
        return _eval_zk_complete_add_6(p, m, nprime, LIMBS)
    if WORKLOAD == 24:
        return _eval_zk_complete_add_9(p, m, nprime, LIMBS)
    if WORKLOAD == 25:
        return _eval_zk_complete_add_10(p, m, nprime, LIMBS)
    if WORKLOAD == 26:
        return _eval_zk_complete_add_11(p, m, nprime, LIMBS)
    if WORKLOAD == 27:
        return _eval_zk_complete_add_12(p, m, nprime, LIMBS)
    if WORKLOAD == 28:
        return _eval_zk_vanilla_zerocheck_hp(p, m, nprime, LIMBS)
    if WORKLOAD == 29:
        return _eval_zk_vanilla_permcheck_hp(p, m, nprime, LIMBS)
    if WORKLOAD == 30:
        return _eval_zk_jellyfish_zerocheck_hp(p, m, nprime, LIMBS)
    if WORKLOAD == 31:
        return _eval_zk_jellyfish_permcheck_hp(p, m, nprime, LIMBS)
    return _eval_zk_opencheck(p, m, nprime, LIMBS)
