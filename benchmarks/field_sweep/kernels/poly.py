from __future__ import annotations

import triton
import triton.language as tl

from kernels.common import add_mod, mont_mul, prod2, prod3, prod4


@triton.jit
def eval_poly_workload(WORKLOAD: tl.constexpr, p, m, nprime, LIMBS: tl.constexpr):
    if WORKLOAD == 1:
        return p[0]
    if WORKLOAD == 2:
        return prod2(p[0], p[1], m, nprime, LIMBS)
    if WORKLOAD == 3:
        return add_mod(prod2(p[0], p[1], m, nprime, LIMBS), p[2], m, LIMBS)
    if WORKLOAD == 4:
        return prod3(p[0], p[1], p[2], m, nprime, LIMBS)
    if WORKLOAD == 5:
        aa = prod2(p[0], p[0], m, nprime, LIMBS)
        bb = prod2(p[1], p[1], m, nprime, LIMBS)
        return mont_mul(mont_mul(aa, bb, m, nprime, LIMBS), p[2], m, nprime, LIMBS)
    if WORKLOAD == 6:
        abc = prod3(p[0], p[1], p[2], m, nprime, LIMBS)
        de = prod2(p[3], p[4], m, nprime, LIMBS)
        return add_mod(abc, de, m, LIMBS)
    abcg = prod4(p[0], p[1], p[2], p[3], m, nprime, LIMBS)
    deg = prod3(p[4], p[5], p[3], m, nprime, LIMBS)
    return add_mod(abcg, deg, m, LIMBS)
