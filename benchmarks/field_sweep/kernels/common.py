from __future__ import annotations

import triton
import triton.language as tl


@triton.jit
def zero_tuple(offs, LIMBS: tl.constexpr):
    z = (offs * 0).to(tl.uint64)
    out = ()
    for _ in tl.static_range(0, LIMBS):
        out += (z,)
    return out


@triton.jit
def load_elem(ptr, idxs, mask, LIMBS: tl.constexpr):
    out = ()
    base = idxs.to(tl.int64) * LIMBS
    for i in tl.static_range(0, LIMBS):
        out += (tl.load(ptr + base + i, mask=mask, other=0).to(tl.uint64),)
    return out


@triton.jit
def load_const(ptr, LIMBS: tl.constexpr):
    out = ()
    for i in tl.static_range(0, LIMBS):
        out += (tl.load(ptr + i).to(tl.uint64),)
    return out


@triton.jit
def store_elem(ptr, idxs, value, mask, LIMBS: tl.constexpr):
    base = idxs.to(tl.int64) * LIMBS
    for i in tl.static_range(0, LIMBS):
        tl.store(ptr + base + i, value[i].to(tl.uint32), mask=mask)


@triton.jit
def load_table_elem(ptr, idxs, row_stride, mask, LIMBS: tl.constexpr, LAYOUT: tl.constexpr):
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
def store_table_elem(ptr, idxs, row_stride, value, mask, LIMBS: tl.constexpr, LAYOUT: tl.constexpr):
    idxs = idxs.to(tl.int64)
    row_stride = row_stride.to(tl.int64)
    for i in tl.static_range(0, LIMBS):
        if LAYOUT == 0:
            addr = ptr + idxs * LIMBS + i
        else:
            addr = ptr + i * row_stride + idxs
        tl.store(addr, value[i].to(tl.uint32), mask=mask)


@triton.jit
def select_tuple(cond, a, b, LIMBS: tl.constexpr):
    out = ()
    for i in tl.static_range(0, LIMBS):
        out += (tl.where(cond, a[i], b[i]),)
    return out


@triton.jit
def ge_tuple(a, b, LIMBS: tl.constexpr):
    gt = a[0] == (a[0] + 1)
    eq = a[0] == a[0]
    for i in tl.static_range(LIMBS - 1, -1, -1):
        gt = gt | (eq & (a[i] > b[i]))
        eq = eq & (a[i] == b[i])
    return gt | eq


@triton.jit
def add_raw(a, b, LIMBS: tl.constexpr):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    carry = a[0] * 0
    out = ()
    for i in tl.static_range(0, LIMBS):
        uv = a[i] + b[i] + carry
        out += (uv & mask32,)
        carry = uv >> 32
    return out, carry


@triton.jit
def sub_raw(a, b, LIMBS: tl.constexpr):
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
def add_mod(a, b, m, LIMBS: tl.constexpr):
    s, carry = add_raw(a, b, LIMBS)
    reduced, _ = sub_raw(s, m, LIMBS)
    need_reduce = (carry != 0) | ge_tuple(s, m, LIMBS)
    return select_tuple(need_reduce, reduced, s, LIMBS)


@triton.jit
def sub_mod(a, b, m, LIMBS: tl.constexpr):
    d, borrow = sub_raw(a, b, LIMBS)
    corrected, _ = add_raw(d, m, LIMBS)
    return select_tuple(borrow != 0, corrected, d, LIMBS)


@triton.jit
def double_mod(a, m, LIMBS: tl.constexpr):
    return add_mod(a, a, m, LIMBS)


@triton.jit
def mul3_mod(a, m, LIMBS: tl.constexpr):
    return add_mod(double_mod(a, m, LIMBS), a, m, LIMBS)


@triton.jit
def mul5_mod(a, m, LIMBS: tl.constexpr):
    two = double_mod(a, m, LIMBS)
    four = double_mod(two, m, LIMBS)
    return add_mod(four, a, m, LIMBS)


@triton.jit
def mul7_mod(a, m, LIMBS: tl.constexpr):
    return add_mod(mul5_mod(a, m, LIMBS), double_mod(a, m, LIMBS), m, LIMBS)


@triton.jit
def mont_mul(a, b, m, nprime, LIMBS: tl.constexpr):
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
    reduced, _ = sub_raw(res, m, LIMBS)
    need_reduce = (t[LIMBS] != 0) | ge_tuple(res, m, LIMBS)
    return select_tuple(need_reduce, reduced, res, LIMBS)


@triton.jit
def prod2(a, b, m, nprime, LIMBS: tl.constexpr):
    return mont_mul(a, b, m, nprime, LIMBS)


@triton.jit
def prod3(a, b, c, m, nprime, LIMBS: tl.constexpr):
    return mont_mul(mont_mul(a, b, m, nprime, LIMBS), c, m, nprime, LIMBS)


@triton.jit
def prod4(a, b, c, d, m, nprime, LIMBS: tl.constexpr):
    return mont_mul(prod3(a, b, c, m, nprime, LIMBS), d, m, nprime, LIMBS)


@triton.jit
def prod5(a, b, c, d, e, m, nprime, LIMBS: tl.constexpr):
    return mont_mul(prod4(a, b, c, d, m, nprime, LIMBS), e, m, nprime, LIMBS)


@triton.jit
def prod6(a, b, c, d, e, f, m, nprime, LIMBS: tl.constexpr):
    return mont_mul(prod5(a, b, c, d, e, m, nprime, LIMBS), f, m, nprime, LIMBS)


@triton.jit
def prod7(a, b, c, d, e, f, g, m, nprime, LIMBS: tl.constexpr):
    return mont_mul(prod6(a, b, c, d, e, f, m, nprime, LIMBS), g, m, nprime, LIMBS)
