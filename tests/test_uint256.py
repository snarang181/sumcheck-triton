from __future__ import annotations

import random

import pytest
import torch

from zkduel.triton_backend.uint256 import (
    UINT256_MASK,
    from_uint256_tensor,
    montgomery_decode_values,
    montgomery_encode_values,
    montgomery_nprime,
    to_uint256_tensor,
    uint256_add,
    uint256_add_mod,
    uint256_cmp,
    uint256_mont_mul,
    uint256_mul_lo,
    uint256_sub,
    uint256_sub_mod,
)

MODULUS = int("73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001", 16)


def _values(n: int) -> list[int]:
    rng = random.Random(256)
    fixed = [
        0,
        1,
        2,
        (1 << 32) - 1,
        1 << 32,
        (1 << 128) - 1,
        1 << 255,
        UINT256_MASK,
        MODULUS - 1,
    ]
    vals = fixed[:]
    while len(vals) < n:
        vals.append(rng.randrange(0, 1 << 256))
    return vals[:n]


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA device unavailable for Triton uint256 kernels"
)


def test_uint256_roundtrip():
    vals = _values(16)
    t = to_uint256_tensor(vals, device="cuda")
    assert from_uint256_tensor(t) == vals


def test_uint256_add_sub_cmp():
    a_vals = _values(32)
    b_vals = list(reversed(_values(32)))
    a = to_uint256_tensor(a_vals, device="cuda")
    b = to_uint256_tensor(b_vals, device="cuda")

    add_out, carry = uint256_add(a, b)
    sub_out, borrow = uint256_sub(a, b)
    cmp_out = uint256_cmp(a, b).cpu().tolist()

    expected_add = [(x + y) & UINT256_MASK for x, y in zip(a_vals, b_vals)]
    expected_carry = [1 if x + y > UINT256_MASK else 0 for x, y in zip(a_vals, b_vals)]
    expected_sub = [(x - y) & UINT256_MASK for x, y in zip(a_vals, b_vals)]
    expected_borrow = [1 if x < y else 0 for x, y in zip(a_vals, b_vals)]
    expected_cmp = [(x > y) - (x < y) for x, y in zip(a_vals, b_vals)]

    assert from_uint256_tensor(add_out) == expected_add
    assert carry.cpu().tolist() == expected_carry
    assert from_uint256_tensor(sub_out) == expected_sub
    assert borrow.cpu().tolist() == expected_borrow
    assert cmp_out == expected_cmp


def test_uint256_mul_lo():
    a_vals = _values(24)
    b_vals = list(reversed(_values(24)))
    a = to_uint256_tensor(a_vals, device="cuda")
    b = to_uint256_tensor(b_vals, device="cuda")

    out = uint256_mul_lo(a, b)
    expected = [(x * y) & UINT256_MASK for x, y in zip(a_vals, b_vals)]
    assert from_uint256_tensor(out) == expected


def test_uint256_mod_add_sub():
    rng = random.Random(99)
    a_vals = [rng.randrange(0, MODULUS) for _ in range(32)]
    b_vals = [rng.randrange(0, MODULUS) for _ in range(32)]
    # Force edge cases around reduction and underflow.
    a_vals[:4] = [0, 1, MODULUS - 1, MODULUS - 2]
    b_vals[:4] = [1, 2, 1, MODULUS - 1]

    a = to_uint256_tensor(a_vals, device="cuda")
    b = to_uint256_tensor(b_vals, device="cuda")

    add_out = uint256_add_mod(a, b, MODULUS)
    sub_out = uint256_sub_mod(a, b, MODULUS)

    assert from_uint256_tensor(add_out) == [(x + y) % MODULUS for x, y in zip(a_vals, b_vals)]
    assert from_uint256_tensor(sub_out) == [(x - y) % MODULUS for x, y in zip(a_vals, b_vals)]


def test_uint256_montgomery_mul_bls12_scalar():
    rng = random.Random(381)
    a_vals = [rng.randrange(0, MODULUS) for _ in range(32)]
    b_vals = [rng.randrange(0, MODULUS) for _ in range(32)]
    a_vals[:6] = [0, 1, 2, MODULUS - 1, MODULUS - 2, (1 << 255) % MODULUS]
    b_vals[:6] = [0, 1, MODULUS - 1, 2, MODULUS - 2, 17]

    a_mont_vals = montgomery_encode_values(a_vals, MODULUS)
    b_mont_vals = montgomery_encode_values(b_vals, MODULUS)
    a = to_uint256_tensor(a_mont_vals, device="cuda")
    b = to_uint256_tensor(b_mont_vals, device="cuda")

    out = uint256_mont_mul(a, b, MODULUS)
    decoded = montgomery_decode_values(from_uint256_tensor(out), MODULUS)

    assert decoded == [(x * y) % MODULUS for x, y in zip(a_vals, b_vals)]


def test_uint256_montgomery_mul_accepts_explicit_nprime():
    a_vals = [3, 5, 7, 11]
    b_vals = [13, 17, 19, 23]
    a = to_uint256_tensor(montgomery_encode_values(a_vals, MODULUS), device="cuda")
    b = to_uint256_tensor(montgomery_encode_values(b_vals, MODULUS), device="cuda")
    nprime = montgomery_nprime(MODULUS)

    out = uint256_mont_mul(a, b, to_uint256_tensor([MODULUS], device="cuda")[0], nprime=nprime)
    decoded = montgomery_decode_values(from_uint256_tensor(out), MODULUS)

    assert decoded == [(x * y) % MODULUS for x, y in zip(a_vals, b_vals)]
