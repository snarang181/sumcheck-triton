"""CPU-only tests for the Montgomery-modulus guard in zkduel.fields.

fields.py is loaded by file path (as cuda/check/check_field_sumcheck.py does), so these
tests do not import torch, triton, or the zkduel package, and never touch a GPU.
"""

from __future__ import annotations

import importlib.util
import random
import sys
import time
from pathlib import Path

import pytest

_FIELDS_PY = Path(__file__).resolve().parents[1] / "src" / "zkduel" / "fields.py"
_spec = importlib.util.spec_from_file_location("zkduel_fields_guard_test", _FIELDS_PY)
fields = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = fields
_spec.loader.exec_module(fields)

GOLDILOCKS = 0xFFFFFFFF00000001
UNSAFE = [((1 << 64) - 59, 2), ((1 << 128) - 159, 4)]
UNSAFE_IDS = ["2^64-59", "2^128-159"]


def _spec_for(modulus: int, limbs: int, nprime: int | None = None):
    return fields.FieldSpec(
        name=f"test_{modulus:#x}",
        bit_width=32 * limbs,
        limbs=limbs,
        modulus=modulus,
        nprime=fields.montgomery_nprime(modulus) if nprime is None else nprime,
    )


def _textbook_cios(a: int, b: int, p: int, limbs: int) -> int:
    """Same word operations as the kernels, but keeping the top carry word t[L+1]."""
    m32, L = 0xFFFFFFFF, limbs
    nprime = fields.montgomery_nprime(p)
    av = [(a >> (32 * i)) & m32 for i in range(L)]
    bv = [(b >> (32 * i)) & m32 for i in range(L)]
    mv = [(p >> (32 * i)) & m32 for i in range(L)]
    t = [0] * (L + 2)
    for i in range(L):
        c = 0
        for j in range(L):
            uv = t[j] + av[j] * bv[i] + c
            t[j], c = uv & m32, uv >> 32
        uvn = t[L] + c
        t[L], t[L + 1] = uvn & m32, uvn >> 32
        q = (t[0] * nprime) & m32
        c = 0
        for j in range(L):
            uv = t[j] + q * mv[j] + c
            c = uv >> 32
            if j > 0:
                t[j - 1] = uv & m32
        last = t[L] + c
        t[L - 1], t[L], t[L + 1] = last & m32, t[L + 1] + (last >> 32), 0
    v = sum(w << (32 * i) for i, w in enumerate(t[:L])) + (t[L] << (32 * L))
    return v - p if v >= p else v


@pytest.mark.parametrize("bit_width", sorted(fields.FIELD_SPECS))
def test_configured_fields_pass(bit_width):
    f = fields.FIELD_SPECS[bit_width]
    assert fields.cios_bound_holds(f.modulus, f.limbs)
    assert f.nprime == fields.montgomery_nprime(f.modulus)
    n_bad, _ = fields.find_cios_mismatches(
        f.modulus, f.limbs, f.nprime, trials=5000, seed=12345, stop_early=False
    )
    assert n_bad == 0
    fields.check_montgomery_modulus(f.modulus, f.limbs, f.nprime)
    assert fields.get_field(bit_width) is f


@pytest.mark.parametrize(
    "modulus,limbs",
    [((1 << 31) - 1, 1), ((1 << 61) - 1, 2), ((1 << 89) - 1, 3), ((1 << 255) - 19, 8)],
    ids=["2^31-1", "2^61-1", "2^89-1", "2^255-19"],
)
def test_emulator_matches_bigint_for_other_safe_moduli(modulus, limbs):
    assert fields.cios_bound_holds(modulus, limbs)
    n_bad, _ = fields.find_cios_mismatches(
        modulus, limbs, fields.montgomery_nprime(modulus), trials=2000, stop_early=False
    )
    assert n_bad == 0
    assert _spec_for(modulus, limbs).modulus == modulus


@pytest.mark.parametrize("modulus,limbs", UNSAFE, ids=UNSAFE_IDS)
def test_unsafe_moduli_are_rejected_by_counterexample(modulus, limbs):
    # Both lines of defense reject these moduli: the bound (checked first) and a concrete
    # counterexample from the randomized test (see test_default_budget_rejects_...).
    assert not fields.cios_bound_holds(modulus, limbs)
    with pytest.raises(fields.UnsafeModulusError, match="does not support this modulus"):
        _spec_for(modulus, limbs)
    # UnsafeModulusError is a ValueError, like get_field's unsupported-width error.
    with pytest.raises(ValueError):
        _spec_for(modulus, limbs)


@pytest.mark.parametrize("modulus,limbs", UNSAFE, ids=UNSAFE_IDS)
def test_default_budget_rejects_unsafe_moduli_for_every_seed(modulus, limbs):
    nprime = fields.montgomery_nprime(modulus)
    r_inv = pow(pow(2, 32 * limbs, modulus), -1, modulus)
    for seed in range(100):
        n_bad, examples = fields.find_cios_mismatches(modulus, limbs, nprime, seed=seed)
        assert n_bad > 0, f"seed {seed} missed the unsafe modulus {modulus:#x}"
        for a, b, got, want in examples:
            assert want == a * b * r_inv % modulus != got
            # The mismatch is due to the dropped carry word alone.
            assert _textbook_cios(a, b, modulus, limbs) == want


def test_uniform_inputs_alone_would_miss_the_bug():
    # Why the inputs are stressed: uniform inputs essentially never trigger the carry.
    p, limbs = UNSAFE[0]
    nprime, r_inv, rng = fields.montgomery_nprime(p), pow(pow(2, 64, p), -1, p), random.Random(1)
    bad = 0
    for _ in range(2000):
        a, b = rng.randrange(p), rng.randrange(p)
        bad += fields.kernel_mont_mul(a, b, p, nprime, limbs) != a * b * r_inv % p
    assert bad == 0


def test_bound_rejects_moduli_just_above_goldilocks():
    assert fields.cios_bound_holds(GOLDILOCKS, 2)
    assert not fields.cios_bound_holds(GOLDILOCKS + 2, 2)
    with pytest.raises(fields.UnsafeModulusError, match="valid only when"):
        _spec_for(GOLDILOCKS + 2, 2)


@pytest.mark.parametrize(
    "modulus,limbs,nprime,match",
    [
        ((1 << 61) - 1, 2, 0x12345, "nprime"),
        ((1 << 61) - 2, 2, None, "odd modulus"),
        ((1 << 64) + 13, 2, None, "does not fit"),
    ],
    ids=["wrong-nprime", "even", "too-wide"],
)
def test_malformed_specs_are_rejected(modulus, limbs, nprime, match):
    if nprime is None:
        nprime = fields.montgomery_nprime(modulus) if modulus % 2 else 1
    with pytest.raises(fields.UnsafeModulusError, match=match):
        _spec_for(modulus, limbs, nprime)


def test_guard_cost_for_configured_fields_is_small():
    fields.check_montgomery_modulus.cache_clear()
    start = time.perf_counter()
    for f in fields.FIELD_SPECS.values():
        fields.check_montgomery_modulus(f.modulus, f.limbs, f.nprime)
    assert time.perf_counter() - start < 2.0
