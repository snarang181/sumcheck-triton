"""Prime fields used by the benchmarks, and a guard on their moduli.

Both backends use the same CIOS Montgomery multiplication over 32-bit limbs
(``mont_mul`` in ``cuda/src/field_sumcheck.cu`` and ``cuda/src/u256_sumcheck.cu``;
``_mont_mul`` in ``benchmarks/field_sweep/triton_sweep.py`` and ``mont_mul`` in
``benchmarks/field_sweep/kernels/common.py``). The routine keeps only L+1 accumulator
words: after the multiply-accumulate step of each outer iteration it stores
``t[L] = (uint32_t)(t[L] + carry)`` and drops bit 32 of that sum (word ``t[L+1]`` in
textbook CIOS). The result is exact only while ``t + a * b_i < 2^(32(L+1))``.

Because both backends share the routine, comparing Triton against CUDA cannot catch
this. This module therefore:

* emulates the kernels' exact word operations in pure Python (``kernel_mont_mul``);
* checks the emulation against big-integer ``a * b * R^-1 mod p`` on seeded random
  inputs, stressed near ``p`` and near 0 (``find_cios_mismatches``);
* checks a closed-form sufficient bound (``cios_bound_holds``). With inputs below p,
  the accumulator satisfies ``t <= 2p - 1`` at the start of every outer iteration, so
  the dropped word is always zero when ``(2p - 1) + (p - 1)(2^32 - 1) < 2^(32(L+1))``.
  Equivalently, ``p < (2^(32(L+1)) + 2^32) / (2^32 + 1)``, which is roughly
  ``2^(32L) - 2^(32(L-1))``. Goldilocks (2^64 - 2^32 + 1) sits exactly on this bound.

``check_montgomery_modulus`` runs all checks and raises ``UnsafeModulusError``. It is
called from ``FieldSpec.__post_init__``, so an unsafe field cannot be constructed.
Results are cached per (modulus, limbs, nprime).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache

MASK32 = 0xFFFFFFFF
MASK64 = 0xFFFFFFFFFFFFFFFF

# Default budget for the randomized check, on top of 36 fixed edge-case pairs. About 17%
# of stressed trials fail for 2^64-59 (about 24% for 2^128-159); over 200 seeds the
# first failure appeared by trial 27 at the latest. Uniform inputs found none in 20,000
# trials. The four configured fields together take about 30 ms, once per process
# (results are cached).
DEFAULT_TRIALS = 256
DEFAULT_SEED = 0x5EED_C105


class UnsafeModulusError(ValueError):
    """The kernels' truncated-CIOS Montgomery multiplication is not exact for this modulus."""


def montgomery_nprime(modulus: int) -> int:
    """Return -modulus^-1 mod 2^32, the per-limb constant the kernels take as ``nprime``."""
    return (-pow(int(modulus), -1, 1 << 32)) & MASK32


def _to_limbs(x: int, limbs: int) -> list[int]:
    return [(x >> (32 * i)) & MASK32 for i in range(limbs)]


def _from_limbs(v: list[int]) -> int:
    return sum(int(w) << (32 * i) for i, w in enumerate(v))


def kernel_mont_mul(a: int, b: int, modulus: int, nprime: int, limbs: int) -> int:
    """Montgomery product computed exactly as the CUDA/Triton kernels compute it.

    Line-by-line mirror of ``mont_mul`` in ``cuda/src/field_sumcheck.cu``: 32-bit limbs,
    uint64 intermediates (these never exceed 2^64 - 1; the ``& MASK64`` mirrors the
    type), L+1 accumulator words with the top carry dropped, and a single conditional
    final subtraction. ``a`` and ``b`` are Montgomery-form values in [0, 2^(32L)).
    """
    L = int(limbs)
    av = _to_limbs(int(a), L)
    bv = _to_limbs(int(b), L)
    mod = _to_limbs(int(modulus), L)
    nprime = int(nprime) & MASK32

    t = [0] * (L + 1)
    for i in range(L):
        carry = 0
        bi = bv[i]
        for j in range(L):
            uv = (t[j] + av[j] * bi + carry) & MASK64
            t[j] = uv & MASK32
            carry = uv >> 32
        uvn = (t[L] + carry) & MASK64
        t[L] = uvn & MASK32  # the kernels' truncation: bit 32 of uvn is discarded

        q = (t[0] * nprime) & MASK32
        carry = 0
        for j in range(L):
            uv = (t[j] + q * mod[j] + carry) & MASK64
            low = uv & MASK32
            carry = uv >> 32
            if j > 0:
                t[j - 1] = low
        last = (t[L] + carry) & MASK64
        t[L - 1] = last & MASK32
        t[L] = (last >> 32) & MASK32

    res = t[:L]
    ge = True  # cmp_mod(res, mod) >= 0, scanning from the top limb
    for i in range(L - 1, -1, -1):
        if res[i] != mod[i]:
            ge = res[i] > mod[i]
            break
    if t[L] != 0 or ge:
        borrow = 0
        out = []
        for i in range(L):  # sub_raw
            bi = mod[i] + borrow
            ai = res[i]
            out.append((ai - bi) & MASK32)
            borrow = 1 if ai < bi else 0
        res = out
    return _from_limbs(res)


def cios_bound_holds(modulus: int, limbs: int) -> bool:
    """Sufficient condition for the truncated CIOS routine to be exact for all a, b < p."""
    p = int(modulus)
    return (2 * p - 1) + (p - 1) * MASK32 < (1 << (32 * (int(limbs) + 1)))


def _stress_value(p: int, limbs: int, rng: random.Random) -> int:
    """One operand in [0, p), biased toward the regions where the dropped carry fires."""
    nbits = 32 * limbs
    kind = rng.randrange(10)
    if kind <= 2:  # just below p (the carry needs a ~ p and b's upper limbs ~ 2^32 - 1)
        x = p - 1 - rng.randrange(1 << 8)
    elif kind == 3:  # below p by a random number of bits
        x = p - 1 - rng.getrandbits(rng.randrange(1, nbits + 1))
    elif kind == 4:  # p's upper limbs, random lower limbs
        k = rng.randrange(1, limbs + 1)
        x = ((p >> (32 * k)) << (32 * k)) | rng.getrandbits(32 * k)
    elif kind == 5:  # limbs drawn from carry-heavy words
        words = (0, 1, 2, MASK32, MASK32 - 1, 0x80000000, 0x7FFFFFFF)
        x = _from_limbs(
            [rng.choice(words) if rng.random() < 0.8 else rng.getrandbits(32) for _ in range(limbs)]
        )
    elif kind == 6:  # near 0
        x = rng.randrange(1 << 8)
    elif kind == 7:  # small magnitude, random length
        x = rng.getrandbits(rng.randrange(1, nbits + 1))
    elif kind == 8:  # Montgomery images of 1, -1, R, -R
        r = (1 << nbits) % p
        x = rng.choice((r, p - r, r * r % p, (p - r * r % p) % p, 1, p - 1))
    else:  # uniform
        x = rng.randrange(p)
    if not 0 <= x < p:
        x %= p
    return x


def find_cios_mismatches(
    modulus: int,
    limbs: int,
    nprime: int,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
    max_report: int = 3,
    stop_early: bool = True,
) -> tuple[int, list[tuple[int, int, int, int]]]:
    """Compare ``kernel_mont_mul`` with big-integer Montgomery multiplication.

    Returns ``(n_mismatches, examples)`` where each example is ``(a, b, got, want)``.
    With ``stop_early`` the search ends after ``max_report`` mismatches.
    """
    p, L = int(modulus), int(limbs)
    rng = random.Random(seed)
    r_inv = pow(pow(2, 32 * L, p), -1, p)
    edge = [0, 1, 2, p - 2, p - 1, (1 << (32 * L)) % p]
    pairs = [(x, y) for x in edge for y in edge if 0 <= x < p and 0 <= y < p]
    n_bad, examples = 0, []
    for k in range(len(pairs) + int(trials)):
        if k < len(pairs):
            a, b = pairs[k]
        else:
            a, b = _stress_value(p, L, rng), _stress_value(p, L, rng)
        got = kernel_mont_mul(a, b, p, nprime, L)
        want = a * b * r_inv % p
        if got != want:
            n_bad += 1
            if len(examples) < max_report:
                examples.append((a, b, got, want))
            if stop_early and n_bad >= max_report:
                break
    return n_bad, examples


@lru_cache(maxsize=None)
def check_montgomery_modulus(
    modulus: int,
    limbs: int,
    nprime: int,
    trials: int = DEFAULT_TRIALS,
    seed: int = DEFAULT_SEED,
) -> None:
    """Raise ``UnsafeModulusError`` unless the kernels' Montgomery multiplication is exact."""
    p, L, nprime = int(modulus), int(limbs), int(nprime)
    where = f"modulus {p:#x} with {L} x 32-bit limbs"
    if L < 1:
        raise UnsafeModulusError(f"{where}: need at least one limb")
    if p < 3 or p % 2 == 0:
        raise UnsafeModulusError(f"{where}: Montgomery arithmetic needs an odd modulus > 2")
    if p >= 1 << (32 * L):
        raise UnsafeModulusError(f"{where}: modulus does not fit in {L} limbs")
    if nprime != montgomery_nprime(p):
        raise UnsafeModulusError(
            f"{where}: nprime {nprime:#x} != -p^-1 mod 2^32 = {montgomery_nprime(p):#x}"
        )
    # 1. The invariant: dropping CIOS's final carry word is valid only inside this bound.
    if not cios_bound_holds(p, L):
        raise UnsafeModulusError(
            f"{where}: the optimized Montgomery routine does not support this modulus. It "
            f"omits CIOS's final carry word, which is valid only when "
            f"(2p-1) + (p-1)(2^32-1) < 2^(32(L+1)). Use a modulus within that bound or add "
            f"the extra carry word to both kernels."
        )
    # 2. Second line of defense: targeted and randomized products against big integers.
    n_bad, examples = find_cios_mismatches(p, L, nprime, trials=trials, seed=seed)
    if n_bad:
        a, b, got, want = examples[0]
        raise UnsafeModulusError(
            f"{where}: the optimized Montgomery routine does not support this modulus "
            f"({n_bad} mismatching products in a seeded stress test; e.g. a={a:#x}, "
            f"b={b:#x}: kernel {got:#x}, expected {want:#x})."
        )


@dataclass(frozen=True)
class FieldSpec:
    name: str
    bit_width: int
    limbs: int
    modulus: int
    nprime: int

    def __post_init__(self) -> None:
        # The kernels' CIOS Montgomery multiplication drops its top carry word, which is
        # exact only for moduli below a bound (see the module docstring). Refuse others.
        check_montgomery_modulus(self.modulus, self.limbs, self.nprime)

    @property
    def r_mod(self) -> int:
        return (1 << (32 * self.limbs)) % self.modulus

    def to_limbs(self, value: int) -> tuple[int, ...]:
        x = int(value) % self.modulus
        return tuple((x >> (32 * i)) & 0xFFFFFFFF for i in range(self.limbs))

    def montgomery_encode(self, value: int) -> int:
        return (int(value) % self.modulus) * self.r_mod % self.modulus

    def montgomery_limbs(self, value: int) -> tuple[int, ...]:
        return self.to_limbs(self.montgomery_encode(value))


FIELD_SPECS: dict[int, FieldSpec] = {
    # Largest 32-bit prime.
    32: FieldSpec(
        name="p32_2^32_minus_5",
        bit_width=32,
        limbs=1,
        modulus=0xFFFFFFFB,
        nprime=0xCCCCCCCD,
    ),
    # Goldilocks field: 2^64 - 2^32 + 1.
    64: FieldSpec(
        name="p64_goldilocks",
        bit_width=64,
        limbs=2,
        modulus=0xFFFFFFFF00000001,
        nprime=0xFFFFFFFF,
    ),
    # Mersenne prime 2^127 - 1, represented in four 32-bit limbs.
    128: FieldSpec(
        name="p128_mersenne_2^127_minus_1",
        bit_width=128,
        limbs=4,
        modulus=(1 << 127) - 1,
        nprime=0x00000001,
    ),
    # BLS12-381 scalar field.
    256: FieldSpec(
        name="p255_bls12_381_scalar",
        bit_width=256,
        limbs=8,
        modulus=int(
            "73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001",
            16,
        ),
        nprime=0xFFFFFFFF,
    ),
}


def get_field(bit_width: int) -> FieldSpec:
    try:
        return FIELD_SPECS[int(bit_width)]
    except KeyError as exc:
        supported = ", ".join(str(x) for x in sorted(FIELD_SPECS))
        raise ValueError(
            f"unsupported field bit width {bit_width}; expected one of {supported}"
        ) from exc
