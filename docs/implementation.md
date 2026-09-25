# Implementation Notes

## SumCheck Round Shape

For a table of length `N = 2^rounds`, each round pairs neighboring multilinear
extension values:

```text
even = table[2*i]
odd  = table[2*i + 1]
diff = odd - even
value(t) = even + t * diff
```

The prover evaluates `g(t) = sum_i expression(value_i(t))` for `t = 0..degree`
and folds the table with the verifier challenge `r`:

```text
next[i] = even + r * diff
```

The benchmark suite keeps this same round/eval/fold shape in Triton and CUDA.

## Field-sweep kernels (used for the paper)

- **Triton.** `benchmarks/field_sweep/triton_sweep.py` holds the round loop and the eval,
  reduction and fold kernels. The per-workload expressions are in `kernels/poly.py` (diagnostic
  tier) and `kernels/zk.py` (zkPHIRE constraints); the modular arithmetic they share is in
  `kernels/common.py`. `LIMBS` and `WORKLOAD` are `tl.constexpr`, so each (workload, width)
  compiles its own kernels.
- **CUDA.** `cuda/src/field_sumcheck.cu` implements the same kernels as templates over the
  workload, the limb count (`L` = 1, 2, 4, 8), the table layout, the point mode and the
  interpolation point. The build instantiates them for every workload and width and takes
  about an hour.
- **Launch structure.** Each round launches one eval kernel per interpolation point. Each is
  followed by a log-depth tree of pairwise reduction launches and a synchronizing
  device-to-host copy; a fold kernel then produces the next table. Both backends issue the same
  launches.
- **Options.** `--layout element|limb` selects element-major or limb-major tables.
  `--point-mode specialized` uses point-specific interpolation arithmetic; `generic` uses the
  general form. The paper's sweeps use element layout and specialized points. `--block` sets the
  CUDA block size (default 128).
- **Reduction.** Triton reduces each block's partial sums in `_store_eval_partial` with a
  reshape/split tree; `experiments/h100/tools/treduce.py` swaps in `tl.reduce` for the ablation
  in `experiments/h100/FINDINGS.md` section 3. CUDA uses a shared-memory tree.

## 256-bit Triton backend (Python package)

The package backend represents each field element as eight little-endian
`uint32` limbs in Montgomery form. The static SumCheck implementation lives in
`src/zkduel/triton_backend/u256_sumcheck.py`; the limb helpers live in
`src/zkduel/triton_backend/uint256.py`. The `benchmarks/u256/` drivers use it.

The backend specializes to a fixed workload id rather than accepting an arbitrary
runtime expression. This keeps the kernels readable and makes compile-time behavior
measurable.

## CUDA executables

The CUDA code is direct `.cu` code under `cuda/src/` and builds to standalone
executables under `build/`. There are no Python bindings in the CUDA path. Python
benchmark drivers only launch executables, collect CSVs, and join results.

## Fields

`src/zkduel/fields.py` defines prime-field parameters for 32, 64, 128,
and 256-bit Montgomery fields. These are used by the prime-field sweep. The
wraparound sweep deliberately uses arithmetic modulo `2^k`; keep those results
separate from the finite-field results.

Both backends share one CIOS Montgomery multiplication that drops its top carry
word. It is exact only for moduli with `(2p-1) + (p-1)(2^32-1) < 2^(32(L+1))`
(L = number of 32-bit limbs). All four configured primes satisfy this. Moduli near
`2^(32L)` do not; for example, 2^64-59 and 2^128-159 give wrong products. Constructing a
`FieldSpec` checks the modulus: it emulates the kernels' exact word operations
against big-integer arithmetic on stressed random inputs and checks the bound, and
it raises `UnsafeModulusError` for an unsafe modulus. See
`tests/test_field_guard.py`.

## Timing Modes

The prime-field sweep supports two timing modes. `whole-run` keeps the original
single timer around the full SumCheck run. `round-sum` measures each round as
separate eval, challenge, and fold phases, then reports their sum. With
`--challenge-mode sha3`, the challenge phase uses host-side SHA3-256 over the
round polynomial coefficients and derives the next fold challenge from the
digest. The Python/Triton driver uses `hashlib`; the CUDA executable uses
OpenSSL EVP for SHA3 and OpenSSL BIGNUM to reduce the digest into the selected
field before Montgomery encoding. This gives a transcript-inclusive timing path
while preserving the fixed-challenge kernel-only path for isolation.
