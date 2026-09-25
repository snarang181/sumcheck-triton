# Benchmarks

Run all commands from the repository root with `PYTHONPATH=src`. `rounds = r` means the initial
table has `N = 2^r` rows. Every driver writes CSVs; `results/` is git-ignored.

## Prime-field sweep (the paper's driver)

`benchmarks/field_sweep/` runs the 32 static workloads (25 zkPHIRE constraints and 7
diagnostic-tier polynomials) at 32, 64, 128 and 256 bits in Triton and CUDA. The CUDA executable
is `build/field_sumcheck_cuda`, built by `cuda/build/build_field_sumcheck.sh` (about 1 h).

| Driver | Use |
|---|---|
| `triton_sweep.py` | Triton only |
| `compare_triton_cuda.py` | Triton, then CUDA, then a joined CSV with checksum agreement |
| `resilient_compare_triton_cuda.py` | the same, one subprocess per (workload, width, rounds) case, with `--per-case-timeout`; a failed case becomes a `skipped` row instead of ending the sweep |

The paper's sweeps use the resilient driver with these settings (the recorded commands are in
`experiments/h100/sweep_fixed/*.command.txt`):

```bash
python -u benchmarks/field_sweep/resilient_compare_triton_cuda.py \
  --zk-only --bit-widths 32 64 128 256 \
  --rounds 14 16 18 20 22 24 26 28 \
  --warmups 2 --repeats 10 --point-mode specialized \
  --per-case-timeout 1800 --skip-build \
  --tmp-dir results/cases --out results/zk_sweep.csv
```

For the diagnostic tier, replace `--zk-only` with
`--workloads poly_a poly_ab poly_ab_plus_c poly_abc poly_aabbc poly_abc_plus_de poly_abcg_plus_deg`.

Workload names: the canonical names (for example `zk_complete_add_11`) and the aliases
`zkphire_0` to `zkphire_24` are accepted. Without `--workloads`, the drivers run all 32 workloads,
or the 25 zkPHIRE constraints with `--zk-only`.

Options shared by the three drivers:

| Option | Values | Default |
|---|---|---|
| `--point-mode` | `generic`, `specialized` (specialized interpolation-point arithmetic) | `generic`; the paper uses `specialized` |
| `--layout` | `element` (element-major tables), `limb` (limb-major) | `element` |
| `--timing-mode` | `whole-run`, `round-sum` | `whole-run` |
| `--challenge-mode` | `fixed`, `sha3` | `fixed` |
| `--block` | CUDA threads per block (not in `triton_sweep.py`) | 128 |

## Transcript-inclusive round timing

The field sweep can also time SumCheck as a sum of per-round phases:

```text
eval round polynomial + SHA3 challenge generation + fold
```

```bash
python -u benchmarks/field_sweep/compare_triton_cuda.py \
  --workloads poly_abc \
  --bit-widths 256 \
  --rounds 8 10 12 14 16 \
  --timing-mode round-sum \
  --challenge-mode sha3
```

The comparison CSV includes `triton_eval_median_ms`, `triton_challenge_median_ms`,
`triton_fold_median_ms`, and the corresponding CUDA columns. The paper's sweeps use the default
`whole-run` timing with fixed challenges.

## 256-bit drivers (earlier)

`benchmarks/u256/` holds the earlier 256-bit-only study: matched Triton and CUDA SumCheck over
the BLS12-381 scalar field for the diagnostic-tier polynomials. It uses the Python package's
backend in `src/zkduel/triton_backend/` and `cuda/src/u256_sumcheck.cu`
(`cuda/build/build_u256_sumcheck.sh`).

```bash
python -u benchmarks/u256/compare_triton_cuda.py --rounds 8 10 12 14 16 18 20 22 \
  --warmups 1 --repeats 5
```

`benchmarks/u256/device_resident_sweep.py --schedule capacity` pushes to large `N`. It estimates
memory before allocation and records skipped rows instead of assuming a particular GPU.

## Wraparound bit-width ablation

`benchmarks/wraparound_sweep/` uses limb arithmetic modulo `2^k`. Use it only when the question is
limb overhead; it is not finite-field performance.

```bash
python -u benchmarks/wraparound_sweep/compare_triton_cuda.py --workloads poly_abc \
  --bit-widths 32 64 128 256 --rounds 8 10 12 14 16 18 20 22
```

## Plotting

```bash
pip install -e '.[plot]'
python scripts/plot_zkduel.py results/triton_cuda_field_round_sum_sha3_all_polys.csv
```

The script writes PDF figures to `results/plots/` and summary CSV tables to `results/tables/`.
Pass large device-resident scaling CSVs with `--scaling-csv`. It skips rows without checksum
agreement unless `--include-bad-checksums` is given (for debugging only).
