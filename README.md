# zkDuel

zkDuel is the artifact of the paper *zkDuel: Characterizing SumCheck Prover Kernels across GPU
Programming Models* (HASP '26). It contains matched Triton and CUDA SumCheck prover kernels over
prime fields, the drivers that time and check them, and the data behind the paper's results.

Repository: <https://github.com/snarang181/zk-duel>

## What zkDuel is

- **Matched kernels.** Each workload has a Triton implementation and a native CUDA
  implementation of the same SumCheck round: evaluate the round polynomial at its interpolation
  points, reduce the partial sums, fold the tables with the challenge. Both backends issue the same
  kernel launches and host synchronizations per run. The CUDA code builds to a standalone
  executable; the Python drivers run it as a subprocess, with no Python bindings.
- **Workloads.** 25 constraints derived from zkPHIRE, in seven families, plus a diagnostic tier of
  7 polynomials. Both sets are listed below.
- **Fields.** Four prime fields in Montgomery form with 32-bit limbs:

  | Width | Prime | Limbs |
  |---:|---|---:|
  | 32 | 2^32 − 5 | 1 |
  | 64 | Goldilocks, 2^64 − 2^32 + 1 | 2 |
  | 128 | 2^127 − 1 | 4 |
  | 256 | BLS12-381 scalar field (255-bit prime) | 8 |

- **Correctness gating.** A configuration counts as a result only when both backends finish and
  their checksums agree. Both backends are also checked against a CPU big-integer reference
  computed from the canonical expressions in `src/zkduel/workloads.py`.
- **Measurements.** Median wall-clock time per SumCheck run, GPU-kernel time from Nsight Systems,
  registers and occupancy, and Triton compile time.

zkPHIRE constraints (`--zk-only` selects all 25; the aliases `zkphire_0` to `zkphire_24` are also
accepted):

| Family | n | Workloads |
|---|---:|---|
| Verifiable / Spartan | 3 | `zk_verifiable_asics`, `zk_spartan_1`, `zk_spartan_2` |
| Halo2 curve checks | 3 | `zk_witness_non_id`, `zk_witness_id_point_1`, `zk_witness_id_point_2` |
| Incomplete addition | 2 | `zk_incomplete_add_1`, `zk_incomplete_add_2` |
| Complete addition | 12 | `zk_complete_add_1` to `zk_complete_add_12` |
| HyperPlonk vanilla | 2 | `zk_vanilla_zerocheck_hp`, `zk_vanilla_permcheck_hp` |
| HyperPlonk Jellyfish | 2 | `zk_jellyfish_zerocheck_hp`, `zk_jellyfish_permcheck_hp` |
| Opening check | 1 | `zk_opencheck` |

Diagnostic tier:

| Workload | Polynomial |
|---|---|
| `poly_a` | `a` |
| `poly_ab` | `a*b` |
| `poly_ab_plus_c` | `a*b + c` |
| `poly_abc` | `a*b*c` |
| `poly_aabbc` | `a*a*b*b*c` |
| `poly_abc_plus_de` | `a*b*c + d*e` |
| `poly_abcg_plus_deg` | `a*b*c*g + d*e*g` |

## Repository layout

```text
src/zkduel/                    Python package: fields and modulus guard (fields.py), canonical
                               workload expressions (workloads.py), CPU big-integer SumCheck
                               reference (reference.py), 256-bit Triton backend (triton_backend/)
benchmarks/field_sweep/        Prime-field drivers used for the paper:
  triton_sweep.py              Triton SumCheck for all 32 workloads and 4 widths
  compare_triton_cuda.py       Triton vs CUDA, joined CSV with checksum agreement
  resilient_compare_triton_cuda.py  Same, one subprocess per case, with a per-case timeout
  workload_specs.py            Workload metadata and aliases
  kernels/                     Triton expressions: poly.py (diagnostic tier), zk.py (zkPHIRE)
benchmarks/u256/               Earlier 256-bit-only drivers (diagnostic-tier polynomials)
benchmarks/wraparound_sweep/   Limb-width ablation with arithmetic mod 2^k (not a prime field)
cuda/src/                      CUDA sources; field_sumcheck.cu is the paper's CUDA backend
cuda/build/                    nvcc build scripts
cuda/check/                    CPU checkers for the CUDA executables
scripts/                       setup_env.sh, run_paper_experiments.sh, plot_zkduel.py
tests/                         pytest suite (field guard, 256-bit arithmetic, CPU reference)
docs/                          Benchmarks, implementation, reproduction, reading the results
experiments/                   Data behind the paper and the scripts that produced it
  README.md                    Index: every dataset, the question it answers, its script
  h100/                        H100 PCIe, the primary platform: sweeps, diagnostics, validation,
                               tools/, patches/, env/, logs/, FINDINGS.md, FINDINGS_TABLES.md
  a100/                        A100: the full sweep CSV and the A100 PCIe follow-ups (followups/)
```

## Requirements

The paper's software stack:

| Component | Version | Used for |
|---|---|---|
| Python | 3.12 | drivers, Triton |
| PyTorch | 2.12.0+cu130 | device tensors for the Triton backend |
| Triton | 3.7.0 | Triton backend (its bundled `ptxas`) |
| nvcc | 12.9 | CUDA executable |
| OpenSSL headers (`libssl-dev`) | 3.0.2 on the H100 host | SHA3 challenges and BIGNUM in the CUDA executable |
| NVIDIA driver | supports CUDA 13 | PyTorch cu130 runtime |
| Nsight Systems (`nsys`) | | GPU-kernel time in the diagnostic phases |
| Nsight Compute (`ncu`) | | hardware counters only; needs root where profiling is restricted |

- **GPU.** The recorded results use an NVIDIA H100 PCIe 80 GB (sm_90) and A100 80 GB GPUs
  (sm_80). Configurations whose tables do not fit in 80 GB are skipped as out of memory.
- **Driver.** If the kernel driver is older than CUDA 13 requires, NVIDIA's forward-compatibility
  user-mode driver (`cuda-compat`, datacenter GPUs) works. The paper's H100 host ran kernel module
  570.148.08 with the 580.126.20 user-mode driver, loaded through `LD_LIBRARY_PATH` and
  `TRITON_LIBCUDA_PATH`. `experiments/h100/env_table5.sh` sets this up; its paths are those of
  the paper's host and need editing elsewhere.
- **Recorded platforms.** `experiments/h100/env/table5_h100.md` and
  `experiments/h100/sweep_fixed/env/table5_h100_fixed.md` (H100),
  `experiments/a100/followups/env/table5_a100.md` (A100 PCIe). Each directory also holds
  `pip-freeze.txt`, `nvidia-smi-q.txt` and `lscpu.txt`.

## Quick start

Set up the environment. `scripts/setup_env.sh` creates a Python environment with the paper's
stack and checks for `nvcc` and the OpenSSL headers; activate that environment before the
commands below. By hand, the equivalent is roughly:

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130
pip install triton==3.7.0
pip install -e '.[dev,plot]'
```

Then, from the repository root:

```bash
export PYTHONPATH=src
mkdir -p results/quick

# Unit tests. The field-guard tests need no GPU; the Triton tests do.
python -m pytest -q

# Triton kernels of one workload against the CPU reference (r=3 and r=10).
python -u experiments/h100/tools/validate_vs_registry.py --bit-width 32 \
  --point-mode specialized --workloads zk_spartan_2 --out results/quick/validate.jsonl

# Triton timing for one workload at a small size.
python -u benchmarks/field_sweep/triton_sweep.py --workloads zk_spartan_2 \
  --bit-widths 32 --rounds 12 --warmups 1 --repeats 5 --point-mode specialized \
  --out results/quick/triton.csv
```

These take minutes, most of it Triton compilation. For a Triton-vs-CUDA comparison, build the
CUDA executable once and run the joined driver:

```bash
cuda/build/build_field_sumcheck.sh        # about 1 h: every workload at every width
python -u benchmarks/field_sweep/compare_triton_cuda.py --skip-build \
  --workloads zk_spartan_2 --bit-widths 32 --rounds 12 --warmups 1 --repeats 5 \
  --point-mode specialized --out results/quick/compare.csv
```

`checksum_match` must be `true`; `cuda_speedup_vs_triton` is Triton time / CUDA time.
`CUDA_ARCH` overrides the default `-arch=native`. If the OpenSSL headers are not in the default
include path, use `experiments/h100/build_field_sumcheck.sh` with `ZKDUEL_OPENSSL_INCLUDE` set.
`results/` is git-ignored.

## Reproducing the paper

```bash
scripts/setup_env.sh
scripts/run_paper_experiments.sh [--dry-run] [--phases "LIST"] [--out DIR] [--skip-gate] \
  [--jobs N] [--include-jellyfish256] [--force]
```

- Output goes to `results/paper_run` (git-ignored) unless `--out DIR` is given.
- `--dry-run` shows what would run. `--phases "LIST"` runs only the listed phases.
- Timing-sensitive phases are preceded by a host-latency gate; `--skip-gate` skips it. The gate
  exists because the paper's host showed launch and synchronization latency drifting during the
  day while GPU-kernel time did not (`experiments/h100/sweep_fixed/probe.log`).
- `zk_jellyfish_zerocheck_hp` at 256 bits is skipped unless `--include-jellyfish256`: its Triton
  compile takes hours.
- See `scripts/run_paper_experiments.sh --help` for `--jobs` and `--force`.

Phases, in order. Durations are approximate, on one H100; a full run is on the order of a
working day.

| Phase | Content | Recorded counterpart (under `experiments/h100/`) | Time |
|---|---|---|---|
| `env` | environment record | `sweep_fixed/env/` | |
| `build` | CUDA executable | | ~1 h |
| `validate` | both backends against the CPU reference | `validation/` | |
| `precompile` | Triton kernels into the cache before any timing (CPU) | | ~1–2 h |
| `sweep` | zkPHIRE tier and diagnostic tier | `sweep_fixed/` | ~1.7 h |
| `split` | wall-clock vs GPU-kernel time (Nsight Systems) | `sweep_fixed/baseline_fixed_all.jsonl`, `sweep_fixed/baseline_fixed_32.jsonl`, `diag/treduce_all.jsonl` | ~1 h |
| `occupancy` | CUDA occupancy ablation | `diag/occupancy.jsonl` | |
| `ablations` | layout x interpolation, CUDA block size | `diag/layout_points.jsonl`, `diag/cuda_block.jsonl` | ~1.5 h |
| `counters` | Nsight Compute counters (needs `sudo`) | `diag/ncu/` | |
| `compile` | Triton compile-time profile | `diag/compile_profile.jsonl`, `diag/compile_passes.jsonl`, `diag/coalesce_scaling.jsonl` | ~1 h |
| `noise` | run-to-run noise and vCPU pinning | `diag/pin_noise.jsonl` | |
| `tables` | summary tables of the run | | |

The A100 results are an existing dataset and are not rerun by the script. On an A100,
`experiments/a100/followups/run_a100_followups.sh` reruns the kernel-level follow-ups
(`experiments/a100/followups/README.md`). The individual scripts behind every recorded dataset
are listed in `experiments/README.md`. More detail: `docs/reproducing.md`.

## Where the reported data lives

Paths are relative to `experiments/`. The analysis is in `h100/FINDINGS.md`; its tables T1–T19
are in `h100/FINDINGS_TABLES.md`, generated from the files below by
`h100/tools/make_findings_tables.py`.

| Result | Data | T-tables |
|---|---|---|
| H100 zkPHIRE tier by width and by family (paper Tables 2 and 4) | `h100/sweep_fixed/zk_32_128.csv`, `h100/sweep_fixed/zk_256.csv` | none; summarize with `summarize_sweep.py` (below) |
| H100 diagnostic tier (paper Figure 4) | `h100/sweep_fixed/diag_tier.csv` | none |
| Wall-clock vs GPU-kernel time, and the `tl.reduce` ablation (paper Table 3) | `h100/sweep_fixed/baseline_fixed_all.jsonl` (32-bit rows from `h100/sweep_fixed/baseline_fixed_32.jsonl`, a rerun on a quiet host), `h100/diag/treduce_all.jsonl` | T19, T5 |
| The same split with the first harness | `h100/diag/baseline_all.jsonl` | T4, T13 |
| Occupancy ablation | `h100/diag/occupancy.jsonl` | T6 |
| Register anatomy, hardware counters | `h100/diag/register_anatomy.jsonl`, `h100/diag/ncu/` | T10, T11 |
| Layout and interpolation ablation; CUDA block size | `h100/diag/layout_points.jsonl`, `h100/diag/cuda_block.jsonl` | T15, T16 |
| Triton compile time (`TritonGPUCoalesce`) | `h100/diag/compile_profile.jsonl`, `h100/diag/compile_passes.jsonl`, `h100/diag/coalesce_scaling.jsonl`, `h100/logs/warm_jellyfish.log` | T12, T17 |
| Run-to-run noise | `h100/diag/pin_noise.jsonl` | T18 |
| CPU-reference validation; r=28 int64-index reruns | `h100/validation/`, `h100/diag/r28_int64_fix.jsonl` | T7 |
| A100 full sweep (A100-SXM4-80GB; portability) | `a100/triton_cuda_field_resilient_r14_28_new.csv` | T1–T3, T14 |
| A100 PCIe kernel-level follow-ups | `a100/followups/diag/`, compared in `a100/followups/COMPARISON.md` | none |
| First H100 sweep; H100 sweep with `tl.reduce`; small-table host vs GPU check | `h100/sweep/`, `h100/sweep_treduce/`, `h100/diag/small_table_64_128.jsonl` | T1–T3, T8, T9 |

The fixed-harness sweep (`h100/sweep_fixed/`, run by `h100/run_fixed_sweep.sh`) uses the
settings of the first H100 sweep (`h100/sweep/`) plus two harness fixes: the int64-index fix in
both backends, and CUDA whole-run timing by a synchronized host timer, as in Triton. It adds the
diagnostic tier and does not rerun `zk_jellyfish_zerocheck_hp` at 256 bits. T1–T3 are computed from
the first sweep. To tabulate the fixed-harness sweep with the same definitions:

```bash
python experiments/h100/tools/summarize_sweep.py experiments/h100/sweep_fixed/zk_32_128.csv \
  experiments/h100/sweep_fixed/zk_256.csv
```

A row is *validated* when `triton_status` and `cuda_status` are `ok` and `checksum_match` is
`true`; the ratio is Triton median / CUDA median.

## Correctness checks

- **Cross-backend agreement.** The comparison drivers record both checksums and
  `checksum_match`. They leave `cuda_speedup_vs_triton` blank for any row that is not validated,
  print `WARNING: CHECKSUM MISMATCH` for each mismatch, and the summary tools exclude such rows
  (`docs/interpreting_results.md`).
- **CPU reference.** `experiments/h100/tools/validate_vs_registry.py` checks the Triton kernels
  against the canonical expressions with plain modular arithmetic, for all 32 workloads, by
  default at r=3 and r=10 with fixed challenges and at r=10 with SHA3 challenges. `experiments/h100/tools/validate_cuda_vs_registry.py`
  checks the CUDA executable against the same reference checksums (the paper's CUDA binary: all
  381 cases, plus jellyfish zerocheck at 256 bits). Recorded results:
  `experiments/h100/validation/` and `experiments/h100/FINDINGS.md` section 4. The older
  `cuda/check/check_field_sumcheck.py` covers 14 of the 32 workloads at r=2–3.
- **256-bit package backend.** `tests/test_zkduel.py` and `tests/test_uint256.py` check the
  Triton backend in `src/zkduel/triton_backend/` against the Python reference.
- **Field-modulus guard.** Both backends share a CIOS Montgomery multiplication that drops its top
  carry word. It is exact only below a modulus bound; the four configured primes satisfy it.
  Constructing a `FieldSpec` in `src/zkduel/fields.py` checks the bound and emulates the kernels'
  word operations against big-integer arithmetic on stressed inputs. An unsafe modulus raises
  `UnsafeModulusError`. `tests/test_field_guard.py` covers this and runs on the CPU:

  ```bash
  python -m pytest -q tests/test_field_guard.py
  ```

## Caveats

- **Wall-clock time depends on host state; GPU-kernel time does not.** Host CPU, VM, driver and
  background load change the per-launch host cost. In T18, the eval-kernel time medians of 20
  repeated configurations agree within 1%, while wall-clock medians deviate by up to 48% (Triton)
  and 32% (CUDA). Compare single configurations on kernel time and use wall-clock only in aggregate.
- **Compile time.** The Triton eval kernels of `zk_jellyfish_zerocheck_hp` at 256 bits take hours
  each to compile. In the profiled compiles (32–128 bits), almost all of the time is in the
  `TritonGPUCoalesce` pass (T12), whose cost grows as O(M·N²) in memory ops M and IR ops N (T17;
  `FINDINGS.md` section 5). The fixed-harness sweep does not rerun this workload at 256 bits, and the
  reproduction script skips it by default.
- **Compiles and timing.** Warm the Triton cache before timed runs (the `precompile` phase). A
  compile running concurrently with a timed run distorts it.
- **Host paths.** `experiments/h100/env_table5.sh` and several scripts under `experiments/`
  contain absolute paths of the paper's hosts.

## Other benchmarks

`docs/benchmarks.md` covers the transcript-inclusive round-by-round timing mode (SHA3
challenges), the earlier 256-bit-only drivers in `benchmarks/u256/`, the wraparound limb-width
ablation, capacity probes, and `scripts/plot_zkduel.py`. `docs/results_summary.md` describes an
earlier NVIDIA L4 run and is kept for reference only.

## Development

```bash
pre-commit install
pre-commit run --all-files
```

## License

MIT; see `LICENSE`.

## Citation

```bibtex
@inproceedings{zkduel-hasp26,
  title     = {zkDuel: Characterizing SumCheck Prover Kernels across GPU Programming Models},
  author    = {Narang, Samarth and Daftardar, Alhad and Reagen, Brandon},
  booktitle = {Proceedings of the Workshop on Hardware and Architectural Support for Security
               and Privacy (HASP '26)},
  address   = {Athens, Greece},
  publisher = {ACM},
  year      = {2026}
}
```
