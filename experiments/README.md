# experiments

Data and scripts behind the results of the paper *zkDuel: Characterizing SumCheck Prover Kernels
across GPU Programming Models* (HASP '26). The NVIDIA H100 PCIe is the primary platform and the
A100 the portability platform.

- `h100/FINDINGS.md`: the H100 analysis, including the reproduction of the submitted version's
  A100 sweep on the H100.
- `h100/FINDINGS_TABLES.md`: every table cited there (T1–T19), generated from the data by
  `h100/tools/make_findings_tables.py`.
- `a100/README.md`: the A100-SXM4 sweep and its provenance.
- `a100/followups/COMPARISON.md`: A100 vs H100 kernel-level follow-ups.

The top-level `README.md` maps each result of the paper to its data file. To rerun the
experiments, use `scripts/run_paper_experiments.sh` (`docs/reproducing.md`); the scripts below are
the ones that produced the recorded data.

## Experiments

Paths are relative to this folder. "Tables" are the T-tables in `h100/FINDINGS_TABLES.md`.
Numbers are in those tables, not here.

### H100 sweeps

| Experiment | Question it answers | Script or tool | Output | Tables |
|---|---|---|---|---|
| First H100 sweep, 800 configs, Table 5 stack | Does the submitted version's A100 result reproduce on a second GPU generation? | `h100/run_sweep.sh`; summaries by `h100/tools/summarize_sweep.py` and `h100/tools/classify_skips.py` (run by `h100/after_sweep.sh`) | `h100/sweep/` (joined CSV, `cases/`, `summary_h100.md`, `summary_a100.md`, `skips_h100.md`) | T1–T3 |
| Full-matrix sweep with `tl.reduce` and the int64-offset fix (792 configs) | Does the block-reduction change hold across the whole matrix? | `h100/run_sweep_treduce.sh`, `h100/tools/triton_sweep_treduce.py` | `h100/sweep_treduce/` | T1, T2, T8 |
| Fixed-harness sweep (the paper's H100 sweep) | The paper's H100 numbers (Tables 2 and 4, Figure 4): int64-index fix in both backends, CUDA whole-run timing by a synchronized host timer as in Triton, otherwise the main sweep's settings; `zk_jellyfish_zerocheck_hp` at 256 bits not rerun | `h100/run_fixed_sweep.sh`; summaries by `h100/tools/summarize_sweep.py` | `h100/sweep_fixed/`: `diag_tier.csv` (7 polynomials, Figure 4), `zk_32_128.csv`, `zk_256.csv` (zkPHIRE tier), `baseline_fixed.jsonl` (wall vs kernel split, 3 workloads), per-case logs in `cases_*/`, commands in `*.command.txt`; `probe.csv`, `probe.log` (host-latency gate); `env/` | none (not in `FINDINGS_TABLES.md`) |

### H100 diagnostics: where the gap comes from

| Experiment | Question it answers | Script or tool | Output | Tables |
|---|---|---|---|---|
| Wall-clock vs GPU-kernel time (nsys), 25 workloads, fixed harness | Is the gap in the kernels or on the host? (paper Table 3, with the `tl.reduce` run below) | `h100/tools/diag_overhead.py` (run by `h100/run_fixed_split.sh`; 32-bit rows rerun on a quiet host by `h100/run_fixed_split32.sh`) | `h100/sweep_fixed/baseline_fixed_all.jsonl`, `h100/sweep_fixed/baseline_fixed_32.jsonl`, raw runs in `h100/diag/raw/baseline_fixed_all/`, `h100/diag/raw/baseline_fixed_32/` | T19 |
| Wall-clock vs GPU-kernel time (nsys), 25 workloads, first harness | The same split before the harness fixes | `h100/tools/diag_overhead.py` (run by `h100/after_sweep.sh`) | `h100/diag/baseline_all.jsonl`, raw runs in `h100/diag/raw/baseline_all/` | T4 |
| Registers vs kernel gap, all 98 kernel-split configurations | Do register counts track the kernel gap across all configurations? | `h100/tools/make_findings_tables.py` | from `h100/diag/baseline_all.jsonl` | T13 |
| `tl.reduce` block-reduction ablation | Where do Triton's extra instructions and registers come from? | `h100/tools/treduce.py`, `h100/tools/triton_sweep_treduce.py`, `h100/tools/diag_overhead.py` | `h100/diag/treduce_all.jsonl` | T5 |
| Small tables at 64/128 bits (r=16), host vs GPU | Why is the `tl.reduce` sweep slower on small tables? | `h100/tools/diag_overhead.py` (labels `small64_base`, `small64_treduce`) | `h100/diag/small_table_64_128.jsonl` | T9 |
| CUDA occupancy ablation (shared-memory padding) | Does occupancy explain the gap? | `h100/patches/cuda_occupancy_knob.patch`, `h100/build_ablation_binary.sh`, `h100/tools/diag_overhead.py` | `h100/diag/occupancy.jsonl` | T6 |
| Register anatomy (SASS, PTX) | Why does Triton use more registers? | `h100/tools/register_anatomy.py` | `h100/diag/register_anatomy.jsonl` | T10 |
| Hardware counters (Nsight Compute) | Executed instructions and achieved occupancy | `h100/tools/ncu_anatomy.sh`, `h100/tools/ncu_summary.py` | `h100/diag/ncu/` | T11 |
| First wall vs kernel split (3 workloads, r=14/20/24) | Earlier, smaller run of the same split | `h100/tools/diag_overhead.py` (its default output) | `h100/diag/overhead.jsonl`, raw runs at the top of `h100/diag/raw/` | none |

### H100 optimization ablations and noise

| Experiment | Question it answers | Script or tool | Output | Tables |
|---|---|---|---|---|
| Layout x interpolation ablation, both backends | Evaluate the paper's two proposed optimizations (limb-major layout, specialized points) | `h100/ablations.sh` | `h100/diag/layout_points.jsonl` | T15 |
| CUDA thread-block size sweep | Is the CUDA baseline tuned? | `h100/ablations.sh` | `h100/diag/cuda_block.jsonl` | T16 |
| Run-to-run noise and vCPU pinning | How much of a wall-clock difference is run-to-run noise? Does pinning to a vCPU remove it? | `h100/tools/pin_noise.sh` (started by `h100/tools/after_ablations.sh`); the repeat comparison by `h100/tools/make_findings_tables.py` | `h100/diag/pin_noise.jsonl`; repeats from `baseline_all.jsonl` vs `layout_points.jsonl` | T18 |

### H100 correctness and skips

| Experiment | Question it answers | Script or tool | Output | Tables |
|---|---|---|---|---|
| Registry validation, all 32 workloads, both backends | Correctness gating against the canonical polynomials | `h100/tools/validate_vs_registry.py`, `h100/tools/run_validation.sh`, `h100/tools/validate_cuda_vs_registry.py` | `h100/validation/` | none (`h100/FINDINGS.md` section 4) |
| r=28 reruns with the int64-offset fix | The r=28 skips and checksum mismatches | `h100/tools/overflow_fix.py`, `h100/tools/triton_sweep_fixed.py`, `h100/patches/cuda_int64_index.patch`; fixed in the released source | `h100/diag/r28_int64_fix.jsonl` | T7 |
| The submitted version's 54 A100 skips by measured cause | What caused the skips, including the 12 classed as non-capacity? | `h100/tools/classify_skips.py`, `h100/tools/make_findings_tables.py` | from `a100/triton_cuda_field_resilient_r14_28_new.csv` and the per-case logs in `h100/sweep/cases/` | T14 |

### H100 compile time (jellyfish zerocheck)

| Experiment | Question it answers | Script or tool | Output | Tables |
|---|---|---|---|---|
| Compile profiling, Triton 3.7.0 and 3.8.0 | Root cause of the jellyfish compile failures: which stage? | `h100/tools/compile_profile.py`, `h100/tools/run_compile_profiles.sh` | `h100/diag/compile_profile.jsonl`, `h100/logs/compile_profile_*.log` | T12 |
| Same-process pass timing | Which MLIR pass takes the TTGIR stage's time? | `h100/tools/compile_profile.py`, `h100/tools/run_compile_passes.sh` (`MLIR_ENABLE_TIMING=1`) | `h100/diag/compile_passes.jsonl`, `h100/logs/mlir_passes_{3.7.0,3.8.0}_{32,64,128}.txt`; earlier separate-compile timing in `h100/logs/mlir_timing_*.txt` | T12 |
| jellyfish warm-up compile times, 128 and 256 bits | How long do these eval kernels take to compile? | `h100/tools/run_warm_jellyfish.sh` | `h100/logs/warm_jellyfish.log` | none (`h100/FINDINGS.md` section 5) |
| `TritonGPUCoalesce` scaling on a synthetic kernel | Is the pass O(M·N²), as its source implies? | `h100/tools/coalesce_scaling.py`, `h100/tools/run_coalesce_scaling.sh` | `h100/diag/coalesce_scaling.jsonl` | T17 |
| Wide loads (`vecload`), kernel side | Does cutting memory ops in the kernel fix the compile time? | `h100/tools/vecload.py`, `h100/tools/triton_sweep_vecload.py`, `h100/tools/compile_profile.py --variant vecload` | `h100/logs/mlir_passes_vecload_3.7.0_32.txt` | none; negative result (`h100/FINDINGS.md` section 5) |
| Coalesce slice-cache patch: IR equivalence | Does the compiler-side fix give byte-identical IR? | `h100/patches/triton_coalesce_slice_cache.patch`, `h100/tools/ir_hashes.py`, `h100/tools/run_ir_equivalence.sh` | `h100/diag/ir_equivalence.jsonl` (stock 3.7.0 only; the patched build is not measured yet) | none |

### A100

| Experiment | Question it answers | Script or tool | Output | Tables |
|---|---|---|---|---|
| Full 800-config sweep, A100-SXM4-80GB (the submission's platform) | The submission's Tables 2 and 4; the portability comparison | `benchmarks/field_sweep/resilient_compare_triton_cuda.py` (command reconstructed in `a100/README.md`) | `a100/triton_cuda_field_resilient_r14_28_new.csv` | T1–T3, T14 |
| A100 follow-ups (A100 80GB PCIe, sm_80; not the submission's SXM4 host) | The same kernel-only split, `tl.reduce`, occupancy and r=28 results on sm_80 | `a100/followups/run_a100_followups.sh`, `a100/followups/compare_gpus.py` | `a100/followups/COMPARISON.md`, `a100/followups/diag/`, `a100/followups/env/`, `a100/followups/logs/` | none (`COMPARISON.md` sections 1–5) |

The A100 follow-ups ran phases 0–2. Phase 3 (the cold-compile reproduction of two A100 skips)
has not been run, so `a100/followups/cold/` does not exist.

### Environment and support scripts

| What | Where |
|---|---|
| Table 5 stack and its record | `h100/env_table5.sh`, `h100/tools/capture_env.py`, `h100/env/`, `h100/sweep_fixed/env/`, `a100/followups/env/` |
| CUDA builds (main, ablation variants) | `h100/build_field_sumcheck.sh`, `h100/build_ablation_binary.sh`, `h100/patches/` |
| Triton cache warm-up before timed runs | `h100/tools/warm_cache.py`, `h100/tools/run_warm_cache.sh`, `h100/tools/precompile_treduce.py`, `h100/tools/run_precompile_treduce.sh` |
| Scheduling (keep compiles off timed runs) | `h100/start_when_ready.sh`, `h100/tools/compile_deadline.sh`, `h100/tools/guard_sweep.sh`, `h100/tools/hold_ablation_timing.sh` |
