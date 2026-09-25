# Interpreting Results

## Runtime Columns

Most CSVs report median runtime in milliseconds after warmup. Matched comparison
CSVs join Triton and CUDA rows by workload, round count, and bit width when
applicable.

## Compile Time

Cold Triton measurements use a fresh `TRITON_CACHE_DIR` and `warmups=0`. Treat
those as compile-plus-first-run measurements. Warm measurements reuse the cache
and represent steady-state runtime.

## Checksums

CUDA comparison and checker scripts report checksums so direct CUDA executables
can be validated without Python bindings. A `checksum_match=false` row means the
Triton and CUDA computations are not comparable until correctness is fixed.

A configuration counts as a validated result only when `triton_status` and
`cuda_status` are both `ok` and `checksum_match` is `true`. The comparison drivers
leave `cuda_speedup_vs_triton` blank for other rows (the medians are still recorded)
and print a `WARNING: CHECKSUM MISMATCH` line and a final count for each mismatch.
`scripts/plot_zkduel.py` and the summary tools under `experiments/` (`summarize_sweep.py`,
`make_findings_tables.py`, `compare_gpus.py`) exclude such rows and name them in their
output. `--include-bad-checksums` overrides this in `plot_zkduel.py`, for debugging
only.

## Wall-clock vs GPU-kernel Time

The diagnostic JSONL files under `experiments/h100/diag/` (written by
`experiments/h100/tools/diag_overhead.py`) split each run into:

- `*_wall_ms`: median wall-clock time per SumCheck run;
- `*_gpu_ms`: summed Nsight Systems durations of the eval, reduce and fold kernels per run;
  `*_eval_ms` is the eval-kernel part;
- `*_host_ms`: wall minus GPU time, i.e. time the GPU is idle;
- `cuda_speedup_wall` and `cuda_speedup_gpu_only`: Triton / CUDA for wall-clock and kernel time;
- `*_launches_per_run`: kernel launches per SumCheck run;
- `*_regs`: registers per thread for each kernel name (minimum and maximum over its launches).

Kernel time does not depend on host state; wall-clock time does. Compare single configurations on
kernel time, and wall-clock only in aggregate (T18 in `experiments/h100/FINDINGS_TABLES.md`).

## Bit-width Studies

Prime-field bit-width results use Montgomery arithmetic over actual primes.
Wraparound bit-width results use `2^k` arithmetic and are only an ablation for
limb count and carry propagation. Do not describe wraparound rows as finite-field
performance.
