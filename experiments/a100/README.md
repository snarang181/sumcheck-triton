# A100 zkPHIRE sweep

`triton_cuda_field_resilient_r14_28_new.csv` is the full Triton-vs-CUDA sweep
over the 25 zkPHIRE workloads, run on the A100 platform of the submission's
Table 5 (NVIDIA A100-SXM4-80GB). It is the data of the submitted version's
Tables 2 and 4 and of the published paper's A100 portability results. Table 4 is
its 256-bit, r = 24 slice (recomputed geomean 1.169 vs. the paper's 1.17x).

`../h100/FINDINGS_TABLES.md` recomputes its tables (T1–T3) and classifies its
skips (T14).

## Provenance

- Date: around 2026-05-19 (file mtime; not recorded in the CSV).
- Stack: not recorded in the CSV. Assumed to be the Table 5 stack.
- Command: reconstructed from the CSV's recorded parameters and log paths.
  The original invocation was not saved.

```bash
PYTHONPATH=src python -u benchmarks/field_sweep/resilient_compare_triton_cuda.py \
  --zk-only --bit-widths 32 64 128 256 \
  --rounds 14 16 18 20 22 24 26 28 \
  --warmups 2 --repeats 10 \
  --point-mode specialized \
  --tmp-dir results/tmp_full_zk_r14_28 \
  --out triton_cuda_field_resilient_r14_28_new.csv
```

Recorded defaults: `timing_mode=whole-run`, `challenge_mode=fixed`,
`layout=element`. There was **no per-case timeout**: `--per-case-timeout` was
added to the driver after this run.

## Contents

800 rows = 25 workloads x 4 widths x 8 round counts. The grid is complete.

- 746 ran on both backends; 744 have `checksum_match=true`.
- 2 ran but have `checksum_match=false`: `zk_vanilla_zerocheck_hp` at 32 and
  64 bits, r = 28. The cause is an int32 index overflow in the harness's
  table-encode kernel, fixed in the released source (`../h100/FINDINGS.md` section 4, T7).
- 54 are `skipped`, all with `skip_reason=returncode=1`.

## Logs

The per-case logs in `results/tmp_full_zk_r14_28/` (the `log_path` column)
were deleted before this was archived. The only record of each skip is
`returncode=1`.

What `returncode=1` does and does not tell you:

- Each case runs `compare_triton_cuda.py`, which runs the Triton sweep first
  and then the CUDA binary, both with `check=True`. Any failure in either
  becomes exit code 1, including a grandchild killed by a signal (for example
  the host OOM killer). The skip rows mark both backends `skipped` regardless of
  which one failed; if Triton failed, CUDA never ran.
- With no timeout, every case terminated on its own; the sweep finished all
  800 cases. So no A100 skip is a hang. Each one exited with an error.

## Skips by footprint

Footprint = `3 x vars x 2^rounds x limbs x 4` bytes (input plus two
full-size fold buffers). The largest case that ran is 77.3 GB
(`zk_witness_non_id`, 256-bit, r = 28).

**36 need more than 77.3 GB** (up to 567 GB). These are out of memory by any
reading.

**6 need 35 to 77.3 GB.** By the footprint formula these fit on the 80 GB
device, so calling them out-of-memory is an inference (CUDA context, Triton
workspace, and allocator overhead on top), not something the evidence shows:

| workload | bits | r | GB |
|---|---:|---:|---:|
| zk_vanilla_permcheck_hp | 32 | 28 | 35.43 |
| zk_jellyfish_zerocheck_hp | 256 | 24 | 35.43 |
| zk_opencheck | 32 | 28 | 38.65 |
| zk_jellyfish_permcheck_hp | 32 | 28 | 48.32 |
| zk_jellyfish_zerocheck_hp | 32 | 28 | 70.87 |
| zk_opencheck | 64 | 28 | 77.31 |

Two of these argue against memory as the cause. `zk_opencheck` 64-bit r = 28
has the same footprint as `zk_witness_non_id` 256-bit r = 28, which ran.
`zk_jellyfish_zerocheck_hp` 256-bit r = 24 is part of a run that fails at every
round count at 256 bits, including 0.03 GB.

**12 need at most 8.9 GB.** Memory is not the cause. The exit path is unknown:

| workload | bits | r | GB |
|---|---:|---:|---:|
| zk_complete_add_5 | 256 | 14 | 0.01 |
| zk_complete_add_6 | 256 | 14 | 0.01 |
| zk_complete_add_12 | 256 | 14 | 0.01 |
| zk_vanilla_permcheck_hp | 256 | 14 | 0.02 |
| zk_jellyfish_permcheck_hp | 256 | 14 | 0.02 |
| zk_jellyfish_zerocheck_hp | 128 | 14 | 0.02 |
| zk_jellyfish_zerocheck_hp | 256 | 14 | 0.03 |
| zk_complete_add_6 | 256 | 16 | 0.05 |
| zk_jellyfish_zerocheck_hp | 256 | 16 | 0.14 |
| zk_jellyfish_zerocheck_hp | 256 | 18 | 0.55 |
| zk_jellyfish_zerocheck_hp | 256 | 20 | 2.21 |
| zk_jellyfish_zerocheck_hp | 256 | 22 | 8.86 |

Most are r = 14, the first (cold-compile) case for each workload and width.

## Related rerun (different GPU)

On an L4, `zk_jellyfish_zerocheck_hp` 256-bit r = 14 was rerun under GNU
`timeout` with a 40-minute limit:

- Triton was still compiling after 40 minutes, with no error, and was killed
  (exit 124).
- CUDA compiled and ran: median 15.73 ms, `eval_kernel<30,22,8,...>` at 252
  registers, no spills.

This shows a Triton compile blow-up on that case. It does not show what ended
the A100 run, which exited with an error rather than timing out. A host-RAM OOM
during compilation fits both, but is unconfirmed, so the A100 record alone does
not show that these cases failed in compilation.

## Measured causes

The H100 sweep (`../h100/sweep/`) reran every configuration with
`--per-case-timeout` and kept the per-case logs. T14 in
`../h100/FINDINGS_TABLES.md` assigns each A100 skip the cause measured there,
and `../h100/FINDINGS.md` section 5 discusses them. Of the 12 small cases, 5
are `zk_jellyfish_zerocheck_hp` at 256 bits (Triton compile longer than the
30-minute limit) and 7 validate on H100.
