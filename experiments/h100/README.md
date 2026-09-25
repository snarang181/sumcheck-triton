# H100 PCIe: sweeps and diagnostics

The H100 PCIe is the paper's primary platform. This folder holds its sweeps, the follow-up
diagnostics, and the scripts that produced them.

**Results: see [`FINDINGS.md`](FINDINGS.md)** (tables generated from the data in
[`FINDINGS_TABLES.md`](FINDINGS_TABLES.md)).

The sweep is the 800-configuration matrix: 25 zkPHIRE workloads x 32/64/128/256-bit fields x
rounds 14..28 (the source of Tables 2 and 4), run with the software stack of the submitted
paper's Table 5 held fixed. There are two H100 runs of it:

- `sweep/`: the first run, before the two harness fixes listed next. `FINDINGS.md` and T1–T3
  use it.
- `sweep_fixed/`: the fixed-harness run behind the paper's H100 numbers (Tables 2 and 4, and the
  diagnostic tier for Figure 4). It adds the int64-index fix in both backends and CUDA
  whole-run timing by a synchronized host timer, as in Triton; it does not rerun
  `zk_jellyfish_zerocheck_hp` at 256 bits. Summarize it with `tools/summarize_sweep.py`.

## Environment

`env_table5.sh` pins the stack; `env/table5_h100.md` is the recorded side-by-side with
Table 5 (regenerate with `python experiments/h100/tools/capture_env.py`).

| Component | Table 5 (A100) | This run | How it was obtained |
|---|---|---|---|
| GPU | A100-SXM4-80GB, 108 SMs, sm_80 | H100 PCIe 80GB, 114 SMs, sm_90 | intended change |
| Driver | 580.126.20 | kernel module 570.148.08; user-mode driver 580.126.20 | `cuda-compat-13-0_580.126.20` (NVIDIA forward compatibility for datacenter GPUs), loaded via `LD_LIBRARY_PATH` only |
| nvcc | 12.9 | 12.9.1 | NVIDIA redist tarballs in `~/opt/cuda-12.9.1` |
| PyTorch | 2.12.0+cu130 | 2.12.0+cu130 | uv venv `~/venvs/zkduel-table5` |
| Triton | 3.7.0 | 3.7.0 | same venv (Triton's bundled ptxas) |
| Python | not listed | 3.12.14 | uv |
| CPU | Xeon @ 2.20GHz, 12 threads | Xeon Platinum 8480+, 26 threads | host differs; matters for Python launch overhead |
| OS | Ubuntu 24.04.4 | Ubuntu 22.04.5 | host compiler / glibc only |

OpenSSL headers for the SHA3 path come from the `libssl-dev` 3.0.2 package unpacked into
`~/opt/openssl-dev` (the system has only the `libssl3` runtime), so the CUDA binary is
built with `build_field_sumcheck.sh` here: the same nvcc flags as
`cuda/build/build_field_sumcheck.sh` plus the include path.

## Sweep settings

From the A100 run's CSV: `--zk-only`, widths 32/64/128/256, rounds 14..28 (step 2),
`--warmups 2 --repeats 10 --point-mode specialized`, defaults otherwise (whole-run timing,
fixed challenge, element layout). Added for this run: `--per-case-timeout 1800` (the A100 run
had no timeout; its skips are `returncode=1`). The fixed-harness run uses the same timing settings and timeout. The exact
commands are in `sweep/command.txt` (written by `run_sweep.sh`) and `sweep_fixed/*.command.txt`
(written by `run_fixed_sweep.sh`).

Harness change: the resilient driver kills a timed-out case's whole process group. Before that fix,
a timed-out case's Triton or CUDA worker kept running while later cases were being timed.

Triton cache: `tools/run_validation.sh` compiles every kernel the sweep launches, into the
sweep's `TRITON_CACHE_DIR`, before the sweep starts. Medians exclude compile time either way;
warming the cache keeps slow 256-bit compiles from hitting the new 1800 s timeout, which the
A100 run (no timeout) could not hit. `first_ms` in this run therefore excludes compilation.

## Contents

The full index of experiments, with the script and T-table for each dataset, is in
[`../README.md`](../README.md).

| Path | What |
|---|---|
| `run_sweep.sh` | the sweep (run detached) |
| `push_checkpoints.sh` | commits and pushes checkpoints every 30 min during a run (used while collecting the data) |
| `sweep/` | joined CSV, per-case CSVs and logs, command, start/finish times |
| `sweep_treduce/` | the same sweep with Triton's block reduction replaced by `tl.reduce` (`run_sweep_treduce.sh`) |
| `sweep_fixed/` | the fixed-harness sweep (`run_fixed_sweep.sh`): zkPHIRE tier, diagnostic tier, a 3-workload wall vs kernel split, host-latency probe, environment |
| `validation/` | independent-reference check of every Triton kernel the sweep uses (see below) |
| `diag/` | follow-up measurements: GPU-kernel vs host time (nsys), ablations, compile profiles, noise |
| `tools/`, `patches/` | the scripts and source patches that produce `diag/` |
| `env/`, `logs/` | environment capture, build/GPU-monitor logs |

`tools/validate_vs_registry.py` checks the Triton field-sweep kernels against the canonical
polynomials in `src/zkduel/workloads.py` with plain Python modular arithmetic (not the
transliterated evaluators in `cuda/check/check_field_sumcheck.py`, which cover 14 of the 32
workloads). Combined with the sweep's Triton-vs-CUDA checksum agreement, this gives a CPU
reference check for all 25 zkPHIRE workloads.
