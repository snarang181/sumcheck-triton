# Reproducing the paper

The top-level `README.md` lists the requirements, the phases of the reproduction script and
where the recorded data lives. This page adds detail.

## 1. Environment

```bash
scripts/setup_env.sh
```

It creates a Python environment with the paper's stack (Python 3.12, PyTorch 2.12.0+cu130,
Triton 3.7.0) and checks for `nvcc` 12.9 and the OpenSSL headers. Activate that environment and
run everything from the repository root with `PYTHONPATH=src`.

The GPU driver must support CUDA 13, which the PyTorch cu130 runtime needs. On a datacenter GPU
with an older kernel driver, NVIDIA's forward-compatibility user-mode driver (`cuda-compat`)
works: put its directory first in `LD_LIBRARY_PATH` and set `TRITON_LIBCUDA_PATH` to it.
`experiments/h100/env_table5.sh` does this for the paper's H100 host; copy it and edit the paths.
The recorded stacks are in `experiments/h100/env/`, `experiments/h100/sweep_fixed/env/` and
`experiments/a100/followups/env/` (`table5_*.md`, `pip-freeze.txt`, `nvidia-smi-q.txt`).

Before a long run, check the setup on one workload (README, "Quick start").

## 2. Full run

```bash
scripts/run_paper_experiments.sh [--dry-run] [--phases "LIST"] [--out DIR] [--skip-gate] \
  [--jobs N] [--include-jellyfish256] [--force]
```

- Phases, in order: `env`, `build`, `validate`, `precompile`, `sweep`, `split`, `occupancy`,
  `ablations`, `counters`, `compile`, `noise`, `tables`.
- Output: `results/paper_run` (git-ignored), or `--out DIR`.
- A full run on one H100 is on the order of a working day. Approximate phase durations:
  `build` ~1 h, `precompile` ~1–2 h (CPU only), `sweep` ~1.7 h, `split` ~1 h, `ablations`
  ~1.5 h, `compile` ~1 h.
- `counters` runs Nsight Compute and needs `sudo`.
- Timing-sensitive phases are preceded by a host-latency gate; `--skip-gate` skips it.
- `zk_jellyfish_zerocheck_hp` at 256 bits is skipped unless `--include-jellyfish256`. Its Triton
  eval kernels take hours each to compile (`experiments/h100/FINDINGS.md` section 5).
- `--dry-run` shows what would run. `scripts/run_paper_experiments.sh --help` describes `--jobs`
  and `--force`.

To run part of it, pass `--phases` with the phases to run. Start with `--dry-run` to see the
plan:

```bash
scripts/run_paper_experiments.sh --dry-run
```

The A100 results are an existing dataset and the script does not rerun them. On an A100,
`experiments/a100/followups/run_a100_followups.sh` reruns the kernel-level follow-ups
(`experiments/a100/followups/README.md`).

## 3. Timing hygiene

- **Keep compiles off timed runs.** Triton compiles on first use. The `precompile` phase fills the
  Triton cache before any timing; a compile running at the same time as a timed run distorts it.
- **Host state.** Wall-clock time includes per-launch host cost, which depends on the CPU, VM,
  driver and background load. The paper's host showed this latency drifting during the day while
  GPU-kernel time stayed constant (`experiments/h100/sweep_fixed/probe.log`); the host-latency
  gate exists for this reason. Kernel time (Nsight Systems) does not depend on host state. T18 in
  `experiments/h100/FINDINGS_TABLES.md` quantifies the run-to-run spread of both.
- **Memory.** Configurations whose tables do not fit in GPU memory are recorded as skipped. On an
  80 GB GPU these are the out-of-memory cases in `experiments/h100/FINDINGS.md` section 5.

## 4. Comparing with the recorded data

The recorded H100 data are in `experiments/h100/`, and `experiments/README.md` names the
script that produced each dataset. `experiments/h100/tools/summarize_sweep.py` computes the
by-width and by-family tables (paper Tables 2 and 4) from any joined sweep CSV, with the same
definitions for a new run and for the recorded one:

```bash
python experiments/h100/tools/summarize_sweep.py experiments/h100/sweep_fixed/zk_32_128.csv \
  experiments/h100/sweep_fixed/zk_256.csv
```

`experiments/h100/tools/make_findings_tables.py` regenerates `FINDINGS_TABLES.md` from the
recorded data, without a GPU; the output matches the committed file:

```bash
mkdir -p results
python experiments/h100/tools/make_findings_tables.py > results/FINDINGS_TABLES.md
diff results/FINDINGS_TABLES.md experiments/h100/FINDINGS_TABLES.md
```

Absolute times depend on the GPU and the host. Compare kernel-time ratios across machines, and
wall-clock ratios only in aggregate.
