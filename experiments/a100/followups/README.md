# A100 follow-ups

One script, `run_a100_followups.sh`, reruns the H100 follow-ups from
`experiments/h100/FINDINGS.md` on an A100. The recorded run used an A100 80GB PCIe (sm_80), not
the A100-SXM4 host of the full sweep in `../`. It answers:

1. Does the A100's Triton-vs-CUDA gap split into host time and GPU-kernel time the same way
   the H100's does? (wall clock vs `nsys` kernel time)
2. Does the `tl.reduce` block reduction close the kernel gap on sm_80 too?
3. Occupancy ablation on sm_80: how much does CUDA slow down when forced to
   Triton's occupancy?
4. With the int64-offset fix, what are the A100 numbers for the r=28 cases that crashed or
   mismatched in the paper run?
5. The two small A100 skips (`zk_complete_add_12` and `zk_vanilla_permcheck_hp`, both 256-bit,
   r=14), rerun from a cold Triton cache with full logs, peak memory (`/usr/bin/time -v`) and
   `dmesg`, to record why they failed.

## Before running

On the A100 host, from the repository root, check the following.

- **Environment:** the paper's Table 5 environment is active: `python` has torch 2.12.0+cu130 and
  Triton 3.7.0, `nvcc` is 12.9, and `libssl-dev` is installed. Set `PY=/path/to/python` if the
  right interpreter isn't `python`.
- **nsys:** `nsys --version` works. If not, on Ubuntu 24.04:
  ```bash
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2025.1.3_2025.1.3.140-1_amd64.deb
  sudo apt install ./nsight-systems-2025.1.3_2025.1.3.140-1_amd64.deb
  ```
- **GPU and disk:** the GPU is idle (the script refuses to start otherwise). About 20 GB of disk
  is free for Triton caches.
- **Reused artifacts, if they still exist:**
  - `.triton-cache-sumcheck-field` in the repo root (the paper sweep's Triton cache) saves most of
    the baseline compile time.
  - `build/field_sumcheck_cuda` (the paper's CUDA binary) is used as is. It is otherwise rebuilt
    from the current source, which takes about an hour.

## Run

Run detached so a dropped SSH session doesn't stop it. The script commits its outputs after each
phase; with `PUSH=1` it also pushes them to branch `a100-followups` using the repository's
`origin` credentials (used while collecting the recorded data).

```bash
mkdir -p experiments/a100/followups/logs
setsid nohup experiments/a100/followups/run_a100_followups.sh \
  > experiments/a100/followups/logs/console.log 2>&1 < /dev/null &
tail -f experiments/a100/followups/logs/run.log
```

Variants:

| Command prefix | What it does |
|---|---|
| (none) | All phases, 8-workload 256-bit subset |
| `QUICK=1` | 4-workload 256-bit subset (shorter compile) |
| `PHASES="0 1 2"` | Skip the cold-compile reproduction (phase 3) |
| `PHASES="3"` | Only the cold-compile reproduction |
| `JOBS=N` | Parallel compiles (default: all cores) |

## Phases and time (12-core host; rough)

| Phase | What | Time |
|---|---|---|
| 0 | Preflight; environment in Table 5 format (`env/table5_a100.md`) | minutes, plus about 1 h if the CUDA binary must be built |
| 1 | Compile Triton kernels (original and `tl.reduce`): all 25 workloads at 32 bits, the r=28 workloads at 64 bits, and the 256-bit subset. Build the occupancy binary. CPU only. | 0.5–1 h with a warm cache; 2–4 h cold (`QUICK=1`: about half) |
| 2 | On a quiet GPU: wall vs kernel split for both variants, occupancy ablation, r=28 reruns, sm_80 CUDA registers, `COMPARISON.md` | about 1–1.5 h |
| 3 | Cold-compile reproduction of the two skips (4 h cap each, run in parallel) | up to 4 h |

Phase 2 does not start until phase 1's compiles have finished, and phase 3 runs last, so no
timing run overlaps a compile.

## Output

Everything goes under `experiments/a100/followups/`:

| Path | Contents |
|---|---|
| `COMPARISON.md` | A100 vs H100 tables |
| `diag/*.jsonl` | Raw results |
| `env/` | Environment record |
| `logs/` | Logs |
| `cold/` | Phase-3 case logs (phase 3 has not been run; `logs/run.log` ends after phase 2) |

The recorded run used an A100 80GB PCIe (`env/nvidia-smi-q.txt`), not the submission's
A100-SXM4 host. It ran phases 0–2.
