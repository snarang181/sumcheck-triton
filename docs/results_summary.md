# Results Summary (historical NVIDIA L4 run)

> **Historical.** This note describes an earlier run on an NVIDIA L4. It is kept for reference;
> its data and numbers are not the paper's. The paper's results (HASP '26) are under
> `experiments/`:
> the H100 PCIe (primary platform) in `experiments/h100/` and the A100 (portability platform) in
> `experiments/a100/`. `experiments/README.md` lists every dataset and the script that produced
> it. `results/` is git-ignored, so the files named below are not in the repository.

It summarized the benchmark artifacts that were then under `results/`.
The recorded environment for the main run is:

```text
GPU: NVIDIA L4, 22.0 GiB, 58 SMs, compute capability 8.9
Driver/CUDA: NVIDIA-SMI 580.126.09, CUDA 13.0
Python: 3.12.3
PyTorch: 2.11.0+cu130
Triton: 3.6.0
```

## Main Experiments of the L4 Run

### 1. All-Polynomial Prime-Field Montgomery Sweep

File:

```text
results/sumcheck_field_triton_cuda_compare_all_polys.csv
```

This was the cleanest apples-to-apples Triton vs CUDA result of the L4 run. It
uses the shared generalized field-sweep backend on both platforms.

Shape:

```text
arithmetic: prime-field-montgomery
workloads: 7 polynomial workloads
bit widths: 32, 64, 128, 256
rounds: 8, 10, 12, 14, 16
rows: 140
checksum_match: true for all rows
triton_status/cuda_status: ok for all rows
```

Both implementations are compile-time specialized by bit width and workload:
Triton uses `LIMBS` / `WORKLOAD` constexprs and CUDA uses template cases
`L=1,2,4,8`. The 256-bit case is the `LIMBS=8` / `L=8` BLS12-381 scalar-field
case.

Median CUDA speedup over Triton by bit width:

| Bit Width | Median Speedup | Median Triton ms | Median CUDA ms |
|---:|---:|---:|---:|
| 32 | 5.887x | 3.642 | 0.693 |
| 64 | 4.116x | 3.639 | 0.891 |
| 128 | 4.028x | 3.633 | 0.931 |
| 256 | 3.669x | 3.717 | 1.055 |

For 256-bit only, median speedup by workload:

| Workload | Median Speedup | Median Triton ms | Median CUDA ms |
|---|---:|---:|---:|
| `poly_a` | 4.376x | 2.165 | 0.523 |
| `poly_ab` | 4.032x | 3.080 | 0.764 |
| `poly_ab_plus_c` | 3.911x | 3.111 | 0.795 |
| `poly_abc` | 3.806x | 4.015 | 1.055 |
| `poly_aabbc` | 3.196x | 5.796 | 1.813 |
| `poly_abc_plus_de` | 2.914x | 4.012 | 1.377 |
| `poly_abcg_plus_deg` | 2.807x | 5.035 | 1.794 |

Median speedup by round across all workloads and bit widths:

| Rounds | N | Median Speedup |
|---:|---:|---:|
| 8 | 256 | 3.650x |
| 10 | 1,024 | 3.803x |
| 12 | 4,096 | 4.029x |
| 14 | 16,384 | 4.299x |
| 16 | 65,536 | 4.442x |

Interpretation: CUDA is consistently faster, but the gap is not catastrophic for
Triton at 256-bit. The gap narrows for higher-arithmetic workloads, which
suggests that polynomial work amortizes some Triton overhead.

### 2. Wraparound Bit-Width Ablation

File:

```text
results/triton_cuda_wraparound_compare.csv
```

This is the supporting ablation for wide-limb overhead. It is not finite-field
performance. Arithmetic is modulo `2^k` rather than modulo a prime.

Shape:

```text
arithmetic: wrap
workload: poly_abc
bit widths: 32, 64, 128, 256
rounds: 8, 10, 12, 14, 16, 18, 20, 22
rows: 32
checksum_match: true for all rows
triton_status/cuda_status: ok for all rows
```

Median CUDA speedup over Triton by bit width:

| Bit Width | Median Speedup | Median Triton ms | Median CUDA ms |
|---:|---:|---:|---:|
| 32 | 6.061x | 5.830 | 0.882 |
| 64 | 6.106x | 5.528 | 0.895 |
| 128 | 4.595x | 5.567 | 1.141 |
| 256 | 3.663x | 6.844 | 1.412 |

At the largest problem size (`rounds=22`, `N=4,194,304`):

| Bit Width | Speedup | Triton ms | CUDA ms |
|---:|---:|---:|---:|
| 32 | 4.437x | 12.053 | 2.717 |
| 64 | 2.799x | 13.008 | 4.647 |
| 128 | 1.796x | 16.573 | 9.230 |
| 256 | 1.352x | 25.129 | 18.582 |

Interpretation: pure limb/carry overhead alone does not explain the full CUDA
advantage in the Montgomery field experiments. At large `N`, the wraparound
256-bit path gets much closer to CUDA, so the Montgomery reduction path and field
kernel structure are important parts of the remaining gap.

### 3. Hand-Only 256-Bit Montgomery Path

File:

```text
results/triton_cuda_poly_ladder_compare.csv
```

This is the older hand-only 256-bit path for both Triton and CUDA:

```text
Triton: src/zkduel/triton_backend/u256_sumcheck.py
CUDA:   cuda/src/u256_sumcheck.cu
```

It computes BLS12-381 scalar-field SumCheck with 8 `uint32` limbs and Montgomery
multiplication. It should be treated separately from the all-polynomial
field-sweep result because it uses a different kernel family.

Shape:

```text
bit width: 256 only
workloads: 7 polynomial workloads
rounds: 8, 10, 12, 14, 16
rows: 35
triton_status/cuda_status: ok for all rows
```

Median CUDA speedup over Triton by round:

| Rounds | N | Median Speedup | Median Triton ms | Median CUDA ms |
|---:|---:|---:|---:|---:|
| 8 | 256 | 6.122x | 3.804 | 0.621 |
| 10 | 1,024 | 7.155x | 5.789 | 0.809 |
| 12 | 4,096 | 8.767x | 9.097 | 1.018 |
| 14 | 16,384 | 10.265x | 13.357 | 1.301 |
| 16 | 65,536 | 11.114x | 18.745 | 1.687 |

Median speedup by workload:

| Workload | Median Speedup |
|---|---:|
| `poly_a` | 9.101x |
| `poly_ab` | 8.767x |
| `poly_ab_plus_c` | 9.122x |
| `poly_abc` | 8.939x |
| `poly_aabbc` | 8.312x |
| `poly_abc_plus_de` | 7.577x |
| `poly_abcg_plus_deg` | 6.635x |

Interpretation: this path shows a much larger CUDA advantage than the generalized
field sweep. Since it uses different kernels, it should be framed as a
hand-specialized 256-bit prototype result, not mixed with the bit-width sweep.

### 4. Large/Capacity Triton Scaling

Files:

```text
results/poly_ladder_large_scaling_resumed.csv
results/poly_ladder_capacity_probe_resumed.csv
```

These are Triton-only 256-bit device-resident scaling runs. They are useful for
showing that the Triton path can push large multilinear tables, but they are not
Triton-vs-CUDA comparisons.

Largest successful capacity-probe rows:

| Workload | Max Rounds | N | Median ms | Input GiB | Peak Estimate GiB |
|---|---:|---:|---:|---:|---:|
| `poly_a` | 28 | 268,435,456 | 612.733 | 8.000 | 13.600 |
| `poly_ab` | 27 | 134,217,728 | 661.152 | 8.000 | 13.600 |
| `poly_ab_plus_c` | 26 | 67,108,864 | 447.640 | 6.000 | 10.200 |
| `poly_abc` | 26 | 67,108,864 | 586.377 | 6.000 | 10.200 |
| `poly_aabbc` | 26 | 67,108,864 | 987.521 | 6.000 | 10.200 |
| `poly_abc_plus_de` | 22 | 4,194,304 | 72.402 | 0.625 | 1.062 |
| `poly_abcg_plus_deg` | 22 | 4,194,304 | 104.619 | 0.750 | 1.275 |

Interpretation: memory scales with `vars * N * 32 bytes` plus scratch/folded
buffers. Higher-var workloads hit practical limits earlier. This is useful for a
capacity/scaling figure, but should be separated from backend speedup figures.

## Supporting / Exploratory Experiments

### Warm Runtime and Cold Compile-Plus-Run

Files:

```text
results/poly_ladder_warm_runtime.csv
results/poly_ladder_cold_compile_plus_run.csv
```

These use the Python API `u256-static-eval` path rather than the device-resident
matched comparison. Warm runtime is useful as historical context; cold
compile-plus-run is very noisy and should not be a main performance claim without
careful rerunning under a fresh cache protocol.

Warm median across workloads:

| Rounds | Median ms |
|---:|---:|
| 8 | 15.504 |
| 10 | 34.671 |
| 12 | 74.115 |
| 14 | 454.397 |
| 16 | 1,141.611 |

### Limb Multiply Microbenchmark

File:

```text
results/bitwidth_mul_chain.csv
```

This is a microbenchmark, not full SumCheck. It estimates limb-multiply scaling
for 32/64/128/256-bit wraparound-style arithmetic.

| Bit Width | Median ms | Effective GMul32/s |
|---:|---:|---:|
| 32 | 0.037 | 113.544 |
| 64 | 0.045 | 278.321 |
| 128 | 0.097 | 431.513 |
| 256 | 0.436 | 346.628 |

### 256-bit Modulus Ablation

File:

```text
results/u256_mod_ablation.csv
```

This compares a 256-bit no-mod wraparound chain with a Montgomery-mod chain.

| Variant | Median ms | Slowdown vs No Mod |
|---|---:|---:|
| `u256-no-mod-wrap` | 0.425 | 1.000x |
| `u256-mont-mod` | 0.440 | 1.035x |

This is a narrow microbenchmark and should be interpreted cautiously. It does not
replace the full SumCheck wraparound and Montgomery comparisons.

## Correctness and Smoke Files

Correctness files include:

```text
results/cuda_poly_ladder_correctness*.csv
results/cuda_sumcheck_field_correctness*.csv
```

Smoke/debug files include names containing:

```text
smoke
tmp_
debug
```

These are useful for development validation, but should not be used as primary
plot sources. The primary plot sources should be:

```text
results/sumcheck_field_triton_cuda_compare_all_polys.csv
results/triton_cuda_wraparound_compare.csv
results/triton_cuda_poly_ladder_compare.csv   # separate hand-only 256-bit result
results/poly_ladder_capacity_probe_resumed.csv # Triton-only capacity/scaling
```
