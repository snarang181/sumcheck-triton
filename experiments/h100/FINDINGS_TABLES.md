
## T1. Paper Table 2, recomputed: A100 (paper run) vs H100 (this run)

A100:

| Width | Validated | Median | Best ratio | Wins | Within 1.1x |
|---:|---:|---:|---:|---:|---:|
| 32 | 195 | 4.39x | 1.897 | 0 | 0 |
| 64 | 196 | 3.58x | 1.238 | 0 | 0 |
| 128 | 188 | 2.58x | 0.891 | 5 | 7 |
| 256 | 165 | 1.60x | 0.640 | 17 | 26 |
| All | 744 | 3.12x | 0.640 | 22 | 33 |

H100:

| Width | Validated | Median | Best ratio | Wins | Within 1.1x |
|---:|---:|---:|---:|---:|---:|
| 32 | 195 | 3.35x | 1.618 | 0 | 0 |
| 64 | 196 | 2.53x | 1.084 | 0 | 1 |
| 128 | 189 | 1.88x | 0.856 | 7 | 12 |
| 256 | 171 | 1.33x | 0.643 | 23 | 33 |
| All | 751 | 2.24x | 0.643 | 30 | 46 |

H100, Triton with tl.reduce (jellyfish zerocheck 256-bit not run):

| Width | Validated | Median | Best ratio | Wins | Within 1.1x |
|---:|---:|---:|---:|---:|---:|
| 32 | 200 | 3.07x | 1.096 | 0 | 1 |
| 64 | 198 | 2.83x | 0.833 | 5 | 11 |
| 128 | 189 | 1.96x | 0.705 | 9 | 13 |
| 256 | 171 | 1.24x | 0.636 | 29 | 50 |
| All | 758 | 1.98x | 0.636 | 43 | 75 |

## T2. Paper Table 4 (256-bit, r=24), A100 vs H100

A100:

| Family (256-bit, r=24) | n | Triton ms | CUDA ms | Speedup | ASIC ms (paper) |
|---|---:|---:|---:|---:|---:|
| Verifiable / Spartan | 3 | 36.70 | 20.55 | 1.79x | 6.31 |
| Halo2 curve checks | 3 | 75.28 | 53.40 | 1.41x | 8.86 |
| Incomplete addition | 2 | 129.77 | 123.10 | 1.05x | 23.06 |
| Complete addition | 12 | 132.45 | 131.37 | 1.01x | 19.95 |
| HyperPlonk vanilla | 2 | 168.84 | 145.69 | 1.16x | 19.81 |
| HyperPlonk Jellyfish | 1 | 555.02 | 474.36 | 1.17x | 30.09 |
| Opening check | 1 | 69.09 | 50.29 | 1.37x | 21.29 |
| All matched | 24 | 110.64 | 94.66 | 1.17x | - |

H100:

| Family (256-bit, r=24) | n | Triton ms | CUDA ms | Speedup | ASIC ms (paper) |
|---|---:|---:|---:|---:|---:|
| Verifiable / Spartan | 3 | 25.64 | 16.51 | 1.55x | 6.31 |
| Halo2 curve checks | 3 | 55.84 | 41.53 | 1.34x | 8.86 |
| Incomplete addition | 2 | 100.53 | 92.24 | 1.09x | 23.06 |
| Complete addition | 12 | 104.11 | 101.23 | 1.03x | 19.95 |
| HyperPlonk vanilla | 2 | 128.89 | 113.25 | 1.14x | 19.81 |
| HyperPlonk Jellyfish | 1 | 404.67 | 387.12 | 1.05x | 30.09 |
| Opening check | 1 | 56.29 | 41.70 | 1.35x | 21.29 |
| All matched | 24 | 84.63 | 73.69 | 1.15x | - |

H100, Triton with tl.reduce:

| Family (256-bit, r=24) | n | Triton ms | CUDA ms | Speedup | ASIC ms (paper) |
|---|---:|---:|---:|---:|---:|
| Verifiable / Spartan | 3 | 24.18 | 16.54 | 1.46x | 6.31 |
| Halo2 curve checks | 3 | 50.97 | 41.01 | 1.24x | 8.86 |
| Incomplete addition | 2 | 92.97 | 92.44 | 1.01x | 23.06 |
| Complete addition | 12 | 99.16 | 100.99 | 0.98x | 19.95 |
| HyperPlonk vanilla | 2 | 126.98 | 113.36 | 1.12x | 19.81 |
| HyperPlonk Jellyfish | 1 | 390.38 | 387.99 | 1.01x | 30.09 |
| Opening check | 1 | 48.50 | 41.53 | 1.17x | 21.29 |
| All matched | 24 | 79.81 | 73.52 | 1.09x | - |

## T3. Same configurations on both GPUs (validated on both)

| Width | Pairs | Median CUDA lead H100 | Median CUDA lead A100 | Lead ratio H100/A100 (geomean) | Triton H100 speedup | CUDA H100 speedup |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 195 | 3.35x | 4.39x | 0.76 | 1.52x | 1.16x |
| 64 | 196 | 2.53x | 3.58x | 0.78 | 1.50x | 1.17x |
| 128 | 188 | 1.88x | 2.58x | 0.78 | 1.55x | 1.21x |
| 256 | 165 | 1.35x | 1.60x | 0.87 | 1.44x | 1.25x |

Status agreement (H100, A100): checksum mismatch / checksum mismatch: 2, skipped / skipped: 47, validated / skipped: 7, validated / validated: 744

## T4. Wall clock vs GPU-kernel time (nsys), 25 workloads, H100

| Width | r | n | Wall-clock CUDA lead (median) | Kernel-only CUDA lead (median, range) | GPU idle share of Triton wall time | Host us/launch Triton / CUDA |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 20 | 25 | 4.03x | 1.36x (1.20-1.42x) | 90% | 21.2 / 4.1 |
| 32 | 24 | 25 | 3.07x | 2.22x (1.33-3.13x) | 59% | 15.2 / 3.3 |
| 256 | 20 | 24 | 1.51x | 1.03x (0.60-1.19x) | 48% | 14.1 / 3.9 |
| 256 | 24 | 24 | 1.24x | 1.13x (0.63-1.36x) | 9% | 9.9 / 3.0 |

All 98 configurations: Triton and CUDA checksums agree: True.

## T5. tl.reduce ablation: kernel-only CUDA lead, original block reduction vs tl.reduce

| Width | r | n | Kernel-only lead: tree -> tl.reduce | Eval-kernel speedup (geomean) | Triton faster than CUDA (kernel-only) | Eval regs Triton tree -> tl.reduce (CUDA), median of max |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 20 | 25 | 1.36x -> 0.93x | 2.21x | 0 -> 24 of 25 | 46 -> 32 (28) |
| 32 | 24 | 25 | 2.22x -> 1.05x | 2.68x | 0 -> 4 of 25 | 46 -> 32 (28) |
| 256 | 20 | 24 | 1.03x -> 0.92x | 1.10x | 8 -> 18 of 24 | 148 -> 146 (127) |
| 256 | 24 | 24 | 1.13x -> 1.01x | 1.10x | 7 -> 9 of 24 | 148 -> 146 (127) |

All 98 tl.reduce configurations: checksums agree with CUDA: True.

## T13. Eval-kernel registers vs eval-kernel gap, all T4 configurations, H100

Registers: maximum over a configuration's eval-kernel variants (points). Eval gap: Triton / CUDA eval-kernel time (nsys). Blocks/SM: register-limited, 128-thread blocks.

| Width | r | n | Triton regs | CUDA regs | Triton uses more | CUDA fits more blocks/SM | Eval gap (range) | Spearman: register difference vs gap | Spearman: occupancy ratio vs gap |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 20 | 25 | 44-46 | 18-32 | 25 of 25 | 25 of 25 (10 vs 16 in all) | 1.34-2.52x | +0.79 | - (constant) |
| 32 | 24 | 25 | 44-46 | 18-32 | 25 of 25 | 25 of 25 (10 vs 16 in all) | 1.53-5.92x | +0.91 | - (constant) |
| 256 | 20 | 24 | 64-216 | 54-166 | 19 of 24 | 11 of 24 | 0.58-1.31x | +0.35 | +0.53 |
| 256 | 24 | 24 | 64-216 | 54-166 | 19 of 24 | 11 of 24 | 0.62-1.61x | +0.17 | +0.32 |

## T6. CUDA occupancy ablation: CUDA eval-kernel time when forced to Triton's occupancy

| Workload | Width | r | Triton eval ms (blocks/SM) | CUDA eval ms at own occupancy (blocks/SM) | CUDA eval ms at <= Triton's occupancy (blocks/SM) | Occupancy cost to CUDA | Actual eval gap Triton/CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| zk_complete_add_3 | 32 | 20 | 1.01 (10) | 0.52 (16) | 0.52 (10) | 1.01x | 1.94x |
| zk_complete_add_3 | 32 | 24 | 10.45 (10) | 3.38 (16) | 3.49 (10) | 1.03x | 3.09x |
| zk_complete_add_3 | 256 | 20 | 11.53 (3) | 19.83 (4) | 26.87 (2) | 1.36x | 0.58x |
| zk_complete_add_3 | 256 | 24 | 111.30 (3) | 179.45 (4) | 298.85 (2) | 1.67x | 0.62x |
| zk_spartan_2 | 32 | 20 | 0.34 (10) | 0.13 (16) | 0.14 (10) | 1.03x | 2.54x |
| zk_spartan_2 | 32 | 24 | 3.44 (10) | 0.58 (16) | 0.73 (10) | 1.25x | 5.93x |
| zk_spartan_2 | 256 | 20 | 0.75 (8) | 0.58 (10) | 0.65 (5) | 1.12x | 1.30x |
| zk_spartan_2 | 256 | 24 | 7.09 (8) | 4.42 (10) | 5.72 (5) | 1.29x | 1.60x |
| zk_vanilla_permcheck_hp | 32 | 20 | 0.96 (10) | 0.55 (16) | 0.55 (10) | 1.01x | 1.76x |
| zk_vanilla_permcheck_hp | 32 | 24 | 9.68 (10) | 4.86 (16) | 4.85 (10) | 1.00x | 1.99x |
| zk_vanilla_permcheck_hp | 256 | 20 | 12.95 (2) | 11.71 (4) | 12.99 (2) | 1.11x | 1.11x |
| zk_vanilla_permcheck_hp | 256 | 24 | 145.70 (2) | 128.63 (4) | 145.20 (2) | 1.13x | 1.13x |
| zk_witness_id_point_1 | 32 | 20 | 0.76 (10) | 0.34 (16) | 0.34 (10) | 1.02x | 2.23x |
| zk_witness_id_point_1 | 32 | 24 | 7.79 (10) | 1.81 (16) | 2.04 (10) | 1.12x | 4.30x |
| zk_witness_id_point_1 | 256 | 20 | 4.26 (5) | 4.10 (5) | 4.10 (5) | 1.00x | 1.04x |
| zk_witness_id_point_1 | 256 | 24 | 45.69 (5) | 39.05 (5) | 39.05 (5) | 1.00x | 1.17x |

## T8. Full-matrix sweeps on H100: Triton with tl.reduce vs original, same configurations

Geomean ratio of wall-clock medians, tl.reduce run / original run, over configurations validated in both (Triton; CUDA is the same binary, shown as a run-to-run control).

| r | 32-bit Triton | 32-bit CUDA | 64-bit Triton | 64-bit CUDA | 128-bit Triton | 128-bit CUDA | 256-bit Triton | 256-bit CUDA |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 14 | 1.00 | 0.99 | 1.15 | 1.00 | 1.07 | 0.99 | 0.94 | 1.00 |
| 16 | 1.03 | 1.00 | 1.08 | 1.00 | 1.09 | 0.99 | 0.83 | 1.00 |
| 18 | 1.01 | 0.99 | 1.10 | 1.01 | 1.12 | 1.01 | 0.83 | 1.00 |
| 20 | 0.97 | 0.98 | 1.08 | 0.98 | 1.06 | 0.99 | 0.90 | 1.00 |
| 22 | 1.02 | 1.00 | 1.01 | 0.99 | 1.02 | 0.99 | 0.94 | 1.00 |
| 24 | 0.89 | 1.00 | 0.93 | 1.00 | 0.94 | 1.00 | 0.94 | 1.00 |
| 26 | 0.65 | 1.00 | 0.80 | 1.00 | 0.86 | 1.00 | 0.93 | 1.00 |
| 28 | 0.46 | 1.00 | 0.71 | 1.00 | 0.80 | 1.00 | 0.87 | 1.00 |

The tl.reduce run also has the int64-offset fix in the encode and fold kernels.

## T9. Small tables at 64/128 bits (r=16): host vs GPU time, original vs tl.reduce (nsys)

| Workload | Width | Triton wall ms | Triton GPU ms | Triton eval-kernel ms | Triton host ms |
|---|---:|---:|---:|---:|---:|
| zk_complete_add_3 | 64 | 10.03 -> 10.45 | 0.762 -> 0.715 | 0.436 -> 0.390 | 9.27 -> 9.73 |
| zk_complete_add_3 | 128 | 9.96 -> 10.29 | 1.285 -> 1.159 | 0.938 -> 0.812 | 8.68 -> 9.13 |
| zk_opencheck | 64 | 4.57 -> 4.88 | 0.314 -> 0.288 | 0.157 -> 0.132 | 4.25 -> 4.59 |
| zk_opencheck | 128 | 4.77 -> 4.67 | 0.448 -> 0.431 | 0.273 -> 0.255 | 4.32 -> 4.24 |
| zk_spartan_2 | 64 | 4.58 -> 4.71 | 0.263 -> 0.230 | 0.113 -> 0.081 | 4.32 -> 4.48 |
| zk_spartan_2 | 128 | 4.74 -> 4.63 | 0.311 -> 0.273 | 0.148 -> 0.111 | 4.43 -> 4.36 |

## T10. Register anatomy of matched eval kernels (point 2, specialized, element layout; static SASS)

| Kernel | Variant | Registers | SASS mul | SASS add | SASS compare/select | SASS shared mem | SASS shuffle | SASS total |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| zk_complete_add_3 32-bit | Triton | 44 | 387 | 169 | 790 | 65 | 0 | 1512 |
| zk_complete_add_3 32-bit | Triton + tl.reduce | 30 | 138 | 49 | 192 | 4 | 8 | 496 |
| zk_complete_add_3 32-bit | CUDA | 26 | 161 | 48 | 82 | 5 | 0 | 416 |
| zk_spartan_2 32-bit | Triton | 46 | 277 | 141 | 663 | 65 | 0 | 1192 |
| zk_spartan_2 32-bit | Triton + tl.reduce | 21 | 28 | 21 | 65 | 4 | 8 | 184 |
| zk_spartan_2 32-bit | CUDA | 18 | 26 | 9 | 22 | 5 | 0 | 120 |
| zk_complete_add_3 256-bit | Triton | 120 | 4793 | 4055 | 2453 | 112 | 0 | 12112 |
| zk_complete_add_3 256-bit | Triton + tl.reduce | 144 | 4644 | 3895 | 2162 | 32 | 64 | 11480 |
| zk_complete_add_3 256-bit | CUDA | 128 | 9392 | 7121 | 1883 | 10 | 0 | 22176 |
| zk_spartan_2 256-bit | Triton | 64 | 617 | 601 | 987 | 112 | 0 | 2576 |
| zk_spartan_2 256-bit | Triton + tl.reduce | 44 | 469 | 441 | 710 | 32 | 64 | 1928 |
| zk_spartan_2 256-bit | CUDA | 54 | 540 | 278 | 256 | 10 | 0 | 1456 |
| zk_vanilla_permcheck_hp 256-bit | Triton | 160 | 3871 | 3272 | 3538 | 112 | 0 | 11632 |
| zk_vanilla_permcheck_hp 256-bit | Triton + tl.reduce | 168 | 3636 | 3114 | 3419 | 32 | 64 | 11128 |
| zk_vanilla_permcheck_hp 256-bit | CUDA | 128 | 4385 | 2633 | 1806 | 10 | 0 | 11208 |
| zk_witness_id_point_1 256-bit | Triton | 86 | 2182 | 1854 | 1625 | 112 | 0 | 6216 |
| zk_witness_id_point_1 256-bit | Triton + tl.reduce | 80 | 2014 | 1694 | 1339 | 32 | 64 | 5552 |
| zk_witness_id_point_1 256-bit | CUDA | 92 | 2438 | 1654 | 738 | 10 | 0 | 5984 |

Static counts: CUDA's modular add/sub branch, so its static SASS includes both arms; use T11 for executed instructions.

## T11. Hardware counters (Nsight Compute), eval kernel of round 0, point 2, rounds=20

| Kernel | Backend | Duration us | Registers | Executed instructions (M) | Theoretical occupancy | Achieved occupancy | Time vs CUDA | Instructions vs CUDA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| zk_complete_add_3 256-bit | Triton | 702.1 | 120 | 198.1 | 25 % | 24.60 % | 0.52x | 0.63x |
| zk_complete_add_3 256-bit | Triton + tl.reduce | 810.3 | 144 | 187.8 | 18.75 % | 18.69 % | 0.60x | 0.59x |
| zk_complete_add_3 256-bit | CUDA | 1341.8 | 128 | 316.4 | 25 % | 24.38 % | 1.00x | 1.00x |
| zk_complete_add_3 32-bit | Triton | 74.1 | 44 | 24.5 | 62.50 % | 54.91 % | 3.03x | 3.25x |
| zk_complete_add_3 32-bit | Triton + tl.reduce | 26.4 | 30 | 7.9 | 100 % | 81.12 % | 1.08x | 1.05x |
| zk_complete_add_3 32-bit | CUDA | 24.4 | 26 | 7.5 | 100 % | 83.25 % | 1.00x | 1.00x |
| zk_spartan_2 256-bit | Triton | 141.2 | 64 | 41.8 | 50 % | 46.16 % | 1.41x | 1.93x |
| zk_spartan_2 256-bit | Triton + tl.reduce | 108.6 | 44 | 31.3 | 62.50 % | 55.88 % | 1.09x | 1.44x |
| zk_spartan_2 256-bit | CUDA | 100.0 | 54 | 21.7 | 56.25 % | 51.93 % | 1.00x | 1.00x |
| zk_spartan_2 32-bit | Triton | 59.3 | 46 | 19.4 | 62.50 % | 54.55 % | 5.02x | 7.42x |
| zk_spartan_2 32-bit | Triton + tl.reduce | 12.4 | 21 | 2.8 | 100 % | 78.78 % | 1.05x | 1.06x |
| zk_spartan_2 32-bit | CUDA | 11.8 | 18 | 2.6 | 100 % | 79.27 % | 1.00x | 1.00x |
| zk_vanilla_permcheck_hp 256-bit | Triton | 1108.8 | 160 | 190.2 | 18.75 % | 18.69 % | 1.17x | 1.25x |
| zk_vanilla_permcheck_hp 256-bit | Triton + tl.reduce | 1134.8 | 168 | 182.1 | 18.75 % | 18.70 % | 1.20x | 1.20x |
| zk_vanilla_permcheck_hp 256-bit | CUDA | 945.7 | 128 | 152.2 | 25 % | 24.89 % | 1.00x | 1.00x |
| zk_witness_id_point_1 256-bit | Triton | 370.8 | 86 | 101.6 | 31.25 % | 30.27 % | 1.07x | 1.20x |
| zk_witness_id_point_1 256-bit | Triton + tl.reduce | 320.6 | 80 | 90.7 | 37.50 % | 36.18 % | 0.93x | 1.07x |
| zk_witness_id_point_1 256-bit | CUDA | 345.9 | 92 | 84.8 | 31.25 % | 30.26 % | 1.00x | 1.00x |

## T12. Triton compile cost of zk_jellyfish_zerocheck_hp's heaviest eval kernel (point 7)

| Triton | Width | Total s | TTIR s | TTGIR s | LLIR s | PTX s | cubin (ptxas) s | LLVM IR lines | SASS instructions | Registers | Peak RSS GB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.7.0 | 32 | 30.0 | 0.18 | 28.37 | 0.44 | 0.1 | 0.22 | 4340 | 6896 | 45 | 0.75 |
| 3.7.0 | 64 | 328.3 | 0.49 | 318.7 | 6.85 | 0.36 | 0.48 | 10774 | 14224 | 96 | 0.8 |
| 3.7.0 | 128 | 2008.6 | 1.82 | 1996.26 | 4.83 | 1.69 | 2.07 | 23739 | 36576 | 159 | 0.97 |
| 3.8.0 | 32 | 28.9 | 0.17 | 25.28 | 1.62 | 0.52 | 0.31 | 4623 | 6912 | 56 | 0.76 |
| 3.8.0 | 64 | 254.5 | 0.86 | 245.87 | 3.5 | 0.69 | 1.36 | 10450 | 14256 | 88 | 0.82 |
| 3.8.0 | 128 | 1658.5 | 1.99 | 1646.18 | 4.77 | 1.8 | 1.81 | 23314 | 36528 | 168 | 0.98 |

Six compiles ran concurrently on a 26-core host alongside other compile jobs, so absolute times are inflated; compare within a table.

Per-pass MLIR timing (`MLIR_ENABLE_TIMING=1`) and stage times from the same compile (`diag/compile_passes.jsonl`, `logs/mlir_passes_*.txt`):

| Triton | Width | Total s | TTGIR stage s | TritonGPUCoalesce s | Coalesce share of TTGIR stage | Next-largest pass |
|---|---:|---:|---:|---:|---:|---|
| 3.7.0 | 32 | 13.4 | 11.48 | 11.1 | 97.0% | TritonGPURemoveLayoutConversions (0.04 s) |
| 3.7.0 | 64 | 128.6 | 124.13 | 122.9 | 99.0% | TritonGPURemoveLayoutConversions (0.13 s) |
| 3.7.0 | 128 | 1529.6 | 1518.93 | 1515.9 | 99.8% | TritonGPUCombineTensorSelectAndIf (0.65 s) |
| 3.8.0 | 32 | 11.6 | 9.5 | 9.1 | 95.8% | TritonGPURemoveLayoutConversions (0.05 s) |
| 3.8.0 | 64 | 102.3 | 98.6 | 97.6 | 99.0% | TritonGPURemoveLayoutConversions (0.14 s) |
| 3.8.0 | 128 | 1259.7 | 1249.84 | 1246.6 | 99.7% | TritonGPUCombineTensorSelectAndIf (0.67 s) |

## T17. TritonGPUCoalesce cost on a synthetic kernel: M memory ops, N IR ops

Kernel: M loads summed into one accumulator, then a chain of C multiply-adds, one store (`tools/coalesce_scaling.py`). Compile only, fresh cache per point, `MLIR_ENABLE_TIMING=1`. If the pass is O(M N^2), the last column stays roughly constant.

| Triton | Varied | Memory ops M | TTIR ops N | Coalesce s | Share of TTGIR pipeline | Coalesce s / (M N^2) x 1e9 |
|---|---|---:|---:|---:|---:|---:|
| 3.7.0 | N | 33 | 3213 | 11.5 | 98.6% | 33.8 |
| 3.7.0 | N | 33 | 6213 | 57.0 | 99.5% | 44.8 |
| 3.7.0 | N | 33 | 12213 | 259.3 | 99.7% | 52.7 |
| 3.7.0 | N | 33 | 24213 | 1110.4 | 99.9% | 57.4 |
| 3.7.0 | M | 17 | 6117 | 33.1 | 99.0% | 52.0 |
| 3.7.0 | M | 33 | 6213 | 63.5 | 99.5% | 49.9 |
| 3.7.0 | M | 65 | 6405 | 139.8 | 99.8% | 52.4 |
| 3.7.0 | M | 129 | 6789 | 274.4 | 99.9% | 46.1 |
| 3.8.0 | N | 33 | 3208 | 11.3 | 98.4% | 33.4 |
| 3.8.0 | N | 33 | 12208 | 265.7 | 99.7% | 54.0 |
| 3.8.0 | M | 17 | 6112 | 29.9 | 98.8% | 47.1 |
| 3.8.0 | M | 65 | 6400 | 124.7 | 99.7% | 46.8 |

Log-log slope of Coalesce time (least squares):

| Triton | vs M (N about fixed) | vs N (M fixed) |
|---|---:|---:|
| 3.7.0 | 1.06 | 2.26 |
| 3.8.0 | 1.07 | 2.36 |

## T15. The paper's two optimizations: layout x interpolation ablation, both backends, H100

Each variant relative to the paper's configuration (element-major, specialized points): geomean over the 5 workloads of variant time / default time. Registers: median over workloads of the eval kernel's maximum.

| Width | r | Layout | Points | n | Triton wall | Triton eval kernel | CUDA wall | CUDA eval kernel | Triton eval regs | CUDA eval regs | Checksums agree (all backends, all variants) |
|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 20 | element | specialized | 5 | 1.00x | 1.00x | 1.00x | 1.00x | 46 | 28 | True |
| 32 | 20 | element | generic | 5 | 0.88x | 1.04x | 0.98x | 1.06x | 44 | 28 | True |
| 32 | 20 | limb | specialized | 5 | 1.01x | 1.00x | 0.98x | 1.00x | 46 | 28 | True |
| 32 | 20 | limb | generic | 5 | 0.99x | 1.04x | 0.96x | 1.06x | 44 | 28 | True |
| 32 | 24 | element | specialized | 5 | 1.00x | 1.00x | 1.00x | 1.00x | 46 | 28 | True |
| 32 | 24 | element | generic | 5 | 1.01x | 1.04x | 1.04x | 1.04x | 44 | 28 | True |
| 32 | 24 | limb | specialized | 5 | 0.93x | 1.00x | 1.06x | 1.00x | 46 | 28 | True |
| 32 | 24 | limb | generic | 5 | 1.01x | 1.04x | 1.02x | 1.05x | 44 | 28 | True |
| 256 | 20 | element | specialized | 5 | 1.00x | 1.00x | 1.00x | 1.00x | 154 | 128 | True |
| 256 | 20 | element | generic | 5 | 1.44x | 1.55x | 1.35x | 1.65x | 119 | 128 | True |
| 256 | 20 | limb | specialized | 5 | 0.95x | 0.97x | 0.96x | 0.95x | 152 | 124 | True |
| 256 | 20 | limb | generic | 5 | 1.46x | 1.56x | 1.33x | 1.62x | 118 | 128 | True |
| 256 | 24 | element | specialized | 5 | 1.00x | 1.00x | 1.00x | 1.00x | 154 | 128 | True |
| 256 | 24 | element | generic | 5 | 1.41x | 1.59x | 1.55x | 1.75x | 119 | 128 | True |
| 256 | 24 | limb | specialized | 5 | 0.99x | 0.99x | 0.97x | 0.96x | 152 | 124 | True |
| 256 | 24 | limb | generic | 5 | 1.43x | 1.63x | 1.55x | 1.77x | 118 | 128 | True |

Per workload, element layout, generic -> specialized points: eval-kernel registers (min-max over interpolation points) and eval-kernel time per SumCheck run (nsys).

| Width | r | Workload | Triton regs | CUDA regs | Triton eval ms | CUDA eval ms | Triton speedup | CUDA speedup |
|---:|---:|---|---:|---:|---:|---:|---:|---:|
| 32 | 20 | zk_complete_add_3 | 44 -> 44-46 | 28 -> 24-28 | 1.00 -> 1.00 | 0.53 -> 0.52 | 1.00x | 1.02x |
| 32 | 20 | zk_opencheck | 44 -> 44 | 32 -> 32 | 0.48 -> 0.42 | 0.28 -> 0.25 | 1.14x | 1.11x |
| 32 | 20 | zk_spartan_2 | 46 -> 46 | 20 -> 16-18 | 0.35 -> 0.34 | 0.14 -> 0.13 | 1.02x | 1.05x |
| 32 | 20 | zk_vanilla_permcheck_hp | 44 -> 44 | 31 -> 28-32 | 0.97 -> 0.95 | 0.60 -> 0.55 | 1.02x | 1.10x |
| 32 | 20 | zk_witness_id_point_1 | 44 -> 46 | 22 -> 17-20 | 0.76 -> 0.76 | 0.35 -> 0.34 | 1.00x | 1.02x |
| 256 | 20 | zk_complete_add_3 | 118-119 -> 115-154 | 128 -> 121-128 | 13.79 -> 11.61 | 22.83 -> 19.79 | 1.19x | 1.15x |
| 256 | 20 | zk_opencheck | 202-204 -> 66-206 | 164 -> 124-128 | 9.00 -> 3.47 | 8.33 -> 2.69 | 2.59x | 3.10x |
| 256 | 20 | zk_spartan_2 | 80 -> 64 | 78 -> 44-54 | 1.13 -> 0.75 | 0.96 -> 0.57 | 1.50x | 1.68x |
| 256 | 20 | zk_vanilla_permcheck_hp | 191-193 -> 154-199 | 168 -> 128 | 19.79 -> 12.93 | 18.33 -> 11.68 | 1.53x | 1.57x |
| 256 | 20 | zk_witness_id_point_1 | 96 -> 84-94 | 94 -> 86-92 | 5.41 -> 4.31 | 5.35 -> 4.08 | 1.26x | 1.31x |

At 32 bits the two layouts are the same memory arrangement (one limb), so the 32-bit limb rows are a noise control for the wall-clock columns.

## T18. Run-to-run reproducibility of the medians (same code, same configuration)

The 20 configurations measured in both the T4 run (`baseline_all`) and the T15 run (`abl_element_specialized`), hours apart; ratio of the two medians.

| Metric | n | Within 2% | Within 5% | Largest deviation |
|---|---:|---:|---:|---:|
| Triton wall-clock | 20 | 12 | 14 | 48% |
| CUDA wall-clock | 20 | 16 | 18 | 32% |
| Triton eval kernels (nsys) | 20 | 20 | 20 | 1% |
| CUDA eval kernels (nsys) | 20 | 20 | 20 | 1% |

One configuration (zk_spartan_2, 32-bit, r=24) pinned to each of the 26 vCPUs (`taskset`), then 10 unpinned runs (`tools/pin_noise.sh`), medians in ms:

| Backend | Pinned: min / median / max | Unpinned: min / median / max |
|---|---:|---:|
| triton | 11.94 / 12.21 / 12.85 | 12.09 / 12.27 / 12.40 |
| cuda | 2.86 / 2.90 / 2.97 | 2.87 / 2.90 / 2.95 |

## T16. CUDA thread-block size (the only CUDA tuning parameter), H100

Geomean over the 5 workloads of time at each block size / time at 128 (the setting used in the paper). Best: workloads whose fastest block size is within 2% of 128's time.

| Width | r | n | wall 64 | eval 64 | wall 128 | eval 128 | wall 256 | eval 256 | wall 512 | eval 512 | 128 within 2% of best (wall) | Checksums identical across sizes |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 20 | 5 | 1.06 | 1.01 | 1.00 | 1.00 | 0.95 | 1.04 | 0.92 | 1.11 | 0 of 5 | True |
| 32 | 24 | 5 | 1.11 | 1.06 | 1.00 | 1.00 | 0.98 | 1.01 | 1.00 | 1.05 | 1 of 5 | True |
| 256 | 20 | 5 | 1.01 | 0.99 | 1.00 | 1.00 | 1.01 | 1.04 | 1.11 | 1.20 | 5 of 5 | True |
| 256 | 24 | 5 | 1.02 | 1.01 | 1.00 | 1.00 | 1.00 | 1.00 | 1.20 | 1.15 | 4 of 5 | True |

## T7. r=28 configurations with more than 2^31 table elements, with the int64-offset fix

| Workload | Width | A100 sweep | H100 sweep | With fix: checksums agree | Wall lead | Kernel-only lead |
|---|---:|---|---|---|---:|---:|
| zk_vanilla_zerocheck_hp | 32 | checksum mismatch | checksum mismatch | True | 2.01x | 1.65x |
| zk_vanilla_zerocheck_hp | 64 | checksum mismatch | checksum mismatch | True | 1.41x | 1.12x |
| zk_vanilla_permcheck_hp | 32 | skipped | skipped | True | 1.87x | 1.60x |
| zk_vanilla_permcheck_hp | 64 | validated | validated | True | 1.32x | 1.13x |
| zk_jellyfish_zerocheck_hp | 32 | skipped | skipped | True | 1.52x | 1.36x |
| zk_jellyfish_zerocheck_hp | 64 | skipped | skipped | does not fit in 80 GB | - | - |
| zk_jellyfish_permcheck_hp | 32 | skipped | skipped | True | 1.68x | 1.47x |
| zk_jellyfish_permcheck_hp | 64 | skipped | skipped | does not fit in 80 GB | - | - |
| zk_opencheck | 32 | skipped | skipped | True | 1.43x | 1.06x |
| zk_opencheck | 64 | skipped | skipped | True | 1.13x | 0.75x |

## T14. The paper's 54 A100 skips (42 capacity / 12 non-capacity) by measured cause

Paper's class: footprint 3 x vars x 2^r x limbs x 4 bytes above 9 GB is 'capacity' (Section 5). Cause: the H100 run's per-case log for the same configuration.

| Paper's class | Measured cause | Cases | Configurations |
|---|---|---:|---|
| capacity | Triton compile > 30 min (jellyfish zerocheck, 256-bit) | 1 | zk_jellyfish_zerocheck_hp 256-bit r=24 (35 GB) |
| capacity | int32 index overflow in encode kernel (illegal memory access) | 5 | zk_vanilla_permcheck_hp 32-bit r=28 (35 GB); zk_jellyfish_zerocheck_hp 32-bit r=28 (71 GB); zk_jellyfish_permcheck_hp 32-bit r=28 (48 GB); zk_opencheck 32-bit r=28 (39 GB); zk_opencheck 64-bit r=28 (77 GB) |
| capacity | out of memory (footprint > 80 GB) | 36 | footprints 90.2-566.9 GB |
| non-capacity | Triton compile > 30 min (jellyfish zerocheck, 256-bit) | 5 | zk_jellyfish_zerocheck_hp 256-bit r=14 (0.035 GB); zk_jellyfish_zerocheck_hp 256-bit r=16 (0.14 GB); zk_jellyfish_zerocheck_hp 256-bit r=18 (0.55 GB); zk_jellyfish_zerocheck_hp 256-bit r=20 (2.2 GB); zk_jellyfish_zerocheck_hp 256-bit r=22 (8.9 GB) |
| non-capacity | validates on H100; on A100 the first round count(s) run for its workload and width | 7 | zk_complete_add_5 256-bit r=14 (0.011 GB); zk_complete_add_6 256-bit r=14 (0.013 GB); zk_complete_add_6 256-bit r=16 (0.05 GB); zk_complete_add_12 256-bit r=14 (0.013 GB); zk_vanilla_permcheck_hp 256-bit r=14 (0.017 GB); zk_jellyfish_zerocheck_hp 128-bit r=14 (0.017 GB); zk_jellyfish_permcheck_hp 256-bit r=14 (0.024 GB) |

## T19. Paper Table 3: wall clock vs GPU-kernel time on the fixed harness, H100

`sweep_fixed/baseline_fixed_all.jsonl`, 32-bit rows from `sweep_fixed/baseline_fixed_32.jsonl`; tl.reduce kernel-only ratio from `diag/treduce_all.jsonl` (T5). Ratios are Triton/CUDA.

| Width | r | n | Wall T/C | Kernel T/C (range) | +tl.reduce kernel T/C | No-kernel share of Triton wall | Host us/launch Triton / CUDA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 20 | 25 | 4.02 | 1.38 (1.21-1.43) | 0.93 (n=25) | 91% | 21.9 / 4.1 |
| 32 | 24 | 25 | 3.17 | 2.20 (1.33-3.14) | 1.05 (n=25) | 60% | 15.7 / 3.3 |
| 256 | 20 | 24 | 1.52 | 1.03 (0.60-1.19) | 0.92 (n=24) | 49% | 14.9 / 4.2 |
| 256 | 24 | 24 | 1.24 | 1.12 (0.63-1.35) | 1.01 (n=24) | 10% | 10.7 / 3.2 |
