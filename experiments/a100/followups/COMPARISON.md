# A100 vs H100 follow-ups

Leads are CUDA's (Triton time / CUDA time); kernel-only uses summed nsys kernel durations.

## 1. Original Triton: wall clock vs GPU-kernel time

| Width | r | n | Wall lead A100 | Wall lead H100 | Kernel-only lead A100 | Kernel-only lead H100 | GPU idle share of Triton wall A100 / H100 | Triton host us/launch A100 / H100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 20 | 25 | 3.44x | 4.03x | 1.42x | 1.36x | 84% / 90% | 17.0 / 21.2 |
| 32 | 24 | 25 | 2.96x | 3.07x | 2.37x | 2.22x | 45% / 59% | 11.9 / 15.2 |
| 256 | 20 | 4 | 1.50x | 1.67x | 1.09x | 1.06x | 42% / 53% | 11.2 / 14.3 |
| 256 | 24 | 4 | 1.22x | 1.24x | 1.16x | 1.14x | 8% / 13% | 7.9 / 10.4 |

Checksums agree (A100 records): True.

## 2. Triton with tl.reduce: wall clock vs GPU-kernel time

| Width | r | n | Wall lead A100 | Wall lead H100 | Kernel-only lead A100 | Kernel-only lead H100 | GPU idle share of Triton wall A100 / H100 | Triton host us/launch A100 / H100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 20 | 25 | 3.54x | 4.13x | 0.98x | 0.93x | 89% / 93% | 18.7 / 22.6 |
| 32 | 24 | 25 | 2.32x | 2.77x | 1.08x | 1.05x | 67% / 77% | 13.9 / 17.6 |
| 256 | 20 | 4 | 1.49x | 1.50x | 0.97x | 0.95x | 45% / 51% | 12.0 / 9.0 |
| 256 | 24 | 4 | 1.14x | 1.17x | 1.07x | 1.05x | 10% / 15% | 8.6 / 11.0 |

Checksums agree (A100 records): True.

## 3. Effect of tl.reduce on the kernel-only CUDA lead

| GPU | Width | r | n | Kernel-only lead original -> tl.reduce | Eval-kernel speedup (geomean) |
|---|---:|---:|---:|---:|---:|
| A100 | 32 | 20 | 25 | 1.42x -> 0.98x | 2.11x |
| A100 | 32 | 24 | 25 | 2.37x -> 1.08x | 2.79x |
| A100 | 256 | 20 | 4 | 1.09x -> 0.97x | 1.14x |
| A100 | 256 | 24 | 4 | 1.16x -> 1.07x | 1.15x |
| H100 | 32 | 20 | 25 | 1.36x -> 0.93x | 2.21x |
| H100 | 32 | 24 | 25 | 2.22x -> 1.05x | 2.68x |
| H100 | 256 | 20 | 4 | 1.06x -> 0.95x | 1.13x |
| H100 | 256 | 24 | 4 | 1.14x -> 1.05x | 1.14x |

## 4. CUDA occupancy ablation on A100 (eval-kernel time vs achieved blocks/SM)

| Workload | Width | r | Triton eval ms (blocks/SM) | CUDA eval ms own occupancy (blocks/SM) | CUDA eval ms at <= Triton's occupancy (blocks/SM) | Occupancy cost | Eval gap |
|---|---:|---:|---:|---:|---:|---:|---:|
| zk_complete_add_3 | 32 | 20 | 1.41 (10) | 0.72 (16) | 0.72 (10) | 1.01x | 1.97x |
| zk_complete_add_3 | 32 | 24 | 14.07 (10) | 4.25 (16) | 4.42 (10) | 1.04x | 3.31x |
| zk_complete_add_3 | 256 | 20 | 14.17 (3) | 24.99 (4) | 31.20 (2) | 1.25x | 0.57x |
| zk_complete_add_3 | 256 | 24 | 148.82 (3) | 246.46 (4) | 350.49 (2) | 1.42x | 0.60x |
| zk_spartan_2 | 32 | 20 | 0.48 (10) | 0.19 (16) | 0.19 (10) | 1.03x | 2.59x |
| zk_spartan_2 | 32 | 24 | 4.54 (10) | 0.73 (16) | 0.92 (10) | 1.27x | 6.25x |
| zk_spartan_2 | 256 | 20 | 1.02 (9) | 0.74 (10) | 0.79 (8) | 1.06x | 1.38x |
| zk_spartan_2 | 256 | 24 | 9.67 (9) | 5.56 (10) | 6.41 (8) | 1.15x | 1.74x |
| zk_vanilla_permcheck_hp | 32 | 20 | 1.30 (10) | 0.73 (16) | 0.73 (10) | 1.00x | 1.79x |
| zk_vanilla_permcheck_hp | 32 | 24 | 12.84 (10) | 5.82 (16) | 5.77 (10) | 0.99x | 2.21x |
| zk_vanilla_permcheck_hp | 256 | 20 | 16.33 (2) | 14.16 (4) | 16.33 (2) | 1.15x | 1.15x |
| zk_vanilla_permcheck_hp | 256 | 24 | 191.22 (2) | 162.61 (4) | 185.56 (2) | 1.14x | 1.18x |
| zk_witness_id_point_1 | 32 | 20 | 1.06 (10) | 0.48 (16) | 0.48 (10) | 1.01x | 2.22x |
| zk_witness_id_point_1 | 32 | 24 | 10.34 (10) | 2.29 (16) | 2.61 (10) | 1.14x | 4.52x |
| zk_witness_id_point_1 | 256 | 20 | 5.53 (5) | 5.20 (5) | 5.20 (5) | 1.00x | 1.06x |
| zk_witness_id_point_1 | 256 | 24 | 60.74 (5) | 51.38 (5) | 51.38 (5) | 1.00x | 1.18x |

## 5. r=28 cases with more than 2^31 table elements, fixed source, A100

| Workload | Width | Checksums agree | Wall lead | Kernel-only lead | Error |
|---|---:|---|---:|---:|---|
| zk_vanilla_zerocheck_hp | 32 | True | 2.21x | 2.16x | |
| zk_vanilla_zerocheck_hp | 64 | True | 1.50x | 1.46x | |
| zk_vanilla_permcheck_hp | 32 | True | 2.03x | 1.97x | |
| zk_vanilla_permcheck_hp | 64 | True | 1.46x | 1.43x | |
| zk_jellyfish_permcheck_hp | 32 | True | 1.80x | 1.75x | |
| zk_jellyfish_permcheck_hp | 64 | - | - | - | CalledProcessError: Command '['/home/ubuntu/venvs/zkduel-tab |
| zk_opencheck | 32 | True | 1.60x | 1.55x | |
| zk_opencheck | 64 | True | 1.19x | 1.16x | |
