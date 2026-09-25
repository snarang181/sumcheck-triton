## H100, Triton with tl.reduce

rows=792 validated=758 checksum_mismatch=0 skipped=34

### Table 2

| Width | Validated | Median | Best ratio | Wins | Within 1.1x |
|---:|---:|---:|---:|---:|---:|
| 32 | 200 | 3.07x | 1.096 | 0 | 1 |
| 64 | 198 | 2.83x | 0.833 | 5 | 11 |
| 128 | 189 | 1.96x | 0.705 | 9 | 13 |
| 256 | 171 | 1.24x | 0.636 | 29 | 50 |
| All | 758 | 1.98x | 0.636 | 43 | 75 |

### Table 4

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

