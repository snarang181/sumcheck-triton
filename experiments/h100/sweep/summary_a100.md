## A100

rows=800 validated=744 checksum_mismatch=2 skipped=54

### Table 2

| Width | Validated | Median | Best ratio | Wins | Within 1.1x |
|---:|---:|---:|---:|---:|---:|
| 32 | 195 | 4.39x | 1.897 | 0 | 0 |
| 64 | 196 | 3.58x | 1.238 | 0 | 0 |
| 128 | 188 | 2.58x | 0.891 | 5 | 7 |
| 256 | 165 | 1.60x | 0.640 | 17 | 26 |
| All | 744 | 3.12x | 0.640 | 22 | 33 |

### Table 4

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

