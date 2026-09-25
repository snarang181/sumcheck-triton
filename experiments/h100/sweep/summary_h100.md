## H100

rows=800 validated=751 checksum_mismatch=2 skipped=47

### Table 2

| Width | Validated | Median | Best ratio | Wins | Within 1.1x |
|---:|---:|---:|---:|---:|---:|
| 32 | 195 | 3.35x | 1.618 | 0 | 0 |
| 64 | 196 | 2.53x | 1.084 | 0 | 1 |
| 128 | 189 | 1.88x | 0.856 | 7 | 12 |
| 256 | 171 | 1.33x | 0.643 | 23 | 33 |
| All | 751 | 2.24x | 0.643 | 30 | 46 |

### Table 4

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

