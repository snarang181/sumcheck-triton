| Field | Paper (A100) | This run (H100) | Same? |
|---|---|---|---|
| GPU | NVIDIA A100-SXM4-80GB | NVIDIA A100 80GB PCIe | **no** |
| GPU memory | 81920 MiB | 81920 MiB | yes |
| SM count | 108 | 108 | yes |
| Compute capability | 8.0 | 8.0 | yes |
| NVIDIA driver | 580.126.20 | kernel module 565.57.01; user-mode 580.126.20 (CUDA API 13.0; /home/ubuntu/opt/cuda-compat-580.126.20/usr/local/cuda-13.0/compat/libcuda.so.580.126.20) | partial (user-mode only) |
| CUDA nvcc | 12.9 | 12.9.86 (/home/ubuntu/opt/cuda-12.9.1/bin/nvcc) | yes |
| PyTorch | 2.12.0+cu130 | 2.12.0+cu130 | yes |
| PyTorch CUDA runtime | 13.0 | 13.0 | yes |
| Triton | 3.7.0 | 3.7.0 | yes |
| CPU | Intel(R) Xeon(R) CPU @ 2.20GHz | Intel(R) Xeon(R) Gold 6338 CPU @ 2.00GHz | **no** |
| CPU logical cores | 12 | 12 | yes |
| System memory | 167 GB | 127 GB | **no** |
| OS | Ubuntu 24.04.4 LTS | Ubuntu 22.04.5 LTS | **no** |

| Additional | This run |
|---|---|
| Python | 3.12.14 |
| GPU max SM / memory clock | 1410 MHz / 1512 MHz |
| GPU power limit | 300.00 W |
| Persistence mode | Enabled |
| ECC | Enabled |
| Triton ptxas | /home/ubuntu/venvs/zkduel-table5/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas, /home/ubuntu/venvs/zkduel-table5/lib/python3.12/site-packages/triton/backends/nvidia/bin/ptxas-blackwell |
| TRITON_CACHE_DIR | (default) |
| LD_LIBRARY_PATH | /home/ubuntu/opt/cuda-compat-580.126.20/usr/local/cuda-13.0/compat |
