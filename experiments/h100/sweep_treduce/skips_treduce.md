### Non-validated cases by class

| Class | Width | Footprint > device | Cases |
|---|---:|---|---:|
| out-of-memory | 64 | yes | 2 |
| out-of-memory | 128 | yes | 11 |
| out-of-memory | 256 | yes | 21 |

### Per case

| Workload | Width | r | Footprint GB | > device? | Class | Evidence |
|---|---:|---:|---:|---|---|---|
| zk_verifiable_asics | 256 | 28 | 103.1 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 14.66 GiB is free. Including non-PyTor` |
| zk_spartan_1 | 256 | 28 | 103.1 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 14.66 GiB is free. Including non-PyTor` |
| zk_incomplete_add_1 | 256 | 28 | 154.6 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 30.66 GiB is free. Including non-PyTor` |
| zk_incomplete_add_2 | 128 | 28 | 90.2 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 22.66 GiB is free. Including non-PyTor` |
| zk_incomplete_add_2 | 256 | 28 | 180.4 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 56.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 22.66 GiB is free. Including non-PyTor` |
| zk_complete_add_1 | 256 | 28 | 154.6 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 30.66 GiB is free. Including non-PyTor` |
| zk_complete_add_2 | 256 | 28 | 154.6 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 30.66 GiB is free. Including non-PyTor` |
| zk_complete_add_3 | 256 | 28 | 128.8 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 38.66 GiB is free. Including non-PyTor` |
| zk_complete_add_4 | 128 | 28 | 90.2 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 22.66 GiB is free. Including non-PyTor` |
| zk_complete_add_4 | 256 | 28 | 180.4 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 56.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 22.66 GiB is free. Including non-PyTor` |
| zk_complete_add_5 | 128 | 28 | 90.2 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 28.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 22.66 GiB is free. Including non-PyTor` |
| zk_complete_add_5 | 256 | 28 | 180.4 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 56.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 22.66 GiB is free. Including non-PyTor` |
| zk_complete_add_6 | 128 | 28 | 103.1 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 14.66 GiB is free. Including non-PyTor` |
| zk_complete_add_6 | 256 | 28 | 206.2 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 14.66 GiB is free. Including non-PyTor` |
| zk_complete_add_7 | 256 | 28 | 128.8 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 38.66 GiB is free. Including non-PyTor` |
| zk_complete_add_8 | 256 | 28 | 128.8 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 38.66 GiB is free. Including non-PyTor` |
| zk_complete_add_9 | 256 | 28 | 128.8 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 38.66 GiB is free. Including non-PyTor` |
| zk_complete_add_10 | 256 | 28 | 128.8 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 40.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 38.66 GiB is free. Including non-PyTor` |
| zk_complete_add_11 | 128 | 28 | 103.1 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 14.66 GiB is free. Including non-PyTor` |
| zk_complete_add_11 | 256 | 28 | 206.2 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 14.66 GiB is free. Including non-PyTor` |
| zk_complete_add_12 | 128 | 28 | 103.1 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 32.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 14.66 GiB is free. Including non-PyTor` |
| zk_complete_add_12 | 256 | 28 | 206.2 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 64.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 14.66 GiB is free. Including non-PyTor` |
| zk_vanilla_zerocheck_hp | 128 | 28 | 116.0 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 36.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 6.66 GiB is free. Including non-PyTorc` |
| zk_vanilla_zerocheck_hp | 256 | 28 | 231.9 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1024.00 MiB. GPU 0 has a total capacity of 79.19 GiB of which 679.00 MiB is free. Including non-Py` |
| zk_vanilla_permcheck_hp | 128 | 28 | 141.7 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 44.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 34.66 GiB is free. Including non-PyTor` |
| zk_vanilla_permcheck_hp | 256 | 28 | 283.5 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 88.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 78.74 GiB is free. Including non-PyTor` |
| zk_jellyfish_permcheck_hp | 64 | 28 | 96.6 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 18.66 GiB is free. Including non-PyTor` |
| zk_jellyfish_permcheck_hp | 128 | 28 | 193.3 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 60.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 18.66 GiB is free. Including non-PyTor` |
| zk_jellyfish_permcheck_hp | 256 | 26 | 96.6 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 30.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 18.66 GiB is free. Including non-PyTor` |
| zk_jellyfish_permcheck_hp | 256 | 28 | 386.5 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 120.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 78.74 GiB is free. Including non-PyTo` |
| zk_opencheck | 128 | 28 | 154.6 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 48.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 30.66 GiB is free. Including non-PyTor` |
| zk_opencheck | 256 | 28 | 309.2 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 96.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 78.74 GiB is free. Including non-PyTor` |
| zk_jellyfish_zerocheck_hp | 64 | 28 | 141.7 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 44.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 34.66 GiB is free. Including non-PyTor` |
| zk_jellyfish_zerocheck_hp | 128 | 28 | 283.5 | yes | out-of-memory | `torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 88.00 GiB. GPU 0 has a total capacity of 79.19 GiB of which 78.74 GiB is free. Including non-PyTor` |

