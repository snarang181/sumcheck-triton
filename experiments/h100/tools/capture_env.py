"""Record the host environment in the layout of the paper's Table 5.

Writes experiments/h100/env/table5_h100.{md,json} plus raw nvidia-smi / lscpu dumps.
Run with experiments/h100/env_table5.sh sourced.
"""

from __future__ import annotations

import ctypes
import glob
import json
import os
import platform
import re
import subprocess
import sys
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "env"

TABLE5_A100 = {
    "GPU": "NVIDIA A100-SXM4-80GB",
    "GPU memory": "81920 MiB",
    "SM count": "108",
    "Compute capability": "8.0",
    "NVIDIA driver": "580.126.20",
    "CUDA nvcc": "12.9",
    "PyTorch": "2.12.0+cu130",
    "PyTorch CUDA runtime": "13.0",
    "Triton": "3.7.0",
    "CPU": "Intel(R) Xeon(R) CPU @ 2.20GHz",
    "CPU logical cores": "12",
    "System memory": "167 GB",
    "OS": "Ubuntu 24.04.4 LTS",
}


def sh(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout.strip()
    except FileNotFoundError:
        return ""


def user_mode_driver() -> str:
    """Version of the libcuda.so.1 this process actually loads (forward-compat aware)."""
    lib = ctypes.CDLL("libcuda.so.1")
    ver = ctypes.c_int()
    lib.cuDriverGetVersion(ctypes.byref(ver))
    path = ""
    for line in Path("/proc/self/maps").read_text().splitlines():
        if "libcuda.so" in line:
            path = os.path.realpath(line.split()[-1])
            break
    m = re.search(r"libcuda\.so\.([\d.]+)$", path)
    return f"{m.group(1) if m else '?'} (CUDA API {ver.value // 1000}.{ver.value % 1000 // 10}; {path})"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT)
    ap.add_argument("--label", default="h100", help="file name suffix: table5_LABEL.{md,json}")
    args = ap.parse_args()
    out = args.out_dir
    import torch
    import triton

    out.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    smi = sh(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,clocks.max.sm,"
            "clocks.max.mem,power.limit,persistence_mode,ecc.mode.current",
            "--format=csv,noheader",
        ]
    )
    name, mem, kdriver, max_sm, max_mem, power, persist, ecc = [x.strip() for x in smi.split(",")]
    nvcc = re.search(r"release ([\d.]+), V([\d.]+)", sh(["nvcc", "--version"]))
    cpu = re.search(r"Model name:\s*(.+)", sh(["lscpu"]))
    os_name = re.search(r'PRETTY_NAME="(.+)"', Path("/etc/os-release").read_text())
    mem_kb = int(re.search(r"MemTotal:\s+(\d+)", Path("/proc/meminfo").read_text()).group(1))

    host = {
        "GPU": name,
        "GPU memory": mem,
        "SM count": str(props.multi_processor_count),
        "Compute capability": f"{props.major}.{props.minor}",
        "NVIDIA driver": f"kernel module {kdriver}; user-mode {user_mode_driver()}",
        "CUDA nvcc": f"{nvcc.group(2) if nvcc else '?'} ({sh(['which', 'nvcc'])})",
        "PyTorch": torch.__version__,
        "PyTorch CUDA runtime": str(torch.version.cuda),
        "Triton": triton.__version__,
        "CPU": cpu.group(1).strip() if cpu else platform.processor(),
        "CPU logical cores": str(os.cpu_count()),
        "System memory": f"{mem_kb * 1024 / 1e9:.0f} GB",
        "OS": os_name.group(1) if os_name else platform.platform(),
    }
    extra = {
        "Python": sys.version.split()[0],
        "GPU max SM / memory clock": f"{max_sm} / {max_mem}",
        "GPU power limit": power,
        "Persistence mode": persist,
        "ECC": ecc,
        "Triton ptxas": ", ".join(
            sorted(
                glob.glob(
                    os.path.join(os.path.dirname(triton.__file__), "backends/nvidia/bin/ptxas*")
                )
            )
        ),
        "TRITON_CACHE_DIR": os.environ.get("TRITON_CACHE_DIR", "(default)"),
        "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
    }

    rows = [f"| Field | Paper (A100) | This run ({args.label.upper()}) | Same? |", "|---|---|---|---|"]
    for key, paper in TABLE5_A100.items():
        mine = host[key]
        if key == "NVIDIA driver":
            user_ok = f"user-mode {paper} " in mine
            kernel_ok = f"kernel module {paper};" in mine
            same = (
                "yes"
                if user_ok and kernel_ok
                else ("partial (user-mode only)" if user_ok else "**no**")
            )
        elif key == "CUDA nvcc":
            same = (
                "yes" if mine.startswith(paper + ".") or mine.startswith(paper + " ") else "**no**"
            )
        else:
            same = "yes" if mine == paper else "**no**"
        rows.append(f"| {key} | {paper} | {mine} | {same} |")
    rows += ["", "| Additional | This run |", "|---|---|"]
    rows += [f"| {k} | {v} |" for k, v in extra.items()]
    (out / f"table5_{args.label}.md").write_text("\n".join(rows) + "\n")
    (out / f"table5_{args.label}.json").write_text(
        json.dumps({"host": host, "extra": extra, "paper_table5": TABLE5_A100}, indent=2) + "\n"
    )
    (out / "nvidia-smi-q.txt").write_text(sh(["nvidia-smi", "-q"]) + "\n")
    (out / "lscpu.txt").write_text(sh(["lscpu"]) + "\n")
    (out / "pip-freeze.txt").write_text(
        sh([sys.executable, "-m", "pip", "freeze"])
        or sh(["uv", "pip", "freeze", "--python", sys.executable])
    )
    print("\n".join(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
