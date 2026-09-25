"""Summarize ncu_anatomy.sh outputs: one row per (case, backend) with key counters."""

import csv
import sys
from pathlib import Path

D = Path(__file__).resolve().parents[1] / "diag" / "ncu"
WANT = {
    "Duration": "duration",
    "Registers Per Thread": "regs",
    "Executed Instructions": "inst_executed",
    "Theoretical Occupancy": "occ_theoretical_%",
    "Achieved Occupancy": "occ_achieved_%",
    "Executed Ipc Active": "ipc_active",
    "Issue Slots Busy": "issue_busy_%",
    "Warp Cycles Per Issued Instruction": "cycles_per_issue",
    "Compute (SM) Throughput": "sm_throughput_%",
    "Memory Throughput": "mem_throughput_%",
    "Block Limit Registers": "block_limit_regs",
    "Shared Memory Configuration Size": "smem_config",
}


def load(p: Path) -> dict:
    rows = [r for r in csv.reader(p.open()) if len(r) > 10]
    if not rows:
        return {}
    hdr = rows[0]
    i_name, i_unit, i_val = (
        hdr.index("Metric Name"),
        hdr.index("Metric Unit"),
        hdr.index("Metric Value"),
    )
    out = {}
    for r in rows[1:]:
        if r[i_name] in WANT and WANT[r[i_name]] not in out:
            out[WANT[r[i_name]]] = f"{r[i_val]} {r[i_unit]}".strip()
    return out


def main() -> int:
    keys = [
        "duration",
        "regs",
        "inst_executed",
        "occ_theoretical_%",
        "occ_achieved_%",
        "ipc_active",
        "cycles_per_issue",
        "issue_busy_%",
        "sm_throughput_%",
        "mem_throughput_%",
    ]
    print("| Case | Backend | " + " | ".join(keys) + " |")
    print("|---|---|" + "---:|" * len(keys))
    for p in sorted(D.glob("*_cuda.csv")):
        case = p.name[: -len("_cuda.csv")]
        for b in ("triton", "treduce", "cuda"):
            f = D / f"{case}_{b}.csv"
            if f.exists():
                m = load(f)
                print(f"| {case} | {b} | " + " | ".join(m.get(k, "-") for k in keys) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
