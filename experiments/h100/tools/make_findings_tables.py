"""Generate the markdown tables of experiments/h100/FINDINGS.md from the recorded data.

  python experiments/h100/tools/make_findings_tables.py > experiments/h100/FINDINGS_TABLES.md

Every number in FINDINGS.md comes from these tables; nothing is transcribed by hand.

--h100-dir DIR reads a new run laid out like experiments/h100 (DIR/diag/*.jsonl, DIR/diag/ncu/,
DIR/logs/mlir_passes_*.txt), e.g. the output of scripts/run_paper_experiments.sh; --sweep-csv
and --treduce-csv name its sweep CSVs. Without options the recorded data is used, as before.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
H = ROOT / "experiments" / "h100"
A100_CSV = ROOT / "experiments" / "a100" / "triton_cuda_field_resilient_r14_28_new.csv"
H100_CSV = H / "sweep" / "triton_cuda_field_resilient_r14_28_h100.csv"
TREDUCE_CSV = H / "sweep_treduce" / "triton_treduce_cuda_field_resilient_r14_28_h100.csv"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from summarize_sweep import table2, table4, validated  # noqa: E402


def load_csv(p: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(p.open(newline=""))) if p.exists() else []


def load_jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.open()] if p.exists() else []


def gmean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def lead(r: dict[str, str]) -> float:
    return float(r["triton_median_ms"]) / float(r["cuda_median_ms"])


def status(r: dict[str, str]) -> str:
    if validated(r):
        return "validated"
    return "checksum mismatch" if r.get("checksum_match") == "false" else "skipped"


def agree(r: dict) -> bool:
    """A diagnostic (JSONL) record whose Triton and CUDA checksums agree."""
    return r.get("checksum_match") is True


def keep_agreeing(records: list[dict], source: str) -> list[dict]:
    """Keep records whose checksums agree. Any others are named in the output and not used."""
    bad = [r for r in records if not agree(r)]
    if bad:
        names = ", ".join(
            f"{r.get('workload')} {r.get('bit_width')}-bit r={r.get('rounds')}" for r in bad
        )
        print(
            f"**Excluded {len(bad)} of {len(records)} records of `{source}` because the Triton "
            f"and CUDA checksums disagree:** {names}.\n"
        )
    return [r for r in records if agree(r)]


def section(title: str) -> None:
    print(f"\n## {title}\n")


def spearman(xs: list[float], ys: list[float]) -> float | None:
    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=v.__getitem__)
        out = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                out[order[k]] = (i + j) / 2
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.fmean(rx), statistics.fmean(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else None


def main(argv: list[str] | None = None) -> int:
    global H, H100_CSV, TREDUCE_CSV, A100_CSV
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--h100-dir",
        type=Path,
        default=None,
        help="run directory with diag/ and logs/ (default: experiments/h100)",
    )
    ap.add_argument(
        "--sweep-csv",
        type=Path,
        default=None,
        help="joined zkPHIRE sweep CSV (default: H100_DIR/sweep/triton_cuda_field_resilient_r14_28_h100.csv)",
    )
    ap.add_argument(
        "--treduce-csv",
        type=Path,
        default=None,
        help="full-matrix tl.reduce sweep CSV (default: H100_DIR/sweep_treduce/...; tables skipped if absent)",
    )
    ap.add_argument(
        "--a100-csv", type=Path, default=A100_CSV, help="A100 sweep CSV (default: the recorded one)"
    )
    args = ap.parse_args(argv)
    if args.h100_dir is not None:
        H = args.h100_dir.resolve()
        H100_CSV = H / "sweep" / H100_CSV.name
        TREDUCE_CSV = H / "sweep_treduce" / TREDUCE_CSV.name
    H100_CSV = args.sweep_csv or H100_CSV
    TREDUCE_CSV = args.treduce_csv or TREDUCE_CSV
    A100_CSV = args.a100_csv
    a100, h100, tred = load_csv(A100_CSV), load_csv(H100_CSV), load_csv(TREDUCE_CSV)
    akey = {(r["workload"], r["bit_width"], r["rounds"]): r for r in a100}

    section("T1. Paper Table 2, recomputed: A100 (paper run) vs H100 (this run)")
    print("A100:\n")
    print("\n".join(table2(a100)))
    print("\nH100:\n")
    print("\n".join(table2(h100)))
    if tred:
        print("\nH100, Triton with tl.reduce (jellyfish zerocheck 256-bit not run):\n")
        print("\n".join(table2(tred)))

    section("T2. Paper Table 4 (256-bit, r=24), A100 vs H100")
    print("A100:\n")
    print("\n".join(table4(a100)))
    print("\nH100:\n")
    print("\n".join(table4(h100)))
    if tred:
        print("\nH100, Triton with tl.reduce:\n")
        print("\n".join(table4(tred)))

    section("T3. Same configurations on both GPUs (validated on both)")
    by = defaultdict(list)
    for r in h100:
        a = akey[(r["workload"], r["bit_width"], r["rounds"])]
        if validated(r) and validated(a):
            by[int(r["bit_width"])].append((r, a))
    print(
        "| Width | Pairs | Median CUDA lead H100 | Median CUDA lead A100 | Lead ratio H100/A100 (geomean) "
        "| Triton H100 speedup | CUDA H100 speedup |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for bw in sorted(by):
        v = by[bw]
        print(
            f"| {bw} | {len(v)} | {statistics.median(lead(r) for r, a in v):.2f}x "
            f"| {statistics.median(lead(a) for r, a in v):.2f}x | {gmean([lead(r) / lead(a) for r, a in v]):.2f} "
            f"| {gmean([float(a['triton_median_ms']) / float(r['triton_median_ms']) for r, a in v]):.2f}x "
            f"| {gmean([float(a['cuda_median_ms']) / float(r['cuda_median_ms']) for r, a in v]):.2f}x |"
        )
    c = Counter(
        (status(r), status(akey[(r["workload"], r["bit_width"], r["rounds"])])) for r in h100
    )
    print(
        "\nStatus agreement (H100, A100): "
        + ", ".join(f"{k[0]} / {k[1]}: {n}" for k, n in sorted(c.items()))
    )

    section("T4. Wall clock vs GPU-kernel time (nsys), 25 workloads, H100")
    rows_all = [
        r for r in load_jsonl(H / "diag" / "baseline_all.jsonl") if "cuda_speedup_gpu_only" in r
    ]
    rows = keep_agreeing(rows_all, "diag/baseline_all.jsonl")
    grp = defaultdict(list)
    for r in rows:
        grp[(r["bit_width"], r["rounds"])].append(r)
    print(
        "| Width | r | n | Wall-clock CUDA lead (median) | Kernel-only CUDA lead (median, range) "
        "| GPU idle share of Triton wall time | Host us/launch Triton / CUDA |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for k in sorted(grp):
        v = grp[k]
        ko = [r["cuda_speedup_gpu_only"] for r in v]
        print(
            f"| {k[0]} | {k[1]} | {len(v)} | {statistics.median(r['cuda_speedup_wall'] for r in v):.2f}x "
            f"| {statistics.median(ko):.2f}x ({min(ko):.2f}-{max(ko):.2f}x) "
            f"| {statistics.median(r['triton_host_ms'] / r['triton_wall_ms'] for r in v):.0%} "
            f"| {statistics.median(1e3 * r['triton_host_ms'] / r['triton_launches_per_run'] for r in v):.1f} / "
            f"{statistics.median(1e3 * r['cuda_host_ms'] / r['cuda_launches_per_run'] for r in v):.1f} |"
        )
    print(
        f"\nAll {len(rows_all)} configurations: Triton and CUDA checksums agree: "
        f"{all(agree(r) for r in rows_all)}."
    )

    section("T5. tl.reduce ablation: kernel-only CUDA lead, original block reduction vs tl.reduce")
    base = {(r["workload"], r["bit_width"], r["rounds"]): r for r in rows}
    trows_all = [
        r for r in load_jsonl(H / "diag" / "treduce_all.jsonl") if "cuda_speedup_gpu_only" in r
    ]
    trows = keep_agreeing(trows_all, "diag/treduce_all.jsonl")
    tg = defaultdict(list)
    for t in trows:
        k = (t["workload"], t["bit_width"], t["rounds"])
        if k in base:
            tg[(t["bit_width"], t["rounds"])].append((base[k], t))
    print(
        "| Width | r | n | Kernel-only lead: tree -> tl.reduce | Eval-kernel speedup (geomean) "
        "| Triton faster than CUDA (kernel-only) | Eval regs Triton tree -> tl.reduce (CUDA), median of max |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|")
    for k in sorted(tg):
        v = tg[k]

        def mreg(r, side, name):
            return max(r[f"{side}_regs"][name])

        print(
            f"| {k[0]} | {k[1]} | {len(v)} | {statistics.median(b['cuda_speedup_gpu_only'] for b, t in v):.2f}x -> "
            f"{statistics.median(t['cuda_speedup_gpu_only'] for b, t in v):.2f}x "
            f"| {gmean([b['triton_eval_ms'] / t['triton_eval_ms'] for b, t in v]):.2f}x "
            f"| {sum(b['cuda_speedup_gpu_only'] < 1 for b, t in v)} -> {sum(t['cuda_speedup_gpu_only'] < 1 for b, t in v)} of {len(v)} "
            f"| {statistics.median(mreg(b, 'triton', '_round_eval_specialized_kernel') for b, t in v):.0f} -> "
            f"{statistics.median(mreg(t, 'triton', '_round_eval_specialized_kernel') for b, t in v):.0f} "
            f"({statistics.median(mreg(b, 'cuda', 'eval_kernel') for b, t in v):.0f}) |"
        )
    print(
        f"\nAll {len(trows_all)} tl.reduce configurations: checksums agree with CUDA: "
        f"{all(agree(t) for t in trows_all)}."
    )

    def blocks(regs: int) -> int:  # 128-thread CTAs, 8-register allocation granularity
        return min(16, 65536 // (((regs + 7) // 8) * 8 * 128))

    section("T13. Eval-kernel registers vs eval-kernel gap, all T4 configurations, H100")
    print(
        "Registers: maximum over a configuration's eval-kernel variants (points). Eval gap: Triton / CUDA "
        "eval-kernel time (nsys). Blocks/SM: register-limited, 128-thread blocks.\n"
    )
    print(
        "| Width | r | n | Triton regs | CUDA regs | Triton uses more | CUDA fits more blocks/SM "
        "| Eval gap (range) | Spearman: register difference vs gap | Spearman: occupancy ratio vs gap |"
    )
    print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def fmt_rho(x: float | None) -> str:
        return "- (constant)" if x is None else f"{x:+.2f}"

    for k in sorted(grp):
        v = grp[k]
        tr = [max(r["triton_regs"]["_round_eval_specialized_kernel"]) for r in v]
        cr = [max(r["cuda_regs"]["eval_kernel"]) for r in v]
        gap = [r["triton_eval_ms"] / r["cuda_eval_ms"] for r in v]
        occ_ratio = [blocks(c) / blocks(t) for t, c in zip(tr, cr)]
        pairs_bs = {(blocks(t), blocks(c)) for t, c in zip(tr, cr)}
        more = f"{sum(blocks(c) > blocks(t) for t, c in zip(tr, cr))} of {len(v)}"
        if len(pairs_bs) == 1:
            more += " ({} vs {} in all)".format(*pairs_bs.pop())
        print(
            f"| {k[0]} | {k[1]} | {len(v)} | {min(tr)}-{max(tr)} | {min(cr)}-{max(cr)} "
            f"| {sum(t > c for t, c in zip(tr, cr))} of {len(v)} "
            f"| {more} "
            f"| {min(gap):.2f}-{max(gap):.2f}x "
            f"| {fmt_rho(spearman([t - c for t, c in zip(tr, cr)], gap))} "
            f"| {fmt_rho(spearman(occ_ratio, gap))} |"
        )

    section("T6. CUDA occupancy ablation: CUDA eval-kernel time when forced to Triton's occupancy")
    occ = defaultdict(list)
    for r in load_jsonl(H / "diag" / "occupancy.jsonl"):
        if "error" not in r and r.get("cuda_blocks_per_sm"):
            occ[(r["workload"], r["bit_width"], r["rounds"])].append(r)

    print(
        "| Workload | Width | r | Triton eval ms (blocks/SM) | CUDA eval ms at own occupancy (blocks/SM) "
        "| CUDA eval ms at <= Triton's occupancy (blocks/SM) | Occupancy cost to CUDA | Actual eval gap Triton/CUDA |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k in sorted(occ):
        if k not in base:
            continue
        t = base[k]
        tb = blocks(max(t["triton_regs"]["_round_eval_specialized_kernel"]))
        own = next(r for r in occ[k] if r["label"] == "occ_smem0")
        cands = [r for r in occ[k] if r["cuda_blocks_per_sm"][1] <= tb] or occ[k]
        at = max(cands, key=lambda r: r["cuda_blocks_per_sm"][1])
        print(
            f"| {k[0]} | {k[1]} | {k[2]} | {t['triton_eval_ms']:.2f} ({tb}) | {own['cuda_eval_ms']:.2f} "
            f"({own['cuda_blocks_per_sm'][1]}) | {at['cuda_eval_ms']:.2f} ({at['cuda_blocks_per_sm'][1]}) "
            f"| {at['cuda_eval_ms'] / own['cuda_eval_ms']:.2f}x | {t['triton_eval_ms'] / own['cuda_eval_ms']:.2f}x |"
        )

    if tred:
        section(
            "T8. Full-matrix sweeps on H100: Triton with tl.reduce vs original, same configurations"
        )
        tk = {(r["workload"], r["bit_width"], r["rounds"]): r for r in tred}
        hk0 = {(r["workload"], r["bit_width"], r["rounds"]): r for r in h100}
        pairs = defaultdict(list)
        for k, t in tk.items():
            b = hk0.get(k)
            if b and validated(b) and validated(t):
                pairs[(int(k[1]), int(k[2]))].append((b, t))
        print(
            "Geomean ratio of wall-clock medians, tl.reduce run / original run, over configurations "
            "validated in both (Triton; CUDA is the same binary, shown as a run-to-run control).\n"
        )
        print(
            "| r | "
            + " | ".join(f"{bw}-bit Triton | {bw}-bit CUDA" for bw in (32, 64, 128, 256))
            + " |"
        )
        print("|---:|" + "---:|---:|" * 4)
        for rr in range(14, 29, 2):
            cells = []
            for bw in (32, 64, 128, 256):
                v = pairs.get((bw, rr))
                if not v:
                    cells += ["-", "-"]
                    continue
                cells.append(
                    f"{gmean([float(t['triton_median_ms']) / float(b['triton_median_ms']) for b, t in v]):.2f}"
                )
                cells.append(
                    f"{gmean([float(t['cuda_median_ms']) / float(b['cuda_median_ms']) for b, t in v]):.2f}"
                )
            print(f"| {rr} | " + " | ".join(cells) + " |")
        print("\nThe tl.reduce run also has the int64-offset fix in the encode and fold kernels.")

    small = load_jsonl(H / "diag" / "small_table_64_128.jsonl")
    if small:
        section(
            "T9. Small tables at 64/128 bits (r=16): host vs GPU time, original vs tl.reduce (nsys)"
        )
        small = keep_agreeing(small, "diag/small_table_64_128.jsonl")
        sk = {(r["workload"], r["bit_width"], r["label"].split("_")[1]): r for r in small}
        print(
            "| Workload | Width | Triton wall ms | Triton GPU ms | Triton eval-kernel ms | Triton host ms |"
        )
        print("|---|---:|---:|---:|---:|---:|")
        for (w, bw, v), b in sorted(sk.items()):
            if v != "base" or (w, bw, "treduce") not in sk:
                continue
            t = sk[(w, bw, "treduce")]
            print(
                f"| {w} | {bw} | {b['triton_wall_ms']:.2f} -> {t['triton_wall_ms']:.2f} "
                f"| {b['triton_gpu_ms']:.3f} -> {t['triton_gpu_ms']:.3f} "
                f"| {b['triton_eval_ms']:.3f} -> {t['triton_eval_ms']:.3f} "
                f"| {b['triton_host_ms']:.2f} -> {t['triton_host_ms']:.2f} |"
            )

    anat = load_jsonl(H / "diag" / "register_anatomy.jsonl")
    if anat:
        section(
            "T10. Register anatomy of matched eval kernels (point 2, specialized, element layout; static SASS)"
        )
        ak = {(r["workload"], r["bit_width"], r["variant"]): r for r in anat}
        cols = ["mul", "add", "compare/select", "shared mem", "shuffle", "total"]
        print("| Kernel | Variant | Registers | " + " | ".join(f"SASS {c}" for c in cols) + " |")
        print("|---|---|---:|" + "---:|" * len(cols))
        for w, bw in sorted(
            {(r["workload"], r["bit_width"]) for r in anat}, key=lambda x: (x[1], x[0])
        ):
            for v, label in (
                ("base", "Triton"),
                ("treduce", "Triton + tl.reduce"),
                ("cuda", "CUDA"),
            ):
                r = ak.get((w, bw, v))
                if r:
                    print(
                        f"| {w} {bw}-bit | {label} | {r['regs']} | "
                        + " | ".join(str(r["sass"][c]) for c in cols)
                        + " |"
                    )
        print(
            "\nStatic counts: CUDA's modular add/sub branch, so its static SASS includes both arms; "
            "use T11 for executed instructions."
        )

    ncu_dir = H / "diag" / "ncu"
    if ncu_dir.exists():
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ncu_summary import load as ncu_load

        section(
            "T11. Hardware counters (Nsight Compute), eval kernel of round 0, point 2, rounds=20"
        )

        def num(s: str) -> float:
            return float(s.split()[0].replace(",", ""))

        print(
            "| Kernel | Backend | Duration us | Registers | Executed instructions (M) | Theoretical occupancy "
            "| Achieved occupancy | Time vs CUDA | Instructions vs CUDA |"
        )
        print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
        for p in sorted(ncu_dir.glob("*_cuda.csv")):
            case = p.name[: -len("_cuda.csv")]
            c = ncu_load(p)
            for b, label in (
                ("triton", "Triton"),
                ("treduce", "Triton + tl.reduce"),
                ("cuda", "CUDA"),
            ):
                f = ncu_dir / f"{case}_{b}.csv"
                if not f.exists():
                    continue
                m = ncu_load(f)
                print(
                    f"| {case.replace('_bw', ' ')}-bit | {label} | {num(m['duration']) / 1e3:.1f} | {m['regs'].split()[0]} "
                    f"| {num(m['inst_executed']) / 1e6:.1f} | {m['occ_theoretical_%']} | {m['occ_achieved_%']} "
                    f"| {num(m['duration']) / num(c['duration']):.2f}x "
                    f"| {num(m['inst_executed']) / num(c['inst_executed']):.2f}x |"
                )

    prof = load_jsonl(H / "diag" / "compile_profile.jsonl")
    if prof:
        section(
            "T12. Triton compile cost of zk_jellyfish_zerocheck_hp's heaviest eval kernel (point 7)"
        )
        print(
            "| Triton | Width | Total s | TTIR s | TTGIR s | LLIR s | PTX s | cubin (ptxas) s | LLVM IR lines "
            "| SASS instructions | Registers | Peak RSS GB |"
        )
        print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in sorted(prof, key=lambda r: (r["triton"], r["bit_width"])):
            s = r.get("stages_s", {})
            print(
                f"| {r['triton']} | {r['bit_width']} | {r.get('total_s', '-')} | {s.get('ttir', '-')} | {s.get('ttgir', '-')} "
                f"| {s.get('llir', '-')} | {s.get('ptx', '-')} | {s.get('cubin', '-')} | {r.get('ir_lines', {}).get('llir', '-')} "
                f"| {r.get('sass_instructions', '-')} | {r.get('n_regs', '-')} | {r.get('peak_rss_gb', '-')} |"
            )
        import re

        print(
            "\nSix compiles ran concurrently on a 26-core host alongside other compile jobs, so absolute "
            "times are inflated; compare within a table."
        )
        passes = load_jsonl(H / "diag" / "compile_passes.jsonl")
        if passes:
            print(
                "\nPer-pass MLIR timing (`MLIR_ENABLE_TIMING=1`) and stage times from the same compile "
                "(`diag/compile_passes.jsonl`, `logs/mlir_passes_*.txt`):\n"
            )
            print(
                "| Triton | Width | Total s | TTGIR stage s | TritonGPUCoalesce s | Coalesce share of TTGIR stage "
                "| Next-largest pass |"
            )
            print("|---|---:|---:|---:|---:|---:|---|")
            for r in sorted(passes, key=lambda r: (r["triton"], r["bit_width"])):
                f = H / "logs" / f"mlir_passes_{r['triton']}_{r['bit_width']}.txt"
                reports = f.read_text().split("Execution time report") if f.exists() else []
                rep = next((x for x in reports if "TritonGPUCoalesce" in x), "")
                top = [
                    (float(m.group(1)), m.group(2))
                    for m in re.finditer(r"^\s+([0-9.]+) \(\s*[0-9.]+%\)  (\S.*)$", rep, re.M)
                    if m.group(2) not in ("Total", "Rest")
                ]
                co = next((t for t, n in top if n == "TritonGPUCoalesce"), None)
                other = max(((t, n) for t, n in top if n != "TritonGPUCoalesce"), default=None)
                ttgir = r.get("stages_s", {}).get("ttgir")
                if co is None or not ttgir:
                    continue
                print(
                    f"| {r['triton']} | {r['bit_width']} | {r.get('total_s', '-')} | {ttgir} | {co:.1f} "
                    f"| {co / ttgir:.1%} | {other[1]} ({other[0]:.2f} s) |"
                )
        else:
            print(
                "\nPer-pass MLIR timing (`MLIR_ENABLE_TIMING=1`), same kernel, separate compile, from "
                "`logs/mlir_timing_*.txt`:\n"
            )
            print("| Triton | Width | TritonGPUCoalesce s | Share of its pass pipeline |")
            print("|---|---:|---:|---:|")
            for f in sorted((H / "logs").glob("mlir_timing_*_*.txt")):
                ver, bw = f.stem.split("_")[2:4]
                m = re.search(
                    r"^\s+([0-9.]+) \(\s*([0-9.]+)%\)\s+TritonGPUCoalesce$", f.read_text(), re.M
                )
                if m:
                    print(f"| {ver} | {bw} | {float(m.group(1)):.1f} | {m.group(2)}% |")

    scal = [r for r in load_jsonl(H / "diag" / "coalesce_scaling.jsonl") if "coalesce_s" in r]
    if scal:
        section("T17. TritonGPUCoalesce cost on a synthetic kernel: M memory ops, N IR ops")
        print(
            "Kernel: M loads summed into one accumulator, then a chain of C multiply-adds, one store "
            "(`tools/coalesce_scaling.py`). Compile only, fresh cache per point, `MLIR_ENABLE_TIMING=1`. "
            "If the pass is O(M N^2), the last column stays roughly constant.\n"
        )
        print(
            "| Triton | Varied | Memory ops M | TTIR ops N | Coalesce s | Share of TTGIR pipeline "
            "| Coalesce s / (M N^2) x 1e9 |"
        )
        print("|---|---|---:|---:|---:|---:|---:|")
        for r in sorted(scal, key=lambda r: (r["triton"], r["sweep"], r["M"], r["C"])):
            print(
                f"| {r['triton']} | {'M' if r['sweep'] == 'M' else 'N'} | {r['mem_ops']} | {r['ttir_ops']} "
                f"| {r['coalesce_s']:.1f} | {r['coalesce_share']}% "
                f"| {1e9 * r['coalesce_s'] / (r['mem_ops'] * r['ttir_ops'] ** 2):.1f} |"
            )
        print("\nLog-log slope of Coalesce time (least squares):\n")
        print("| Triton | vs M (N about fixed) | vs N (M fixed) |")
        print("|---|---:|---:|")

        def slope(pts: list[tuple[float, float]]) -> str:
            if len(pts) < 2:
                return "-"
            xs, ys = [math.log(x) for x, _ in pts], [math.log(y) for _, y in pts]
            mx, my = statistics.fmean(xs), statistics.fmean(ys)
            return f"{sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs):.2f}"

        for ver in sorted({r["triton"] for r in scal}):
            v = [r for r in scal if r["triton"] == ver]
            print(
                f"| {ver} | {slope([(r['mem_ops'], r['coalesce_s']) for r in v if r['sweep'] == 'M'])} "
                f"| {slope([(r['ttir_ops'], r['coalesce_s']) for r in v if r['sweep'] == 'C'])} |"
            )

    def eval_regs(r: dict, side: str) -> int:
        return max(max(v) for k, v in r[f"{side}_regs"].items() if "eval" in k)

    abl = [
        r for r in load_jsonl(H / "diag" / "layout_points.jsonl") if "cuda_eval_ms" in r
    ]
    if abl:
        section(
            "T15. The paper's two optimizations: layout x interpolation ablation, both backends, H100"
        )
        abl = keep_agreeing(abl, "diag/layout_points.jsonl")
        print(
            "Each variant relative to the paper's configuration (element-major, specialized points): "
            "geomean over the 5 workloads of variant time / default time. Registers: median over "
            "workloads of the eval kernel's maximum.\n"
        )
        ak = {(r["label"], r["workload"], r["bit_width"], r["rounds"]): r for r in abl}
        default = "abl_element_specialized"
        print(
            "| Width | r | Layout | Points | n | Triton wall | Triton eval kernel | CUDA wall | CUDA eval kernel "
            "| Triton eval regs | CUDA eval regs | Checksums agree (all backends, all variants) |"
        )
        print("|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
        for bw, rr in sorted({(r["bit_width"], r["rounds"]) for r in abl}):
            for layout in ("element", "limb"):
                for pm in ("specialized", "generic"):
                    lab = f"abl_{layout}_{pm}"
                    v = [
                        (ak[(default, w, b, x)], r)
                        for (lb, w, b, x), r in ak.items()
                        if lb == lab and (b, x) == (bw, rr) and (default, w, b, x) in ak
                    ]
                    if not v:
                        continue

                    def rel(key: str) -> str:
                        return f"{gmean([r[key] / d[key] for d, r in v]):.2f}x"

                    same = all(
                        r["triton_checksum"] == r["cuda_checksum"] == d["cuda_checksum"] for d, r in v
                    )
                    print(
                        f"| {bw} | {rr} | {layout} | {pm} | {len(v)} | {rel('triton_wall_ms')} "
                        f"| {rel('triton_eval_ms')} | {rel('cuda_wall_ms')} | {rel('cuda_eval_ms')} "
                        f"| {statistics.median(eval_regs(r, 'triton') for d, r in v):.0f} "
                        f"| {statistics.median(eval_regs(r, 'cuda') for d, r in v):.0f} | {same} |"
                    )

    if abl:
        print(
            "\nPer workload, element layout, generic -> specialized points: eval-kernel registers "
            "(min-max over interpolation points) and eval-kernel time per SumCheck run (nsys).\n"
        )
        print(
            "| Width | r | Workload | Triton regs | CUDA regs | Triton eval ms | CUDA eval ms "
            "| Triton speedup | CUDA speedup |"
        )
        print("|---:|---:|---|---:|---:|---:|---:|---:|---:|")

        def rng(r: dict, side: str) -> str:
            lo, hi = next(v for k, v in r[f"{side}_regs"].items() if "eval" in k)
            return f"{lo}" if lo == hi else f"{lo}-{hi}"

        for (lb, w, b, x), sp in sorted(ak.items(), key=lambda kv: (kv[0][2], kv[0][3], kv[0][1])):
            gen = ak.get(("abl_element_generic", w, b, x))
            if lb != default or not gen or x != 20:
                continue
            print(
                f"| {b} | {x} | {w} | {rng(gen, 'triton')} -> {rng(sp, 'triton')} "
                f"| {rng(gen, 'cuda')} -> {rng(sp, 'cuda')} "
                f"| {gen['triton_eval_ms']:.2f} -> {sp['triton_eval_ms']:.2f} "
                f"| {gen['cuda_eval_ms']:.2f} -> {sp['cuda_eval_ms']:.2f} "
                f"| {gen['triton_eval_ms'] / sp['triton_eval_ms']:.2f}x "
                f"| {gen['cuda_eval_ms'] / sp['cuda_eval_ms']:.2f}x |"
            )
        print(
            "\nAt 32 bits the two layouts are the same memory arrangement (one limb), so the 32-bit "
            "limb rows are a noise control for the wall-clock columns."
        )

    if abl:
        section("T18. Run-to-run reproducibility of the medians (same code, same configuration)")
        print(
            "The 20 configurations measured in both the T4 run (`baseline_all`) and the T15 run "
            "(`abl_element_specialized`), hours apart; ratio of the two medians.\n"
        )
        print("| Metric | n | Within 2% | Within 5% | Largest deviation |")
        print("|---|---:|---:|---:|---:|")
        pairs_r = [
            (base[(r["workload"], r["bit_width"], r["rounds"])], r)
            for r in abl
            if r["label"] == default and (r["workload"], r["bit_width"], r["rounds"]) in base
        ]
        for key, name in (
            ("triton_wall_ms", "Triton wall-clock"),
            ("cuda_wall_ms", "CUDA wall-clock"),
            ("triton_eval_ms", "Triton eval kernels (nsys)"),
            ("cuda_eval_ms", "CUDA eval kernels (nsys)"),
        ):
            dev = [abs(r[key] / b[key] - 1) for b, r in pairs_r]
            print(
                f"| {name} | {len(dev)} | {sum(d <= 0.02 for d in dev)} | {sum(d <= 0.05 for d in dev)} "
                f"| {max(dev):.0%} |"
            )
        pin = load_jsonl(H / "diag" / "pin_noise.jsonl")
        if pin:
            print(
                "\nOne configuration (zk_spartan_2, 32-bit, r=24) pinned to each of the 26 vCPUs "
                "(`taskset`), then 10 unpinned runs (`tools/pin_noise.sh`), medians in ms:\n"
            )
            print("| Backend | Pinned: min / median / max | Unpinned: min / median / max |")
            print("|---|---:|---:|")
            for b in ("triton", "cuda"):
                for lab in ("pinned", "unpinned"):
                    v = [r["median_ms"] for r in pin if r["backend"] == b and r["label"] == lab]
                    if lab == "pinned":
                        cell = f"{min(v):.2f} / {statistics.median(v):.2f} / {max(v):.2f}"
                    else:
                        print(
                            f"| {b} | {cell} | {min(v):.2f} / {statistics.median(v):.2f} / {max(v):.2f} |"
                        )

    blk = [r for r in load_jsonl(H / "diag" / "cuda_block.jsonl") if "cuda_eval_ms" in r]
    if blk:
        section("T16. CUDA thread-block size (the only CUDA tuning parameter), H100")
        print(
            "Geomean over the 5 workloads of time at each block size / time at 128 (the setting used "
            "in the paper). Best: workloads whose fastest block size is within 2% of 128's time.\n"
        )
        bk = {(int(r["label"].split("_")[1]), r["workload"], r["bit_width"], r["rounds"]): r for r in blk}
        sizes = sorted({k[0] for k in bk})
        print(
            "| Width | r | n | "
            + " | ".join(f"wall {b} | eval {b}" for b in sizes)
            + " | 128 within 2% of best (wall) | Checksums identical across sizes |"
        )
        print("|---:|---:|---:|" + "---:|---:|" * len(sizes) + "---:|---|")
        for bw, rr in sorted({(k[2], k[3]) for k in bk}):
            ws = sorted({k[1] for k in bk if (k[2], k[3]) == (bw, rr) and (128, k[1], bw, rr) in bk})
            cells = []
            for b in sizes:
                pairs = [(bk[(128, w, bw, rr)], bk[(b, w, bw, rr)]) for w in ws if (b, w, bw, rr) in bk]
                for key in ("cuda_wall_ms", "cuda_eval_ms"):
                    cells.append(
                        f"{gmean([r[key] / d[key] for d, r in pairs]):.2f}" if pairs else "-"
                    )
            near = sum(
                bk[(128, w, bw, rr)]["cuda_wall_ms"]
                <= 1.02 * min(bk[(b, w, bw, rr)]["cuda_wall_ms"] for b in sizes if (b, w, bw, rr) in bk)
                for w in ws
            )
            same = all(
                len({bk[(b, w, bw, rr)]["cuda_checksum"] for b in sizes if (b, w, bw, rr) in bk}) == 1
                for w in ws
            )
            print(f"| {bw} | {rr} | {len(ws)} | " + " | ".join(cells) + f" | {near} of {len(ws)} | {same} |")

    section("T7. r=28 configurations with more than 2^31 table elements, with the int64-offset fix")
    hkey = {(r["workload"], r["bit_width"], r["rounds"]): r for r in h100}

    def hstatus(k: tuple[str, str, str]) -> str:
        return status(hkey[k]) if k in hkey else "not in this sweep"

    print(
        "| Workload | Width | A100 sweep | H100 sweep | With fix: checksums agree | Wall lead | Kernel-only lead |"
    )
    print("|---|---:|---|---|---|---:|---:|")
    for r in load_jsonl(H / "diag" / "r28_int64_fix.jsonl"):
        k = (r["workload"], str(r["bit_width"]), "28")
        if "error" in r:
            fixed = "does not fit in 80 GB"
            print(
                f"| {r['workload']} | {r['bit_width']} | {status(akey[k])} | {hstatus(k)} | {fixed} | - | - |"
            )
        else:
            leads = (
                f"| {r['cuda_speedup_wall']:.2f}x | {r['cuda_speedup_gpu_only']:.2f}x |"
                if agree(r)
                else "| - | - |"
            )
            print(
                f"| {r['workload']} | {r['bit_width']} | {status(akey[k])} | {hstatus(k)} | {r['checksum_match']} "
                + leads
            )

    section("T14. The paper's 54 A100 skips (42 capacity / 12 non-capacity) by measured cause")
    from classify_skips import classify

    print(
        "Paper's class: footprint 3 x vars x 2^r x limbs x 4 bytes above 9 GB is 'capacity' (Section 5). "
        "Cause: the H100 run's per-case log for the same configuration.\n"
    )
    causes = Counter()
    examples = defaultdict(list)
    for a in a100:
        k = (a["workload"], a["bit_width"], a["rounds"])
        if status(a) != "skipped":
            continue
        h = hkey.get(k)
        gb = 3 * int(a["vars"]) * (1 << int(a["rounds"])) * int(a["bit_width"]) // 32 * 4 / 1e9
        paper = "capacity" if gb > 9 else "non-capacity"
        if h is None:
            cause = "not run in this H100 sweep"
        elif validated(h):
            cause = (
                "validates on H100; on A100 the first round count(s) run for its workload and width"
            )
        else:
            label = classify(h, ROOT)[0]
            cause = {
                "out-of-memory": "out of memory (footprint > 80 GB)",
                "launch-error": "int32 index overflow in encode kernel (illegal memory access)",
                "timeout": "Triton compile > 30 min (jellyfish zerocheck, 256-bit)",
            }.get(label, label)
        causes[(paper, cause)] += 1
        examples[(paper, cause)].append((f"{a['workload']} {a['bit_width']}-bit r={a['rounds']}", gb))
    print("| Paper's class | Measured cause | Cases | Configurations |")
    print("|---|---|---:|---|")
    for (paper, cause), n in sorted(causes.items()):
        ex = examples[(paper, cause)]
        gbs = [g for _, g in ex]
        shown = (
            "; ".join(f"{name} ({g:.2g} GB)" for name, g in ex)
            if n <= 8
            else f"footprints {min(gbs):.1f}-{max(gbs):.1f} GB"
        )
        print(f"| {paper} | {cause} | {n} | {shown} |")

    # The paper's Table 3: the T4 split rerun with the fixed-harness CUDA binary. Its 32-bit rows
    # come from a rerun on a quiet host (baseline_fixed_32), which supersedes baseline_fixed_all.
    fixed = {}
    for name in ("baseline_fixed_all.jsonl", "baseline_fixed_32.jsonl"):
        for r in load_jsonl(H / "sweep_fixed" / name):
            if "cuda_speedup_gpu_only" in r and (name != "baseline_fixed_32.jsonl" or r["bit_width"] == 32):
                fixed[(r["workload"], r["bit_width"], r["rounds"])] = r
    if fixed:
        section("T19. Paper Table 3: wall clock vs GPU-kernel time on the fixed harness, H100")
        print(
            "`sweep_fixed/baseline_fixed_all.jsonl`, 32-bit rows from `sweep_fixed/baseline_fixed_32.jsonl`; "
            "tl.reduce kernel-only ratio from `diag/treduce_all.jsonl` (T5). Ratios are Triton/CUDA.\n"
        )
        frows = keep_agreeing(list(fixed.values()), "sweep_fixed/baseline_fixed_{all,32}.jsonl")
        tred_k = {(t["workload"], t["bit_width"], t["rounds"]): t for t in trows}
        fg = defaultdict(list)
        for r in frows:
            fg[(r["bit_width"], r["rounds"])].append(r)
        print(
            "| Width | r | n | Wall T/C | Kernel T/C (range) | +tl.reduce kernel T/C | No-kernel share "
            "of Triton wall | Host us/launch Triton / CUDA |"
        )
        print("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for k in sorted(fg):
            v = fg[k]
            ko = [r["cuda_speedup_gpu_only"] for r in v]
            tk = [
                tred_k[w]["cuda_speedup_gpu_only"]
                for w in ((r["workload"], r["bit_width"], r["rounds"]) for r in v)
                if w in tred_k
            ]
            print(
                f"| {k[0]} | {k[1]} | {len(v)} | {statistics.median(r['cuda_speedup_wall'] for r in v):.2f} "
                f"| {statistics.median(ko):.2f} ({min(ko):.2f}-{max(ko):.2f}) "
                f"| {statistics.median(tk):.2f} (n={len(tk)}) "
                f"| {statistics.median(r['triton_host_ms'] / r['triton_wall_ms'] for r in v):.0%} "
                f"| {statistics.median(1e3 * r['triton_host_ms'] / r['triton_launches_per_run'] for r in v):.1f} / "
                f"{statistics.median(1e3 * r['cuda_host_ms'] / r['cuda_launches_per_run'] for r in v):.1f} |"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
