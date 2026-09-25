#!/usr/bin/env python3
"""Generate zkDuel paper plots and summary tables from SumCheck CSVs.

The main inputs are joined Triton/CUDA CSVs with columns such as:

    workload, polynomial, bit_width, rounds, N,
    triton_eval_median_ms, triton_challenge_median_ms, triton_fold_median_ms,
    triton_total_median_ms, cuda_eval_median_ms, cuda_challenge_median_ms,
    cuda_fold_median_ms, cuda_total_median_ms, cuda_speedup_vs_triton,
    checksum_match

Optional large device-resident scaling CSVs may be supplied with --scaling-csv.
Those are expected to have at least workload, backend, rounds, N, median_ms, and
status columns.
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import matplotlib
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib is required for plotting. Install it with `pip install matplotlib` "
        "or `pip install -e '.[plot]'`."
    ) from exc

matplotlib.use("Agg")
import matplotlib.pyplot as plt

WORKLOAD_ORDER = [
    "poly_a",
    "poly_ab",
    "poly_ab_plus_c",
    "poly_abc",
    "poly_aabbc",
    "poly_abc_plus_de",
    "poly_abcg_plus_deg",
    "zk_verifiable_asics",
    "zk_spartan_1",
    "zk_spartan_2",
    "zk_witness_non_id",
    "zk_complete_add_1",
    "zk_complete_add_7",
    "zk_complete_add_8",
]

WORKLOAD_LABELS = {
    "poly_a": "a",
    "poly_ab": "a*b",
    "poly_ab_plus_c": "a*b+c",
    "poly_abc": "a*b*c",
    "poly_aabbc": "a*a*b*b*c",
    "poly_abc_plus_de": "a*b*c+d*e",
    "poly_abcg_plus_deg": "a*b*c*g+d*e*g",
    "zk_verifiable_asics": "ASIC",
    "zk_spartan_1": "Spartan-1",
    "zk_spartan_2": "Spartan-2",
    "zk_witness_non_id": "EC witness",
    "zk_complete_add_1": "EC add-1",
    "zk_complete_add_7": "EC add-7",
    "zk_complete_add_8": "EC add-8",
}

BACKEND_COLORS = {
    "triton": "#377eb8",
    "cuda": "#e41a1c",
}
PHASE_COLORS = {
    "eval": "#4daf4a",
    "challenge": "#984ea3",
    "update": "#ff7f00",
}
MARKERS = ["o", "s", "^", "D", "v", "P", "X"]


@dataclass(frozen=True)
class JoinedRow:
    workload: str
    polynomial: str
    bit_width: int
    rounds: int
    n: int
    triton_eval_ms: float
    triton_challenge_ms: float
    triton_update_ms: float
    triton_total_ms: float
    cuda_eval_ms: float
    cuda_challenge_ms: float
    cuda_update_ms: float
    cuda_total_ms: float
    speedup: float
    checksum_match: str
    source: str


@dataclass(frozen=True)
class ScalingRow:
    workload: str
    polynomial: str
    backend: str
    bit_width: int | None
    rounds: int
    n: int
    median_ms: float
    status: str
    input_gib: float | None
    peak_est_gib: float | None
    source: str


def workload_key(workload: str) -> tuple[int, str]:
    try:
        return (WORKLOAD_ORDER.index(workload), workload)
    except ValueError:
        return (len(WORKLOAD_ORDER), workload)


def fnum(value: str | None, default: float = math.nan) -> float:
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def inum(value: str | None, default: int = 0) -> int:
    if value is None or value == "":
        return default
    try:
        return int(float(value))
    except ValueError:
        return default


def get_total(row: dict[str, str], backend: str) -> float:
    for col in (f"{backend}_total_median_ms", f"{backend}_median_ms"):
        value = fnum(row.get(col))
        if math.isfinite(value):
            return value
    return math.nan


def get_phase(row: dict[str, str], backend: str, phase: str) -> float:
    value = fnum(row.get(f"{backend}_{phase}_median_ms"), 0.0)
    return value if math.isfinite(value) else 0.0


def read_joined(
    paths: Iterable[Path],
    include_bad_checksums: bool,
    checksum_counts: Counter[str] | None = None,
) -> list[JoinedRow]:
    """Read joined Triton/CUDA CSVs.

    A timed row is kept only if its checksum_match is "true", i.e. the Triton and CUDA
    checksums agree. ``include_bad_checksums`` also keeps rows whose checksum_match is
    "false" or blank (for debugging; these are not validated results). A CSV without a
    checksum_match column (e.g. the u256 matched comparison, which records only the CUDA
    checksum) is kept with a warning, because it carries no cross-check. Counts of
    excluded and unvalidated rows are added to ``checksum_counts``.
    """
    counts = checksum_counts if checksum_counts is not None else Counter()
    rows: list[JoinedRow] = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            has_checksum_column = "checksum_match" in (reader.fieldnames or [])
            if not has_checksum_column:
                print(
                    f"WARNING: {path} has no checksum_match column; its rows are not "
                    "cross-checked between Triton and CUDA"
                )
            for raw in reader:
                workload = raw.get("workload", "")
                if not workload:
                    continue
                checksum_match = raw.get("checksum_match", "").lower()
                triton_total = get_total(raw, "triton")
                cuda_total = get_total(raw, "cuda")
                if not (math.isfinite(triton_total) and math.isfinite(cuda_total)):
                    continue
                if not has_checksum_column:
                    counts["unvalidated_rows_included"] += 1
                elif checksum_match != "true":
                    reason = "mismatch" if checksum_match == "false" else "not_compared"
                    if not include_bad_checksums:
                        counts[f"excluded_checksum_{reason}_rows"] += 1
                        continue
                    counts["unvalidated_rows_included"] += 1
                    counts[f"included_checksum_{reason}_rows"] += 1
                speedup = triton_total / cuda_total if cuda_total > 0 else math.nan
                rows.append(
                    JoinedRow(
                        workload=workload,
                        polynomial=raw.get("polynomial", WORKLOAD_LABELS.get(workload, workload)),
                        bit_width=inum(raw.get("bit_width")),
                        rounds=inum(raw.get("rounds")),
                        n=inum(raw.get("N")),
                        triton_eval_ms=get_phase(raw, "triton", "eval"),
                        triton_challenge_ms=get_phase(raw, "triton", "challenge"),
                        triton_update_ms=get_phase(raw, "triton", "fold"),
                        triton_total_ms=triton_total,
                        cuda_eval_ms=get_phase(raw, "cuda", "eval"),
                        cuda_challenge_ms=get_phase(raw, "cuda", "challenge"),
                        cuda_update_ms=get_phase(raw, "cuda", "fold"),
                        cuda_total_ms=cuda_total,
                        speedup=speedup,
                        checksum_match=checksum_match,
                        source=str(path),
                    )
                )
    dedup: dict[tuple[str, int, int], JoinedRow] = {}
    for row in rows:
        dedup[(row.workload, row.bit_width, row.rounds)] = row
    return sorted(dedup.values(), key=lambda r: (r.bit_width, workload_key(r.workload), r.rounds))


def read_scaling(paths: Iterable[Path]) -> list[ScalingRow]:
    rows: list[ScalingRow] = []
    for path in paths:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            for raw in reader:
                workload = raw.get("workload", "")
                backend = (
                    raw.get("backend") or raw.get("triton_backend") or raw.get("cuda_backend") or ""
                )
                if not workload or not backend:
                    continue
                status = (
                    raw.get("status") or raw.get("triton_status") or raw.get("cuda_status") or ""
                )
                median_ms = fnum(raw.get("median_ms"))
                if not math.isfinite(median_ms):
                    median_ms = fnum(raw.get("triton_median_ms"))
                if not math.isfinite(median_ms):
                    continue
                rows.append(
                    ScalingRow(
                        workload=workload,
                        polynomial=raw.get("polynomial", WORKLOAD_LABELS.get(workload, workload)),
                        backend=backend,
                        bit_width=inum(raw.get("bit_width")) if raw.get("bit_width") else None,
                        rounds=inum(raw.get("rounds")),
                        n=inum(raw.get("N")),
                        median_ms=median_ms,
                        status=status,
                        input_gib=fnum(raw.get("input_gib")) if raw.get("input_gib") else None,
                        peak_est_gib=fnum(raw.get("peak_est_gib"))
                        if raw.get("peak_est_gib")
                        else None,
                        source=str(path),
                    )
                )
    return sorted(rows, key=lambda r: (r.backend, workload_key(r.workload), r.rounds))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "figure.titlesize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.0,
            "savefig.bbox": "tight",
        }
    )


def save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def space_panels(fig: plt.Figure) -> None:
    # Compact conference figures still need enough room for rotated x labels
    # between subplot rows; otherwise lower-row titles collide with upper labels.
    fig.subplots_adjust(left=0.08, right=0.99, bottom=0.16, top=0.88, hspace=0.72, wspace=0.32)


def maybe_log_y(ax: plt.Axes, values: Iterable[float]) -> None:
    vals = [v for v in values if math.isfinite(v) and v > 0]
    if not vals:
        return
    if max(vals) / min(vals) >= 100.0:
        ax.set_yscale("log")


def no_data_pdf(path: Path, title: str, message: str) -> None:
    fig, ax = plt.subplots(figsize=(3.35, 1.8))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    save(fig, path)


def panel_grid(n: int) -> tuple[int, int]:
    if n <= 1:
        return (1, 1)
    if n <= 2:
        return (1, 2)
    if n <= 4:
        return (2, 2)
    return (2, 3)


def bit_widths(rows: list[JoinedRow]) -> list[int]:
    return sorted({r.bit_width for r in rows if r.bit_width})


def workloads(rows: list[JoinedRow]) -> list[str]:
    present = {r.workload for r in rows}
    return [w for w in WORKLOAD_ORDER if w in present] + sorted(present - set(WORKLOAD_ORDER))


def largest_round_rows(rows: list[JoinedRow]) -> list[JoinedRow]:
    by_key: dict[tuple[str, int], list[JoinedRow]] = defaultdict(list)
    for row in rows:
        by_key[(row.workload, row.bit_width)].append(row)
    out = []
    for group in by_key.values():
        out.append(max(group, key=lambda r: r.rounds))
    return sorted(out, key=lambda r: (r.bit_width, workload_key(r.workload)))


def largest_common_round_rows(rows: list[JoinedRow]) -> list[JoinedRow]:
    out: list[JoinedRow] = []
    for bw in bit_widths(rows):
        subset = [r for r in rows if r.bit_width == bw]
        ws = workloads(subset)
        round_sets = [{r.rounds for r in subset if r.workload == workload} for workload in ws]
        round_sets = [rs for rs in round_sets if rs]
        common_rounds = set.intersection(*round_sets) if round_sets else set()
        if common_rounds:
            target_round = max(common_rounds)
            out.extend(r for r in subset if r.rounds == target_round and r.workload in set(ws))
        else:
            out.extend(largest_round_rows(subset))
    return sorted(out, key=lambda r: (r.bit_width, workload_key(r.workload)))


def common_largest_round(
    rows: list[JoinedRow], bit_width: int, selected_workloads: list[str]
) -> int | None:
    rounds_by_workload = []
    for workload in selected_workloads:
        rs = {r.rounds for r in rows if r.bit_width == bit_width and r.workload == workload}
        if not rs:
            continue
        rounds_by_workload.append(rs)
    if not rounds_by_workload:
        return None
    common = set.intersection(*rounds_by_workload)
    return max(common) if common else None


def plot_backend_runtime_vs_rounds(rows: list[JoinedRow], backend: str, out: Path) -> None:
    bws = bit_widths(rows)
    if not rows:
        no_data_pdf(out, f"{backend.title()} Runtime", "No joined rows available.")
        return
    nr, nc = panel_grid(len(bws))
    fig, axes = plt.subplots(nr, nc, figsize=(3.35 * nc, 2.05 * nr), squeeze=False)
    all_values = []
    for ax, bw in zip(axes.flat, bws, strict=False):
        subset = [r for r in rows if r.bit_width == bw]
        for idx, workload in enumerate(workloads(subset)):
            ws = sorted([r for r in subset if r.workload == workload], key=lambda r: r.rounds)
            if not ws:
                continue
            values = [getattr(r, f"{backend}_total_ms") for r in ws]
            all_values.extend(values)
            ax.plot(
                [r.rounds for r in ws],
                values,
                marker=MARKERS[idx % len(MARKERS)],
                label=WORKLOAD_LABELS.get(workload, workload),
            )
        ax.set_title(f"{bw}-bit field")
        ax.set_xlabel("rounds")
        ax.set_ylabel("total runtime (ms)")
        ax.grid(True, alpha=0.25)
        maybe_log_y(ax, [getattr(r, f"{backend}_total_ms") for r in subset])
    for ax in axes.flat[len(bws) :]:
        ax.axis("off")
    axes.flat[0].legend(loc="best", ncols=1, frameon=False)
    fig.suptitle(f"{backend.title()} total runtime: eval + SHA3 challenge + update")
    space_panels(fig)
    save(fig, out)


def grouped_backend_bars(rows: list[JoinedRow], out: Path) -> None:
    largest = largest_common_round_rows(rows)
    bws = bit_widths(largest)
    if not largest:
        no_data_pdf(out, "Triton vs CUDA Runtime", "No joined rows available.")
        return
    nr, nc = panel_grid(len(bws))
    fig, axes = plt.subplots(nr, nc, figsize=(3.35 * nc, 2.15 * nr), squeeze=False)
    for ax, bw in zip(axes.flat, bws, strict=False):
        subset = [r for r in largest if r.bit_width == bw]
        ws = workloads(subset)
        x = list(range(len(ws)))
        width = 0.38
        triton = [next(r.triton_total_ms for r in subset if r.workload == w) for w in ws]
        cuda = [next(r.cuda_total_ms for r in subset if r.workload == w) for w in ws]
        ax.bar(
            [i - width / 2 for i in x],
            triton,
            width,
            label="Triton",
            color=BACKEND_COLORS["triton"],
        )
        ax.bar([i + width / 2 for i in x], cuda, width, label="CUDA", color=BACKEND_COLORS["cuda"])
        ax.set_xticks(x, [WORKLOAD_LABELS.get(w, w) for w in ws], rotation=35, ha="right")
        ax.set_ylabel("total runtime (ms)")
        ax.set_title(f"{bw}-bit, largest round")
        ax.grid(True, axis="y", alpha=0.25)
        maybe_log_y(ax, triton + cuda)
    for ax in axes.flat[len(bws) :]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Triton vs CUDA total runtime by workload")
    space_panels(fig)
    save(fig, out)


def speedup_by_workload(rows: list[JoinedRow], out: Path) -> None:
    largest = largest_common_round_rows(rows)
    bws = bit_widths(largest)
    if not largest:
        no_data_pdf(out, "CUDA Speedup", "No joined rows available.")
        return
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    ws = workloads(largest)
    x = list(range(len(ws)))
    for idx, bw in enumerate(bws):
        vals = []
        for workload in ws:
            matches = [r for r in largest if r.bit_width == bw and r.workload == workload]
            vals.append(matches[0].speedup if matches else math.nan)
        ax.plot(x, vals, marker=MARKERS[idx % len(MARKERS)], label=f"{bw}-bit")
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xticks(x, [WORKLOAD_LABELS.get(w, w) for w in ws], rotation=35, ha="right")
    ax.set_ylabel("CUDA speedup over Triton")
    ax.set_title("Speedup by workload")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out)


def stacked_breakdown(rows: list[JoinedRow], backend: str, out: Path) -> None:
    largest = largest_common_round_rows(rows)
    bws = bit_widths(largest)
    if not largest:
        no_data_pdf(out, f"{backend.title()} Breakdown", "No joined rows available.")
        return
    nr, nc = panel_grid(len(bws))
    fig, axes = plt.subplots(nr, nc, figsize=(3.35 * nc, 2.2 * nr), squeeze=False)
    for ax, bw in zip(axes.flat, bws, strict=False):
        subset = [r for r in largest if r.bit_width == bw]
        ws = workloads(subset)
        x = list(range(len(ws)))
        eval_vals = [
            getattr(next(r for r in subset if r.workload == w), f"{backend}_eval_ms") for w in ws
        ]
        chal_vals = [
            getattr(next(r for r in subset if r.workload == w), f"{backend}_challenge_ms")
            for w in ws
        ]
        upd_vals = [
            getattr(next(r for r in subset if r.workload == w), f"{backend}_update_ms") for w in ws
        ]
        ax.bar(x, eval_vals, label="Eval", color=PHASE_COLORS["eval"])
        ax.bar(x, chal_vals, bottom=eval_vals, label="SHA3", color=PHASE_COLORS["challenge"])
        bottoms = [a + b for a, b in zip(eval_vals, chal_vals, strict=True)]
        ax.bar(x, upd_vals, bottom=bottoms, label="Update", color=PHASE_COLORS["update"])
        ax.set_xticks(x, [WORKLOAD_LABELS.get(w, w) for w in ws], rotation=35, ha="right")
        ax.set_ylabel("runtime (ms)")
        ax.set_title(f"{bw}-bit, largest round")
        ax.grid(True, axis="y", alpha=0.25)
        maybe_log_y(ax, [a + b + c for a, b, c in zip(eval_vals, chal_vals, upd_vals, strict=True)])
    for ax in axes.flat[len(bws) :]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False)
    fig.suptitle(f"{backend.title()} phase breakdown")
    space_panels(fig)
    save(fig, out)


def challenge_fraction(rows: list[JoinedRow], out: Path) -> None:
    largest = largest_common_round_rows(rows)
    bws = bit_widths(largest)
    if not largest:
        no_data_pdf(out, "SHA3 Challenge Fraction", "No joined rows available.")
        return
    nr, nc = panel_grid(len(bws))
    fig, axes = plt.subplots(nr, nc, figsize=(3.35 * nc, 2.0 * nr), squeeze=False)
    for ax, bw in zip(axes.flat, bws, strict=False):
        subset = [r for r in largest if r.bit_width == bw]
        ws = workloads(subset)
        x = list(range(len(ws)))
        width = 0.38
        tri = [
            100 * r.triton_challenge_ms / r.triton_total_ms
            for w in ws
            for r in subset
            if r.workload == w
        ]
        cu = [
            100 * r.cuda_challenge_ms / r.cuda_total_ms
            for w in ws
            for r in subset
            if r.workload == w
        ]
        ax.bar(
            [i - width / 2 for i in x], tri, width, label="Triton", color=BACKEND_COLORS["triton"]
        )
        ax.bar([i + width / 2 for i in x], cu, width, label="CUDA", color=BACKEND_COLORS["cuda"])
        ax.set_xticks(x, [WORKLOAD_LABELS.get(w, w) for w in ws], rotation=35, ha="right")
        ax.set_ylabel("SHA3 fraction (%)")
        ax.set_title(f"{bw}-bit, largest round")
        ax.grid(True, axis="y", alpha=0.25)
    for ax in axes.flat[len(bws) :]:
        ax.axis("off")
    axes.flat[0].legend(frameon=False)
    fig.suptitle("Transcript challenge overhead")
    space_panels(fig)
    save(fig, out)


def representative_workloads(rows: list[JoinedRow], requested: list[str]) -> list[str]:
    present = set(workloads(rows))
    reps = [w for w in requested if w in present]
    if reps:
        return reps
    fallback = ["poly_ab", "poly_abc", "poly_aabbc", "poly_abcg_plus_deg"]
    reps = [w for w in fallback if w in present]
    return reps or workloads(rows)[:4]


def rows_for_field_width(rows: list[JoinedRow], reps: list[str]) -> list[JoinedRow]:
    selected_reps = representative_workloads(rows, reps)
    bws = bit_widths(rows)
    round_sets = []
    for workload in selected_reps:
        for bw in bws:
            rs = {r.rounds for r in rows if r.workload == workload and r.bit_width == bw}
            if rs:
                round_sets.append(rs)
    common_rounds = set.intersection(*round_sets) if round_sets else set()
    if not common_rounds:
        return []
    target_round = max(common_rounds)
    selected = [
        r
        for r in rows
        if r.workload in set(selected_reps) and r.bit_width in set(bws) and r.rounds == target_round
    ]
    return sorted(selected, key=lambda r: (workload_key(r.workload), r.bit_width))


def field_width_sweep(rows: list[JoinedRow], reps: list[str], out: Path) -> None:
    selected = rows_for_field_width(rows, reps)
    if not selected or len(bit_widths(selected)) < 2:
        no_data_pdf(out, "Field-width Sweep", "Need at least two bit widths in the joined CSVs.")
        return
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    for idx, workload in enumerate(representative_workloads(selected, reps)):
        ws = sorted([r for r in selected if r.workload == workload], key=lambda r: r.bit_width)
        ax.plot(
            [r.bit_width for r in ws],
            [r.triton_total_ms for r in ws],
            marker=MARKERS[idx % len(MARKERS)],
            color=BACKEND_COLORS["triton"],
            linestyle="-",
            label=f"Triton {WORKLOAD_LABELS.get(workload, workload)}",
        )
        ax.plot(
            [r.bit_width for r in ws],
            [r.cuda_total_ms for r in ws],
            marker=MARKERS[idx % len(MARKERS)],
            color=BACKEND_COLORS["cuda"],
            linestyle="--",
            label=f"CUDA {WORKLOAD_LABELS.get(workload, workload)}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256], ["32", "64", "128", "256"])
    ax.set_xlabel("field width (bits)")
    ax.set_ylabel("total runtime (ms)")
    ax.set_title("Field-width sweep")
    ax.grid(True, alpha=0.25)
    maybe_log_y(ax, [r.triton_total_ms for r in selected] + [r.cuda_total_ms for r in selected])
    ax.legend(frameon=False, ncols=1)
    save(fig, out)


def speedup_vs_bit_width(rows: list[JoinedRow], reps: list[str], out: Path) -> None:
    selected = rows_for_field_width(rows, reps)
    if not selected or len(bit_widths(selected)) < 2:
        no_data_pdf(
            out, "Speedup vs Field Width", "Need at least two bit widths in the joined CSVs."
        )
        return
    fig, ax = plt.subplots(figsize=(3.35, 2.1))
    for idx, workload in enumerate(representative_workloads(selected, reps)):
        ws = sorted([r for r in selected if r.workload == workload], key=lambda r: r.bit_width)
        ax.plot(
            [r.bit_width for r in ws],
            [r.speedup for r in ws],
            marker=MARKERS[idx % len(MARKERS)],
            label=WORKLOAD_LABELS.get(workload, workload),
        )
    ax.axhline(1.0, color="black", linewidth=0.7, linestyle="--")
    ax.set_xscale("log", base=2)
    ax.set_xticks([32, 64, 128, 256], ["32", "64", "128", "256"])
    ax.set_xlabel("field width (bits)")
    ax.set_ylabel("CUDA speedup over Triton")
    ax.set_title("Speedup vs field width")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    save(fig, out)


def scaling_plot(scaling_rows: list[ScalingRow], out: Path) -> None:
    ok = [r for r in scaling_rows if r.status in {"", "ok"} and r.n > 0 and r.median_ms > 0]
    if not ok:
        no_data_pdf(out, "Large Device-resident Scaling", "No successful scaling rows supplied.")
        return
    fig, ax = plt.subplots(figsize=(3.35, 2.2))
    grouped: dict[tuple[str, str], list[ScalingRow]] = defaultdict(list)
    for row in ok:
        grouped[(row.backend, row.workload)].append(row)
    for idx, ((backend, workload), group) in enumerate(sorted(grouped.items())):
        group = sorted(group, key=lambda r: r.n)
        label = f"{backend.replace('u256-static-eval-', '').replace('u256-static-eval', 'CUDA')} {WORKLOAD_LABELS.get(workload, workload)}"
        ax.plot(
            [r.n for r in group],
            [r.median_ms for r in group],
            marker=MARKERS[idx % len(MARKERS)],
            label=label,
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("N")
    ax.set_ylabel("runtime (ms)")
    ax.set_title("Large device-resident scaling")
    ax.grid(True, alpha=0.25)
    maybe_log_y(ax, [r.median_ms for r in ok])
    ax.legend(frameon=False)
    save(fig, out)


def make_tables(
    rows: list[JoinedRow],
    scaling_rows: list[ScalingRow],
    reps: list[str],
    table_dir: Path,
    checksum_counts: Counter[str] | None = None,
) -> None:
    clean_fields = [
        "workload",
        "polynomial",
        "bit_width",
        "rounds",
        "N",
        "triton_total_median_ms",
        "cuda_total_median_ms",
        "cuda_speedup_vs_triton",
        "checksum_match",
        "source",
    ]
    write_csv(
        table_dir / "joined_clean_rows.csv",
        [
            {
                "workload": r.workload,
                "polynomial": r.polynomial,
                "bit_width": r.bit_width,
                "rounds": r.rounds,
                "N": r.n,
                "triton_total_median_ms": f"{r.triton_total_ms:.6g}",
                "cuda_total_median_ms": f"{r.cuda_total_ms:.6g}",
                "cuda_speedup_vs_triton": f"{r.speedup:.6g}",
                "checksum_match": r.checksum_match,
                "source": r.source,
            }
            for r in rows
        ],
        clean_fields,
    )

    largest = largest_common_round_rows(rows)
    speed_rows = [
        {
            "workload": r.workload,
            "polynomial": r.polynomial,
            "bit_width": r.bit_width,
            "rounds": r.rounds,
            "N": r.n,
            "triton_total_median_ms": f"{r.triton_total_ms:.6g}",
            "cuda_total_median_ms": f"{r.cuda_total_ms:.6g}",
            "cuda_speedup_vs_triton": f"{r.speedup:.6g}",
        }
        for r in largest
    ]
    write_csv(
        table_dir / "speedup_by_workload_largest_round.csv",
        speed_rows,
        [
            "workload",
            "polynomial",
            "bit_width",
            "rounds",
            "N",
            "triton_total_median_ms",
            "cuda_total_median_ms",
            "cuda_speedup_vs_triton",
        ],
    )

    phase_rows = []
    for r in largest:
        for backend in ("triton", "cuda"):
            total = getattr(r, f"{backend}_total_ms")
            phase_rows.append(
                {
                    "backend": backend,
                    "workload": r.workload,
                    "polynomial": r.polynomial,
                    "bit_width": r.bit_width,
                    "rounds": r.rounds,
                    "eval_ms": f"{getattr(r, f'{backend}_eval_ms'):.6g}",
                    "sha3_challenge_ms": f"{getattr(r, f'{backend}_challenge_ms'):.6g}",
                    "update_ms": f"{getattr(r, f'{backend}_update_ms'):.6g}",
                    "total_ms": f"{total:.6g}",
                    "sha3_fraction": f"{getattr(r, f'{backend}_challenge_ms') / total:.6g}",
                }
            )
    write_csv(
        table_dir / "phase_breakdown_largest_round.csv",
        phase_rows,
        [
            "backend",
            "workload",
            "polynomial",
            "bit_width",
            "rounds",
            "eval_ms",
            "sha3_challenge_ms",
            "update_ms",
            "total_ms",
            "sha3_fraction",
        ],
    )

    selected = rows_for_field_width(rows, reps)
    write_csv(
        table_dir / "field_width_sweep.csv",
        [
            {
                "workload": r.workload,
                "polynomial": r.polynomial,
                "bit_width": r.bit_width,
                "rounds": r.rounds,
                "N": r.n,
                "triton_total_median_ms": f"{r.triton_total_ms:.6g}",
                "cuda_total_median_ms": f"{r.cuda_total_ms:.6g}",
                "cuda_speedup_vs_triton": f"{r.speedup:.6g}",
            }
            for r in selected
        ],
        [
            "workload",
            "polynomial",
            "bit_width",
            "rounds",
            "N",
            "triton_total_median_ms",
            "cuda_total_median_ms",
            "cuda_speedup_vs_triton",
        ],
    )

    ok_scaling = [r for r in scaling_rows if r.status in {"", "ok"}]
    largest_scaling: dict[tuple[str, str], ScalingRow] = {}
    for row in ok_scaling:
        key = (row.backend, row.workload)
        if key not in largest_scaling or row.n > largest_scaling[key].n:
            largest_scaling[key] = row
    if not largest_scaling and rows:
        for row in rows:
            for backend, total in (("triton", row.triton_total_ms), ("cuda", row.cuda_total_ms)):
                key = (backend, row.workload)
                fake = ScalingRow(
                    workload=row.workload,
                    polynomial=row.polynomial,
                    backend=backend,
                    bit_width=row.bit_width,
                    rounds=row.rounds,
                    n=row.n,
                    median_ms=total,
                    status="ok" if row.checksum_match in {"", "true"} else row.checksum_match,
                    input_gib=None,
                    peak_est_gib=None,
                    source=row.source,
                )
                if key not in largest_scaling or fake.n > largest_scaling[key].n:
                    largest_scaling[key] = fake
    write_csv(
        table_dir / "largest_completed_size.csv",
        [
            {
                "backend": r.backend,
                "workload": r.workload,
                "polynomial": r.polynomial,
                "bit_width": "" if r.bit_width is None else r.bit_width,
                "rounds": r.rounds,
                "N": r.n,
                "median_ms": f"{r.median_ms:.6g}",
                "input_gib": "" if r.input_gib is None else f"{r.input_gib:.6g}",
                "peak_est_gib": "" if r.peak_est_gib is None else f"{r.peak_est_gib:.6g}",
                "source": r.source,
            }
            for r in sorted(
                largest_scaling.values(), key=lambda x: (x.backend, workload_key(x.workload))
            )
        ],
        [
            "backend",
            "workload",
            "polynomial",
            "bit_width",
            "rounds",
            "N",
            "median_ms",
            "input_gib",
            "peak_est_gib",
            "source",
        ],
    )

    summary_rows = []
    if rows:
        speedups = [r.speedup for r in rows if math.isfinite(r.speedup)]
        summary_rows.append(
            {
                "metric": "joined_rows",
                "value": len(rows),
            }
        )
        summary_rows.append(
            {
                "metric": "median_cuda_speedup_vs_triton",
                "value": f"{statistics.median(speedups):.6g}" if speedups else "",
            }
        )
        summary_rows.append(
            {
                "metric": "all_checksum_match_true_or_blank",
                "value": all(r.checksum_match in {"", "true"} for r in rows),
            }
        )
    if checksum_counts is not None:
        for metric in ("excluded_checksum_mismatch_rows", "unvalidated_rows_included"):
            summary_rows.append({"metric": metric, "value": checksum_counts[metric]})
    write_csv(table_dir / "plot_summary.csv", summary_rows, ["metric", "value"])


def report_checksum_filter(counts: Counter[str], include_bad_checksums: bool) -> None:
    """Say which rows the checksum rule excluded or let through."""
    mismatch = counts["excluded_checksum_mismatch_rows"]
    not_compared = counts["excluded_checksum_not_compared_rows"]
    if mismatch:
        print(
            f"WARNING: excluded {mismatch} row(s) whose Triton and CUDA checksums disagree "
            "(checksum_match=false)"
        )
    if not_compared:
        print(
            f"WARNING: excluded {not_compared} timed row(s) whose checksums were not compared "
            "(blank checksum_match)"
        )
    if include_bad_checksums and (
        counts["included_checksum_mismatch_rows"] or counts["included_checksum_not_compared_rows"]
    ):
        print(
            "WARNING: --include-bad-checksums: plots and tables include "
            f"{counts['included_checksum_mismatch_rows']} row(s) with checksum_match=false and "
            f"{counts['included_checksum_not_compared_rows']} with a blank checksum_match; "
            "these are not validated results"
        )


def make_plots(
    rows: list[JoinedRow], scaling_rows: list[ScalingRow], reps: list[str], plot_dir: Path
) -> None:
    plot_backend_runtime_vs_rounds(
        rows, "triton", plot_dir / "01_triton_total_runtime_vs_rounds.pdf"
    )
    plot_backend_runtime_vs_rounds(rows, "cuda", plot_dir / "02_cuda_total_runtime_vs_rounds.pdf")
    grouped_backend_bars(rows, plot_dir / "03_triton_vs_cuda_runtime_by_workload.pdf")
    speedup_by_workload(rows, plot_dir / "04_cuda_speedup_by_workload.pdf")
    stacked_breakdown(rows, "triton", plot_dir / "05_triton_phase_breakdown.pdf")
    stacked_breakdown(rows, "cuda", plot_dir / "06_cuda_phase_breakdown.pdf")
    challenge_fraction(rows, plot_dir / "07_sha3_challenge_overhead_fraction.pdf")
    field_width_sweep(rows, reps, plot_dir / "08_field_width_sweep.pdf")
    speedup_vs_bit_width(rows, reps, plot_dir / "09_speedup_vs_bit_width.pdf")
    scaling_plot(scaling_rows, plot_dir / "10_large_device_resident_scaling.pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate zkDuel paper plots from joined CSVs.")
    parser.add_argument("csv", nargs="+", type=Path, help="joined Triton/CUDA comparison CSVs")
    parser.add_argument(
        "--scaling-csv",
        nargs="*",
        type=Path,
        default=[],
        help="optional large device-resident scaling CSVs",
    )
    parser.add_argument("--plot-dir", type=Path, default=Path("results/plots"))
    parser.add_argument("--table-dir", type=Path, default=Path("results/tables"))
    parser.add_argument(
        "--representative-workloads",
        nargs="+",
        default=["poly_abc", "poly_aabbc", "poly_abcg_plus_deg"],
        help="workloads to use for field-width plots",
    )
    parser.add_argument(
        "--include-bad-checksums",
        action="store_true",
        help="also include rows whose checksum_match is false or blank (not validated results; "
        "for debugging only)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_style()
    checksum_counts: Counter[str] = Counter()
    rows = read_joined(
        args.csv, include_bad_checksums=args.include_bad_checksums, checksum_counts=checksum_counts
    )
    report_checksum_filter(checksum_counts, args.include_bad_checksums)
    scaling_rows = read_scaling(args.scaling_csv)
    reps = representative_workloads(rows, args.representative_workloads)
    args.plot_dir.mkdir(parents=True, exist_ok=True)
    args.table_dir.mkdir(parents=True, exist_ok=True)
    make_tables(rows, scaling_rows, reps, args.table_dir, checksum_counts)
    make_plots(rows, scaling_rows, reps, args.plot_dir)
    print(f"wrote plots to {args.plot_dir}")
    print(f"wrote tables to {args.table_dir}")
    print(f"joined rows: {len(rows)}")
    print(f"scaling rows: {len(scaling_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
