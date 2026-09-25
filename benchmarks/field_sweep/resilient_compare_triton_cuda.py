from __future__ import annotations

import argparse
import contextlib
import csv
import os
import signal
import subprocess
import sys
from pathlib import Path

from workload_specs import (
    WORKLOAD_BY_NAME,
    WORKLOAD_META,
    WORKLOAD_NAMES,
    ZK_WORKLOAD_NAMES,
    canonicalize_workload_names,
    known_workload_names,
)

SUPPORTED_BIT_WIDTHS = [32, 64, 128, 256]
FIELDNAMES = [
    "workload",
    "polynomial",
    "arithmetic",
    "layout",
    "point_mode",
    "field",
    "bit_width",
    "limbs",
    "modulus_hex",
    "rounds",
    "N",
    "vars",
    "degree",
    "terms",
    "warmups",
    "repeats",
    "timing_mode",
    "challenge_mode",
    "triton_backend",
    "triton_first_ms",
    "triton_median_ms",
    "triton_eval_median_ms",
    "triton_challenge_median_ms",
    "triton_fold_median_ms",
    "triton_total_median_ms",
    "triton_status",
    "cuda_backend",
    "cuda_first_ms",
    "cuda_median_ms",
    "cuda_eval_median_ms",
    "cuda_challenge_median_ms",
    "cuda_fold_median_ms",
    "cuda_total_median_ms",
    "cuda_status",
    "checksum_match",
    "triton_checksum",
    "cuda_checksum",
    "cuda_speedup_vs_triton",
    "skip_reason",
    "log_path",
]


TIMEOUT_RC = 124  # GNU `timeout` convention


def run_case(cmd: list[str], log_path: Path, timeout: float | None = None) -> int:
    print("$ " + " ".join(cmd), flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # compare_triton_cuda.py launches the Triton sweep and the CUDA binary as its own
    # children. Start each case in a new session so a timeout kills that whole process
    # group; killing only the direct child leaves the workers compiling or running on
    # the GPU while later cases are being timed.
    proc = subprocess.Popen(
        cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True
    )
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        stdout, _ = proc.communicate()
        captured = (stdout or "") + f"\n[resilient driver] killed after {timeout:.0f}s timeout\n"
        log_path.write_text(captured)
        print(captured, end="", flush=True)
        return TIMEOUT_RC
    except BaseException:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGKILL)
        raise
    log_path.write_text(stdout)
    if stdout:
        print(stdout, end="", flush=True)
    return proc.returncode


def read_first_row(path: Path) -> dict[str, str] | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else None


def skipped_row(
    workload: str,
    bit_width: int,
    rounds: int,
    warmups: int,
    repeats: int,
    timing_mode: str,
    challenge_mode: str,
    layout: str,
    point_mode: str,
    reason: str,
    log_path: Path,
) -> dict[str, str | int]:
    polynomial, vars_, degree, terms = WORKLOAD_META[workload]
    return {
        "workload": workload,
        "polynomial": polynomial,
        "arithmetic": "prime-field-montgomery",
        "layout": layout,
        "point_mode": point_mode,
        "field": "",
        "bit_width": bit_width,
        "limbs": "",
        "modulus_hex": "",
        "rounds": rounds,
        "N": 1 << rounds,
        "vars": vars_,
        "degree": degree,
        "terms": terms,
        "warmups": warmups,
        "repeats": repeats,
        "timing_mode": timing_mode,
        "challenge_mode": challenge_mode,
        "triton_backend": "",
        "triton_first_ms": "",
        "triton_median_ms": "",
        "triton_eval_median_ms": "",
        "triton_challenge_median_ms": "",
        "triton_fold_median_ms": "",
        "triton_total_median_ms": "",
        "triton_status": "skipped",
        "cuda_backend": "",
        "cuda_first_ms": "",
        "cuda_median_ms": "",
        "cuda_eval_median_ms": "",
        "cuda_challenge_median_ms": "",
        "cuda_fold_median_ms": "",
        "cuda_total_median_ms": "",
        "cuda_status": "skipped",
        "checksum_match": "skipped",
        "triton_checksum": "",
        "cuda_checksum": "",
        "cuda_speedup_vs_triton": "",
        "skip_reason": reason,
        "log_path": str(log_path),
    }


def normalize_row(row: dict[str, str], log_path: Path) -> dict[str, str]:
    out = {name: row.get(name, "") for name in FIELDNAMES}
    out["skip_reason"] = ""
    out["log_path"] = str(log_path)
    return out


def validated(row: dict[str, str | int]) -> bool:
    """Both backends ran and their checksums agree: the row may enter reported results."""
    return (
        row.get("triton_status") == "ok"
        and row.get("cuda_status") == "ok"
        and str(row.get("checksum_match", "")).lower() == "true"
    )


def print_summary(rows: list[dict[str, str | int]], out: Path) -> None:
    n_valid = sum(validated(r) for r in rows)
    mismatched = [r for r in rows if str(r.get("checksum_match", "")).lower() == "false"]
    skipped = sum(r.get("skip_reason", "") != "" for r in rows)
    other = len(rows) - n_valid - len(mismatched) - skipped
    print(
        f"summary: {len(rows)} cases: {n_valid} validated (checksums agree), "
        f"{len(mismatched)} checksum mismatch, {skipped} skipped, {other} other not validated "
        f"-> {out}",
        flush=True,
    )
    if mismatched:
        names = ", ".join(
            f"{r['workload']} {r['bit_width']}-bit r={r['rounds']}" for r in mismatched
        )
        print(
            f"WARNING: {len(mismatched)} case(s) have mismatching Triton/CUDA checksums "
            f"(checksum_match=false) and are not validated results: {names}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run field Triton/CUDA comparison one case at a time and continue after failures."
    )
    parser.add_argument("--workloads", nargs="+", default=None)
    parser.add_argument(
        "--zk-only",
        action="store_true",
        help="Restrict default workload set to the 25 ZK workloads "
        "(no effect if --workloads is given)",
    )
    parser.add_argument("--bit-widths", nargs="+", type=int, default=SUPPORTED_BIT_WIDTHS)
    parser.add_argument("--rounds", nargs="+", type=int, default=[8, 10, 12, 14, 16, 18, 20, 22])
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timing-mode", choices=["whole-run", "round-sum"], default="whole-run")
    parser.add_argument("--challenge-mode", choices=["fixed", "sha3"], default="fixed")
    parser.add_argument("--layout", choices=["element", "limb"], default="element")
    parser.add_argument("--point-mode", choices=["generic", "specialized"], default="generic")
    parser.add_argument("--block", type=int, default=128)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--triton-script", default="benchmarks/field_sweep/triton_sweep.py")
    parser.add_argument("--cuda-binary", default="build/field_sumcheck_cuda")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--tmp-dir", type=Path, default=Path("results/tmp_field_resilient"))
    parser.add_argument("--out", type=Path, default=Path("results/triton_cuda_field_resilient.csv"))
    parser.add_argument(
        "--per-case-timeout",
        type=float,
        default=None,
        help="Kill any (workload, bit-width, rounds) case taking longer than "
        "this many seconds and mark it as skipped. Default: no timeout.",
    )
    args = parser.parse_args()

    if args.workloads is None:
        args.workloads = list(ZK_WORKLOAD_NAMES) if args.zk_only else list(WORKLOAD_NAMES)
    args.workloads = canonicalize_workload_names(args.workloads)
    unknown = sorted(set(args.workloads) - set(WORKLOAD_BY_NAME))
    if unknown:
        known = ", ".join(known_workload_names())
        raise SystemExit(f"unknown workloads: {unknown}; known: {known}")
    unsupported = sorted(set(args.bit_widths) - set(SUPPORTED_BIT_WIDTHS))
    if unsupported:
        raise SystemExit(f"unsupported bit widths: {unsupported}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.tmp_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_build:
        subprocess.run(["cuda/build/build_field_sumcheck.sh"], check=True)

    rows: list[dict[str, str | int]] = []
    for workload in args.workloads:
        for bit_width in args.bit_widths:
            for rounds in args.rounds:
                stem = f"{workload}_bw{bit_width}_r{rounds}"
                joined = args.tmp_dir / f"{stem}_joined.csv"
                triton = args.tmp_dir / f"{stem}_triton.csv"
                cuda = args.tmp_dir / f"{stem}_cuda.csv"
                log_path = args.tmp_dir / f"{stem}.log"
                cmd = [
                    args.python,
                    "-u",
                    "benchmarks/field_sweep/compare_triton_cuda.py",
                    "--workloads",
                    workload,
                    "--bit-widths",
                    str(bit_width),
                    "--rounds",
                    str(rounds),
                    "--warmups",
                    str(args.warmups),
                    "--repeats",
                    str(args.repeats),
                    "--seed",
                    str(args.seed),
                    "--timing-mode",
                    args.timing_mode,
                    "--challenge-mode",
                    args.challenge_mode,
                    "--layout",
                    args.layout,
                    "--point-mode",
                    args.point_mode,
                    "--block",
                    str(args.block),
                    "--python",
                    args.python,
                    "--triton-script",
                    args.triton_script,
                    "--cuda-binary",
                    args.cuda_binary,
                    "--skip-build",
                    "--triton-out",
                    str(triton),
                    "--cuda-out",
                    str(cuda),
                    "--out",
                    str(joined),
                ]
                code = run_case(cmd, log_path, timeout=args.per_case_timeout)
                row = read_first_row(joined) if code == 0 else None
                if code == 0 and row is not None:
                    rows.append(normalize_row(row, log_path))
                    if validated(row):
                        print(f"ok: {stem}", flush=True)
                    elif row.get("checksum_match", "").lower() == "false":
                        print(
                            f"WARNING: CHECKSUM MISMATCH: {stem} triton={row.get('triton_checksum', '')} "
                            f"cuda={row.get('cuda_checksum', '')}; not a validated result",
                            flush=True,
                        )
                    else:
                        print(
                            f"WARNING: not validated: {stem} triton_status={row.get('triton_status', '')} "
                            f"cuda_status={row.get('cuda_status', '')} "
                            f"checksum_match={row.get('checksum_match', '')!r}",
                            flush=True,
                        )
                else:
                    if code == TIMEOUT_RC:
                        reason = f"timeout after {args.per_case_timeout:.0f}s"
                    else:
                        reason = f"returncode={code}"
                    rows.append(
                        skipped_row(
                            workload,
                            bit_width,
                            rounds,
                            args.warmups,
                            args.repeats,
                            args.timing_mode,
                            args.challenge_mode,
                            args.layout,
                            args.point_mode,
                            reason,
                            log_path,
                        )
                    )
                    print(f"skipped: {stem} {reason}", flush=True)

                with args.out.open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
                    writer.writeheader()
                    writer.writerows(rows)
                print(args.out, flush=True)

    print_summary(rows, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
