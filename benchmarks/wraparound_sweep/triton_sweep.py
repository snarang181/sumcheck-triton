"""Full SumCheck bit-width sweep using Triton kernels.

This is an ablation benchmark, not the main cryptographic-field backend. It runs
actual SumCheck round/eval/fold kernels over 1, 2, 4, and 8 32-bit limbs using
wraparound arithmetic modulo 2^(32 * limbs). The goal is to isolate how limb
width changes the full SumCheck round/eval/fold kernel shape. This is not a
prime-field Montgomery benchmark.
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TRITON_CACHE_DIR", str(_ROOT / ".triton-cache-sumcheck-bitwidth"))
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
import triton
import triton.language as tl

BLOCK = 128
SUPPORTED_BIT_WIDTHS = (32, 64, 128, 256)

WORKLOADS = [
    ("poly_a", "a", 1, 1, 1, 1),
    ("poly_ab", "a*b", 2, 2, 2, 1),
    ("poly_ab_plus_c", "a*b + c", 3, 3, 2, 2),
    ("poly_abc", "a*b*c", 4, 3, 3, 1),
    ("poly_aabbc", "a*a*b*b*c", 5, 3, 5, 1),
    ("poly_abc_plus_de", "a*b*c + d*e", 6, 5, 3, 2),
    ("poly_abcg_plus_deg", "a*b*c*g + d*e*g", 7, 6, 4, 2),
]
WORKLOAD_BY_NAME = {
    name: (poly, ident, num_vars, degree, terms)
    for name, poly, ident, num_vars, degree, terms in WORKLOADS
}


@triton.jit
def _zero_tuple(offs, LIMBS: tl.constexpr):
    z = (offs * 0).to(tl.uint64)
    out = ()
    for _ in tl.static_range(0, LIMBS):
        out += (z,)
    return out


@triton.jit
def _load_elem(ptr, idxs, mask, LIMBS: tl.constexpr):
    out = ()
    for i in tl.static_range(0, LIMBS):
        out += (tl.load(ptr + idxs * LIMBS + i, mask=mask, other=0).to(tl.uint64),)
    return out


@triton.jit
def _store_elem(ptr, idxs, value, mask, LIMBS: tl.constexpr):
    for i in tl.static_range(0, LIMBS):
        tl.store(ptr + idxs * LIMBS + i, value[i].to(tl.uint32), mask=mask)


@triton.jit
def _add_wrap(a, b, LIMBS: tl.constexpr):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    carry = a[0] * 0
    out = ()
    for i in tl.static_range(0, LIMBS):
        uv = a[i] + b[i] + carry
        out += (uv & mask32,)
        carry = uv >> 32
    return out


@triton.jit
def _sub_wrap(a, b, LIMBS: tl.constexpr):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    borrow = a[0] * 0
    out = ()
    for i in tl.static_range(0, LIMBS):
        bi = b[i] + borrow
        ai = a[i]
        out += ((ai - bi) & mask32,)
        borrow = tl.where(ai < bi, 1, 0).to(tl.uint64)
    return out


@triton.jit
def _mul_low(a, b, LIMBS: tl.constexpr):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    carry = a[0] * 0
    out = ()
    for k in tl.static_range(0, LIMBS):
        acc = carry
        for i in tl.static_range(0, LIMBS):
            if i <= k:
                acc += a[i] * b[k - i]
        out += (acc & mask32,)
        carry = acc >> 32
    return out


@triton.jit
def _mul_small(a, scalar, LIMBS: tl.constexpr):
    mask32 = a[0] * 0 + 0xFFFFFFFF
    carry = a[0] * 0
    s = scalar.to(tl.uint64)
    out = ()
    for i in tl.static_range(0, LIMBS):
        uv = a[i] * s + carry
        out += (uv & mask32,)
        carry = uv >> 32
    return out


@triton.jit
def _sum_block_wrap(value, mask, LIMBS: tl.constexpr):
    mask32 = tl.full((), 0xFFFFFFFF, tl.uint64)
    carry = tl.full((), 0, tl.uint64)
    out = ()
    for i in tl.static_range(0, LIMBS):
        limb = tl.where(mask, value[i], 0)
        total = tl.sum(limb, axis=0) + carry
        out += (total & mask32,)
        carry = total >> 32
    return out


@triton.jit
def _eval_workload(p0, p1, p2, p3, p4, p5, WORKLOAD: tl.constexpr, LIMBS: tl.constexpr):
    if WORKLOAD == 1:
        return p0
    if WORKLOAD == 2:
        return _mul_low(p0, p1, LIMBS)
    if WORKLOAD == 3:
        return _add_wrap(_mul_low(p0, p1, LIMBS), p2, LIMBS)
    if WORKLOAD == 4:
        return _mul_low(_mul_low(p0, p1, LIMBS), p2, LIMBS)
    if WORKLOAD == 5:
        aa = _mul_low(p0, p0, LIMBS)
        bb = _mul_low(p1, p1, LIMBS)
        return _mul_low(_mul_low(aa, bb, LIMBS), p2, LIMBS)
    if WORKLOAD == 6:
        abc = _mul_low(_mul_low(p0, p1, LIMBS), p2, LIMBS)
        de = _mul_low(p3, p4, LIMBS)
        return _add_wrap(abc, de, LIMBS)
    abcg = _mul_low(_mul_low(_mul_low(p0, p1, LIMBS), p2, LIMBS), p3, LIMBS)
    deg = _mul_low(_mul_low(p4, p5, LIMBS), p3, LIMBS)
    return _add_wrap(abcg, deg, LIMBS)


@triton.jit(
    do_not_specialize=["half_n", "point"],
    do_not_specialize_on_alignment=["tables_ptr", "partials_ptr"],
)
def _round_eval_kernel(
    tables_ptr,
    partials_ptr,
    half_n,
    point,
    WORKLOAD: tl.constexpr,
    NUM_VARS: tl.constexpr,
    LIMBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < half_n
    row_stride = 2 * half_n
    twice = 2 * offs

    p0 = _zero_tuple(offs, LIMBS)
    p1 = _zero_tuple(offs, LIMBS)
    p2 = _zero_tuple(offs, LIMBS)
    p3 = _zero_tuple(offs, LIMBS)
    p4 = _zero_tuple(offs, LIMBS)
    p5 = _zero_tuple(offs, LIMBS)

    for var in tl.static_range(0, 6):
        if var < NUM_VARS:
            base = tables_ptr + var * row_stride * LIMBS
            even = _load_elem(base, twice, mask, LIMBS)
            odd = _load_elem(base, twice + 1, mask, LIMBS)
            diff = _sub_wrap(odd, even, LIMBS)
            value = _add_wrap(even, _mul_small(diff, point, LIMBS), LIMBS)
            if var == 0:
                p0 = value
            if var == 1:
                p1 = value
            if var == 2:
                p2 = value
            if var == 3:
                p3 = value
            if var == 4:
                p4 = value
            if var == 5:
                p5 = value

    total = _eval_workload(p0, p1, p2, p3, p4, p5, WORKLOAD, LIMBS)
    partial = _sum_block_wrap(total, mask, LIMBS)
    for i in tl.static_range(0, LIMBS):
        tl.store(partials_ptr + pid * LIMBS + i, partial[i].to(tl.uint32))


@triton.jit(
    do_not_specialize=["n"],
    do_not_specialize_on_alignment=["partials_ptr"],
)
def _reduce_pairs_kernel(partials_ptr, n, LIMBS: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    out_n = (n + 1) // 2
    mask = offs < out_n
    lhs = 2 * offs
    rhs = lhs + 1
    a = _load_elem(partials_ptr, lhs, mask, LIMBS)
    b = _load_elem(partials_ptr, rhs, mask & (rhs < n), LIMBS)
    out = _add_wrap(a, b, LIMBS)
    _store_elem(partials_ptr, offs, out, mask, LIMBS)


@triton.jit(
    do_not_specialize=["half_n", "challenge"],
    do_not_specialize_on_alignment=["tables_ptr", "out_ptr"],
)
def _fold_kernel(
    tables_ptr,
    out_ptr,
    half_n,
    challenge,
    NUM_VARS: tl.constexpr,
    LIMBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = NUM_VARS * half_n
    mask = offs < total
    var = offs // half_n
    local = offs - var * half_n
    row_stride = 2 * half_n
    base = tables_ptr + var * row_stride * LIMBS
    even = _load_elem(base, 2 * local, mask, LIMBS)
    odd = _load_elem(base, 2 * local + 1, mask, LIMBS)
    diff = _sub_wrap(odd, even, LIMBS)
    folded = _add_wrap(even, _mul_small(diff, challenge, LIMBS), LIMBS)
    _store_elem(out_ptr, offs, folded, mask, LIMBS)


def _grid(n: int) -> tuple[int]:
    return (triton.cdiv(n, BLOCK),)


def _make_tables(
    num_vars: int, n: int, limbs: int, device: torch.device, seed: int
) -> torch.Tensor:
    tables = torch.zeros((num_vars, n, limbs), dtype=torch.uint32, device=device)
    offsets = torch.arange(1, n + 1, dtype=torch.int64, device=device)
    wrap = 1 << 32
    for var in range(num_vars):
        values = seed + 7919 * (var + 1) + 104729 * offsets
        tables[var, :, 0] = torch.remainder(values, wrap).to(torch.uint32)
    return tables.contiguous()


def _reduce_partials(partials: torch.Tensor, n: int, limbs: int) -> None:
    while n > 1:
        out_n = (n + 1) // 2
        _reduce_pairs_kernel[_grid(out_n)](partials, n, LIMBS=limbs, BLOCK_SIZE=BLOCK)
        n = out_n


def _run_once(
    workload_id: int,
    num_vars: int,
    degree: int,
    tables0: torch.Tensor,
    scratch_a: torch.Tensor,
    scratch_b: torch.Tensor,
    partials: torch.Tensor,
    rounds: int,
    limbs: int,
) -> int:
    cur = tables0
    write_to_a = True
    cur_n = 1 << rounds
    checksum = 0
    for round_idx in range(rounds):
        half = cur_n // 2
        num_blocks = triton.cdiv(half, BLOCK)
        for point in range(degree + 1):
            _round_eval_kernel[(num_blocks,)](
                cur,
                partials,
                half,
                point,
                WORKLOAD=workload_id,
                NUM_VARS=num_vars,
                LIMBS=limbs,
                BLOCK_SIZE=BLOCK,
            )
            _reduce_partials(partials, num_blocks, limbs)
            host_limbs = partials[0, :limbs].detach().cpu().tolist()
            checksum += sum((i + 1) * int(v) for i, v in enumerate(host_limbs))

        if round_idx + 1 < rounds:
            total = num_vars * half
            nxt = scratch_a if write_to_a else scratch_b
            _fold_kernel[_grid(total)](
                cur,
                nxt,
                half,
                1,
                NUM_VARS=num_vars,
                LIMBS=limbs,
                BLOCK_SIZE=BLOCK,
            )
            cur = nxt
            write_to_a = not write_to_a
            cur_n = half
    return checksum


def _time_one(workload: str, bit_width: int, rounds: int, warmups: int, repeats: int, seed: int):
    if bit_width not in SUPPORTED_BIT_WIDTHS:
        raise ValueError(f"unsupported bit width {bit_width}; expected {SUPPORTED_BIT_WIDTHS}")
    limbs = bit_width // 32
    poly, workload_id, num_vars, degree, terms = WORKLOAD_BY_NAME[workload]
    n = 1 << rounds
    device = torch.device("cuda")
    tables = _make_tables(num_vars, n, limbs, device, seed + rounds)
    scratch_a = torch.empty_like(tables)
    scratch_b = torch.empty_like(tables)
    max_partials = triton.cdiv(n // 2, BLOCK)
    partials = torch.empty((max_partials, limbs), dtype=torch.uint32, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    first_checksum = _run_once(
        workload_id, num_vars, degree, tables, scratch_a, scratch_b, partials, rounds, limbs
    )
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1000.0

    checksum = first_checksum
    for _ in range(warmups):
        checksum += _run_once(
            workload_id, num_vars, degree, tables, scratch_a, scratch_b, partials, rounds, limbs
        )
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        checksum += _run_once(
            workload_id, num_vars, degree, tables, scratch_a, scratch_b, partials, rounds, limbs
        )
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - t0) * 1000.0)

    median_ms = statistics.median(samples)
    input_bytes = num_vars * n * limbs * 4
    return {
        "workload": workload,
        "polynomial": poly,
        "backend": "triton-sumcheck-bitwidth-wrap",
        "arithmetic": "wrap",
        "bit_width": bit_width,
        "limbs": limbs,
        "rounds": rounds,
        "N": n,
        "vars": num_vars,
        "degree": degree,
        "terms": terms,
        "warmups": warmups,
        "repeats": repeats,
        "first_ms": f"{first_ms:.3f}",
        "median_ms": f"{median_ms:.3f}",
        "input_mib": f"{input_bytes / (1024**2):.6f}",
        "checksum": checksum,
        "status": "ok",
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit-widths", type=int, nargs="+", default=list(SUPPORTED_BIT_WIDTHS))
    ap.add_argument("--workloads", nargs="+", default=[x[0] for x in WORKLOADS])
    ap.add_argument("--rounds", type=int, nargs="+", default=[8, 10, 12, 14, 16])
    ap.add_argument("--warmups", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="results/triton_wraparound_sweep.csv")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for benchmarks/wraparound_sweep/triton_sweep.py")

    unknown = sorted(set(args.workloads) - set(WORKLOAD_BY_NAME))
    if unknown:
        raise SystemExit(f"unknown workloads: {unknown}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "workload",
        "polynomial",
        "backend",
        "arithmetic",
        "bit_width",
        "limbs",
        "rounds",
        "N",
        "vars",
        "degree",
        "terms",
        "warmups",
        "repeats",
        "first_ms",
        "median_ms",
        "input_mib",
        "checksum",
        "status",
    ]

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for workload in args.workloads:
            for rounds in args.rounds:
                for bit_width in args.bit_widths:
                    row = _time_one(
                        workload,
                        bit_width,
                        rounds,
                        warmups=args.warmups,
                        repeats=args.repeats,
                        seed=args.seed,
                    )
                    writer.writerow(row)
                    f.flush()
                    print(
                        f"{workload:24s} bit_width={bit_width:<3d} rounds={rounds:<2d} "
                        f"first_ms={row['first_ms']} median_ms={row['median_ms']} "
                        f"checksum={row['checksum']}",
                        flush=True,
                    )

    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
