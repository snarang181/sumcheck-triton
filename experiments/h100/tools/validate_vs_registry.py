"""Independent correctness check for the Triton field-sweep kernels.

Ground truth is the canonical expression registry in src/zkduel/workloads.py
(plain modular arithmetic, no Montgomery tricks, no hand transliteration of the
kernels). The expected checksum mirrors triton_sweep.py's convention: per round
and point, sum_i (i+1) * limb_i(montgomery(g(t))).

It covers all 32 static workloads, both layouts, both point modes and both challenge
modes, at sizes small enough for Python big ints (rounds 3 and 10 by default; rounds=10
spans several 128-lane blocks, so the partial-sum reduction tree is exercised).

Usage: python experiments/h100/tools/validate_vs_registry.py --bit-width 256 \
          --layout element --point-mode specialized --out results.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / "benchmarks" / "field_sweep"), str(ROOT / "src")]

import triton_sweep as ts  # noqa: E402
from workload_specs import WORKLOAD_ALIASES, WORKLOAD_BY_NAME, WORKLOAD_NAMES  # noqa: E402

from zkduel._expr import expression_degree, expression_vars, normalize_expression  # noqa: E402
from zkduel.fields import get_field  # noqa: E402
from zkduel.workloads import WORKLOADS as REGISTRY  # noqa: E402

SPEC_TO_ALIAS = {v: k for k, v in WORKLOAD_ALIASES.items()}


def registry_expr(spec_name: str):
    return (
        REGISTRY[spec_name] if spec_name.startswith("poly_") else REGISTRY[SPEC_TO_ALIAS[spec_name]]
    )


def spec_terms(spec):
    return sorted(
        (c, tuple(sorted(spec.term_vars[o : o + n])))
        for c, n, o in zip(spec.coeffs, spec.term_lens, spec.term_offsets)
    )


def registry_terms(expr):
    idx = {v: i for i, v in enumerate(expression_vars(expr))}
    return sorted((c, tuple(sorted(idx[v] for v in t))) for c, t in normalize_expression(expr))


def combine(terms):
    acc: dict[tuple, int] = {}
    for c, t in terms:
        acc[t] = acc.get(t, 0) + c
    return sorted((t, c) for t, c in acc.items() if c != 0)


def find_permutation(spec, expr):
    """Return perm with perm[registry_idx] = spec_idx making the term multisets equal, or None."""
    import itertools

    st = spec_terms(spec)
    nv = len(expression_vars(expr))
    if registry_terms(expr) == st:
        return list(range(nv))
    raw = [(c, [i for i in t]) for c, t in registry_terms(expr)]
    for perm in itertools.permutations(range(nv)):
        cand = sorted((c, tuple(sorted(perm[i] for i in t))) for c, t in raw)
        if cand == st:
            return list(perm)
    return None


def metadata_report(spec_name: str) -> dict:
    spec = WORKLOAD_BY_NAME[spec_name]
    expr = registry_expr(spec_name)
    st, rt = spec_terms(spec), registry_terms(expr)
    perm = find_permutation(spec, expr) if spec.num_vars <= 9 or st == rt else None
    return {
        "vars_match": spec.num_vars == len(expression_vars(expr)),
        "degree_match": spec.degree == expression_degree(expr),
        "terms_identical": st == rt,
        "same_polynomial_up_to_var_order": perm is not None,
        "var_perm": perm,
    }


def reference_checksum(
    spec_name: str, bit_width: int, rounds: int, seed: int, challenge_mode: str
) -> int:
    field = get_field(bit_width)
    p = field.modulus
    R = 1 << (32 * field.limbs)
    expr = registry_expr(spec_name)
    names = expression_vars(expr)
    perm = metadata_report(spec_name)["var_perm"]
    if perm is None:
        raise ValueError(
            f"{spec_name}: spec is not a variable relabelling of the registry expression"
        )
    # Table index used by the kernels (spec numbering) for each registry variable.
    idx = {v: perm[i] for i, v in enumerate(names)}
    terms = [(c % p, [idx[v] for v in t]) for c, t in normalize_expression(expr)]
    deg = expression_degree(expr)
    nv = len(names)
    n = 1 << rounds
    tables = [[(seed + 7919 * (v + 1) + 104729 * (o + 1)) % p for o in range(n)] for v in range(nv)]
    checksum = 0
    for rnd in range(rounds):
        half = n // 2
        coeffs = []
        for t in range(deg + 1):
            total = 0
            for o in range(half):
                vals = [(tb[2 * o] + t * (tb[2 * o + 1] - tb[2 * o])) % p for tb in tables]
                for c, factors in terms:
                    prod = c
                    for f in factors:
                        prod = prod * vals[f] % p
                    total += prod
            total %= p
            mont = total * R % p
            limbs = [(mont >> (32 * i)) & 0xFFFFFFFF for i in range(field.limbs)]
            coeffs.append(limbs)
            checksum += sum((i + 1) * x for i, x in enumerate(limbs))
        if rnd + 1 < rounds:
            if challenge_mode == "sha3":
                h = hashlib.sha3_256()
                h.update(b"zkduel:field-sweep:v1")
                h.update(rnd.to_bytes(4, "little"))
                for limbs in coeffs:
                    for x in limbs:
                        h.update(x.to_bytes(4, "little"))
                r = int.from_bytes(h.digest(), "little") % p
            else:
                r = 1
            tables = [
                [(tb[2 * o] + r * (tb[2 * o + 1] - tb[2 * o])) % p for o in range(half)]
                for tb in tables
            ]
            n = half
    return checksum


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bit-width", type=int, required=True)
    ap.add_argument("--layout", choices=["element", "limb"], default="element")
    ap.add_argument("--point-mode", choices=["generic", "specialized"], default="generic")
    ap.add_argument("--workloads", nargs="+", default=WORKLOAD_NAMES)
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--rounds", nargs="+", type=int, default=[3, 10])
    ap.add_argument("--sha3-rounds", nargs="*", type=int, default=[10])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    seed = 1
    cases = [(r, "fixed") for r in args.rounds] + [(r, "sha3") for r in args.sha3_rounds]
    with args.out.open("w") as f:
        for w in args.workloads:
            if w in args.exclude:
                continue
            meta = metadata_report(w)
            for rounds, cm in cases:
                rec = {
                    "workload": w,
                    "bit_width": args.bit_width,
                    "layout": args.layout,
                    "point_mode": args.point_mode,
                    "rounds": rounds,
                    "challenge_mode": cm,
                    **meta,
                }
                t0 = time.time()
                try:
                    row = ts._time_one(
                        w,
                        args.bit_width,
                        rounds,
                        warmups=0,
                        repeats=1,
                        seed=seed,
                        timing_mode="whole-run",
                        challenge_mode=cm,
                        layout=args.layout,
                        point_mode=args.point_mode,
                    )
                    got = int(row["checksum"])
                    want = 2 * reference_checksum(w, args.bit_width, rounds, seed + rounds, cm)
                    rec.update(triton_checksum=got, reference_checksum=want, match=(got == want))
                except Exception as exc:  # record and keep going
                    rec.update(
                        match=None,
                        error=f"{type(exc).__name__}: {exc}",
                        tb=traceback.format_exc()[-800:],
                    )
                rec["seconds"] = round(time.time() - t0, 2)
                f.write(json.dumps(rec) + "\n")
                f.flush()
                print(
                    json.dumps(
                        {
                            k: rec.get(k)
                            for k in (
                                "workload",
                                "bit_width",
                                "layout",
                                "point_mode",
                                "rounds",
                                "challenge_mode",
                                "match",
                                "seconds",
                            )
                        }
                    ),
                    flush=True,
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
