"""Reusable SumCheck expression presets for Triton experiments."""

from __future__ import annotations

from ._expr import Expression

# Deterministic placeholder challenge used by registry-only zkPHIRE presets.
# Benchmark scripts that model transcript challenges pass their own challenge mode.
alpha = 7

WORKLOADS: dict[str, Expression] = {
    # Small polynomial ladder for Triton fusion experiments.
    "poly_a": [["a"]],
    "poly_ab": [["a", "b"]],
    "poly_ab_plus_c": [["a", "b"], ["c"]],
    "poly_abc": [["a", "b", "c"]],
    "poly_aabbc": [["a", "a", "b", "b", "c"]],
    "poly_abc_plus_de": [["a", "b", "c"], ["d", "e"]],
    "poly_abcg_plus_deg": [["a", "b", "c", "g"], ["d", "e", "g"]],
    # Backward-compatible/simple names.
    "linear": [["a"]],
    "quadratic": [["a", "b"]],
    "cubic_plus_quad": [["a", "b", "c"], ["d", "e"]],
    "opencheck": [
        ["y1", "k1"],
        ["y2", "k2"],
        ["y3", "k3"],
        ["y4", "k4"],
        ["y5", "k5"],
        ["y6", "k6"],
    ],
    "vanilla_zerocheck_hp": [
        ["qL", "w1", "fz1"],
        ["qR", "w2", "fz1"],
        ["qM", "w1", "w2", "fz1"],
        (-1, ["qO", "w3", "fz1"]),
        ["qC", "fz1"],
    ],
    "vanilla_permcheck_hp": [
        ["pi", "fz2"],
        (-1, ["p1", "p2", "fz2"]),
        (alpha, ["phi", "D1", "D2", "D3", "fz2"]),
        (-alpha, ["N1", "N2", "N3", "fz2"]),
    ],
    "jellyfish_zerocheck_hp": [
        ["qL", "w1", "fz"],
        ["qR", "w2", "fz"],
        ["qM", "w1", "w2", "fz"],
        (-1, ["qO", "w3", "fz"]),
        ["q1", "w1", "fz"],
        ["q2", "w2", "fz"],
        ["q3", "w3", "fz"],
        ["q4", "w4", "fz"],
        (-1, ["qO", "w5", "fz"]),
        ["qM1", "w1", "w2", "fz"],
        ["qM2", "w3", "w4", "fz"],
        ["qH1", "w1", "w1", "w1", "w1", "w1", "fz"],
        ["qH2", "w2", "w2", "w2", "w2", "w2", "fz"],
        ["qH3", "w3", "w3", "w3", "w3", "w3", "fz"],
        ["qH4", "w4", "w4", "w4", "w4", "w4", "fz"],
        ["qECC", "w1", "w2", "w3", "w4", "fz"],
        ["qc", "fz"],
    ],
    "jellyfish_permcheck_hp": [
        ["pi", "fz"],
        (-1, ["p1", "p2", "fz"]),
        (alpha, ["phi", "D1", "D2", "D3", "D4", "D5", "fz"]),
        (-alpha, ["N1", "N2", "N3", "N4", "N5", "fz"]),
    ],
    "verifiable_asics": [["qadd", "a"], ["qadd", "b"], ["qmul", "a", "b"]],
    "spartan_1": [["A", "B", "fz"], (-1, ["C", "fz"])],
    "spartan_2": [["ABC", "Z"]],
    "witness_non_id_point": [
        ["q_non-id-point", "y", "y"],
        (-1, ["q_non-id-point", "x", "x", "x"]),
        (-5, ["q_non-id-point"]),
    ],
    "witness_id_point_1": [
        ["q_point", "x", "y", "y"],
        (-1, ["q_point", "x", "x", "x", "x"]),
        (-5, ["q_point", "x"]),
    ],
    "witness_id_point_2": [
        ["q_point", "y", "y", "y"],
        (-1, ["q_point", "y", "x", "x", "x"]),
        (-5, ["q_point", "y"]),
    ],
    "incomplete_addition_1": [
        ["q_add-incomplete", "x_r", "x_p", "x_p"],
        (-2, ["q_add-incomplete", "x_r", "x_p", "x_q"]),
        ["q_add-incomplete", "x_r", "x_q", "x_q"],
        ["q_add-incomplete", "x_q", "x_p", "x_p"],
        (-2, ["q_add-incomplete", "x_q", "x_q", "x_p"]),
        ["q_add-incomplete", "x_q", "x_q", "x_q"],
        ["q_add-incomplete", "x_p", "x_p", "x_p"],
        (-2, ["q_add-incomplete", "x_p", "x_p", "x_q"]),
        ["q_add-incomplete", "x_p", "x_q", "x_q"],
        (-1, ["q_add-incomplete", "y_p", "y_p"]),
        (2, ["q_add-incomplete", "y_p", "y_q"]),
        (-1, ["q_add-incomplete", "y_q", "y_q"]),
    ],
    "incomplete_addition_2": [
        ["q_add-incomplete", "y_r", "x_p"],
        (-1, ["q_add-incomplete", "y_r", "x_q"]),
        ["q_add-incomplete", "y_q", "x_p"],
        (
            -1,
            ["q_add-incomplete", "y_q", "x_q"],
        ),  # this term can be cancelled but kept it in zkphire
        (-1, ["q_add-incomplete", "y_p", "x_q"]),
        ["q_add-incomplete", "y_p", "x_r"],
        (-1, ["q_add-incomplete", "y_q", "x_r"]),
        ["q_add-incomplete", "y_q", "x_q"],  # this term can be cancelled but kept it in zkphire
    ],
    "complete_addition_1": [
        ["q_add", "x_q", "x_q", "lambda"],
        (-2, ["q_add", "x_q", "x_p", "lambda"]),
        ["q_add", "x_p", "x_p", "lambda"],
        (-1, ["q_add", "x_q", "y_q"]),
        ["q_add", "x_q", "y_p"],
        ["q_add", "x_p", "y_q"],
        (-1, ["q_add", "x_p", "y_p"]),
    ],
    "complete_addition_2": [
        (2, ["q_add", "y_p", "lambda"]),
        (-3, ["q_add", "x_p", "x_p"]),
        (-2, ["q_add", "x_q", "y_p", "lambda", "alpha"]),
        (3, ["q_add", "x_q", "x_p", "x_p", "alpha"]),
        (2, ["q_add", "x_p", "y_p", "lambda", "alpha"]),
        (-3, ["q_add", "x_p", "x_p", "x_p", "alpha"]),
    ],
    "complete_addition_3": [
        ["q_add", "x_p", "x_q", "x_q", "lambda", "lambda"],
        (-1, ["q_add", "x_p", "x_q", "x_q", "x_q"]),
        (-1, ["q_add", "x_p", "x_q", "x_q", "x_r"]),
        (-1, ["q_add", "x_p", "x_p", "x_q", "x_q"]),  # this term can be cancelled
        (-1, ["q_add", "x_p", "x_p", "x_q", "lambda", "lambda"]),
        ["q_add", "x_p", "x_p", "x_p", "x_q"],
        ["q_add", "x_p", "x_p", "x_q", "x_r"],
        ["q_add", "x_p", "x_p", "x_q", "x_q"],  # this term can be cancelled
    ],
    "complete_addition_4": [
        ["q_add", "x_p", "x_p", "x_q", "x_q", "lambda"],
        (-1, ["q_add", "x_p", "x_q", "x_q", "x_r", "lambda"]),
        (-1, ["q_add", "x_p", "x_q", "x_q", "y_p"]),
        (-1, ["q_add", "x_p", "x_q", "x_q", "y_r"]),
        (-1, ["q_add", "x_p", "x_p", "x_p", "x_q", "lambda"]),
        ["q_add", "x_p", "x_p", "x_q", "x_r", "lambda"],
        ["q_add", "x_p", "x_p", "x_q", "y_p"],
        ["q_add", "x_p", "x_p", "x_q", "y_r"],
    ],
    "complete_addition_5": [
        ["q_add", "x_p", "x_q", "y_q", "lambda", "lambda"],
        ["q_add", "x_p", "x_q", "y_p", "lambda", "lambda"],
        (-1, ["q_add", "x_p", "x_p", "x_q", "y_q"]),
        (-1, ["q_add", "x_p", "x_p", "x_q", "y_p"]),
        (-1, ["q_add", "x_p", "x_q", "x_q", "y_q"]),
        (-1, ["q_add", "x_p", "x_q", "x_q", "y_p"]),
        (-1, ["q_add", "x_p", "x_q", "y_q", "x_r"]),
        (-1, ["q_add", "x_p", "x_q", "y_p", "x_r"]),
    ],
    "complete_addition_6": [
        ["q_add", "x_p", "x_p", "x_q", "y_q", "lambda"],
        ["q_add", "x_p", "x_p", "x_q", "y_p", "lambda"],
        (-1, ["q_add", "x_p", "x_q", "y_p", "x_r", "lambda"]),
        (-1, ["q_add", "x_p", "x_q", "y_q", "x_r", "lambda"]),
        (-1, ["q_add", "x_p", "x_q", "y_p", "y_p"]),
        (-1, ["q_add", "x_p", "x_q", "y_p", "y_q"]),
        (-1, ["q_add", "x_p", "x_q", "y_p", "y_r"]),
        (-1, ["q_add", "x_p", "x_q", "y_q", "y_r"]),
    ],
    "complete_addition_7": [
        ["q_add", "x_r"],
        (-1, ["q_add", "x_q"]),
        (-1, ["q_add", "x_p", "x_r", "beta"]),
        ["q_add", "x_p", "x_q", "beta"],
    ],
    "complete_addition_8": [
        ["q_add", "y_r"],
        (-1, ["q_add", "y_q"]),
        (-1, ["q_add", "x_p", "y_r", "beta"]),
        ["q_add", "x_p", "y_q", "beta"],
    ],
    "complete_addition_9": [
        ["q_add", "x_r"],
        (-1, ["q_add", "x_p"]),
        (-1, ["q_add", "x_q", "x_r", "gamma"]),
        ["q_add", "x_p", "x_q", "gamma"],
    ],
    "complete_addition_10": [
        ["q_add", "y_r"],
        (-1, ["q_add", "y_p"]),
        (-1, ["q_add", "x_q", "y_r", "gamma"]),
        ["q_add", "x_q", "y_p", "gamma"],
    ],
    "complete_addition_11": [
        ["q_add", "x_r"],
        (-1, ["q_add", "x_q", "x_r", "alpha"]),
        ["q_add", "x_p", "x_r", "alpha"],
        (-1, ["q_add", "y_q", "x_r", "delta"]),
        (-1, ["q_add", "y_p", "x_r", "delta"]),
    ],
    "complete_addition_12": [
        ["q_add", "y_r"],
        (-1, ["q_add", "x_q", "y_r", "alpha"]),
        ["q_add", "x_p", "y_r", "alpha"],
        (-1, ["q_add", "y_q", "y_r", "delta"]),
        (-1, ["q_add", "y_p", "y_r", "delta"]),
    ],
}

# Human-readable aliases for the polynomial ladder.
WORKLOADS["a"] = WORKLOADS["poly_a"]
WORKLOADS["ab"] = WORKLOADS["poly_ab"]
WORKLOADS["ab_plus_c"] = WORKLOADS["poly_ab_plus_c"]
WORKLOADS["abc"] = WORKLOADS["poly_abc"]
WORKLOADS["aabbc"] = WORKLOADS["poly_aabbc"]
WORKLOADS["abc_plus_de"] = WORKLOADS["poly_abc_plus_de"]
WORKLOADS["abcg_plus_deg"] = WORKLOADS["poly_abcg_plus_deg"]

# Aliases matching the zkPHIRE Table-I IDs we keep using for HyperPlonk-shaped
# benchmarking.
WORKLOADS["zkphire_0"] = WORKLOADS["verifiable_asics"]
WORKLOADS["zkphire_1"] = WORKLOADS["spartan_1"]
WORKLOADS["zkphire_2"] = WORKLOADS["spartan_2"]

WORKLOADS["zkphire_3"] = WORKLOADS["witness_non_id_point"]
WORKLOADS["zkphire_4"] = WORKLOADS["witness_id_point_1"]
WORKLOADS["zkphire_5"] = WORKLOADS["witness_id_point_2"]
WORKLOADS["zkphire_6"] = WORKLOADS["incomplete_addition_1"]
WORKLOADS["zkphire_7"] = WORKLOADS["incomplete_addition_2"]

WORKLOADS["zkphire_8"] = WORKLOADS["complete_addition_1"]
WORKLOADS["zkphire_9"] = WORKLOADS["complete_addition_2"]
WORKLOADS["zkphire_10"] = WORKLOADS["complete_addition_3"]
WORKLOADS["zkphire_11"] = WORKLOADS["complete_addition_4"]
WORKLOADS["zkphire_12"] = WORKLOADS["complete_addition_5"]
WORKLOADS["zkphire_13"] = WORKLOADS["complete_addition_6"]
WORKLOADS["zkphire_14"] = WORKLOADS["complete_addition_7"]
WORKLOADS["zkphire_15"] = WORKLOADS["complete_addition_8"]
WORKLOADS["zkphire_16"] = WORKLOADS["complete_addition_9"]
WORKLOADS["zkphire_17"] = WORKLOADS["complete_addition_10"]
WORKLOADS["zkphire_18"] = WORKLOADS["complete_addition_11"]
WORKLOADS["zkphire_19"] = WORKLOADS["complete_addition_12"]

WORKLOADS["zkphire_20"] = WORKLOADS["vanilla_zerocheck_hp"]
WORKLOADS["zkphire_21"] = WORKLOADS["vanilla_permcheck_hp"]
WORKLOADS["zkphire_22"] = WORKLOADS["jellyfish_zerocheck_hp"]
WORKLOADS["zkphire_23"] = WORKLOADS["jellyfish_permcheck_hp"]
WORKLOADS["zkphire_24"] = WORKLOADS["opencheck"]

# Backward-compatible HyperPlonk aliases.
WORKLOADS["hyperplonk_zero"] = WORKLOADS["vanilla_zerocheck_hp"]
WORKLOADS["hyperplonk_perm"] = WORKLOADS["vanilla_permcheck_hp"]
WORKLOADS["hyperplonk_open"] = WORKLOADS["opencheck"]


def get_workload(name: str) -> Expression:
    try:
        return WORKLOADS[name]
    except KeyError as exc:
        known = ", ".join(sorted(WORKLOADS))
        raise KeyError(f"unknown workload {name!r}; known: {known}") from exc
