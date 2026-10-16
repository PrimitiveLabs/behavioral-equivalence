"""Discovery-only refinement selectors with explicit matched-capacity controls."""

import numpy as np
from .random_response import supported_witnesses
from .response_refinement import split_response_witness, random_capacity_refinement


def select_partition(
    discovery,
    estimation_support,
    initial,
    steps,
    policy,
    seed,
    *,
    min_cases=2000,
    min_users=50,
    minimum_probability_gap=0.0,
    exact_count_gate=False,
):
    if (
        policy
        not in {"targeted", "random_environment", "fixed_grid", "uncertainty", "random_witness"}
        or not isinstance(steps, int)
        or steps < 0
        or not np.isfinite(minimum_probability_gap)
        or not 0 <= minimum_probability_gap <= 1
        or not isinstance(exact_count_gate, bool)
    ):
        raise ValueError("Known selector and nonnegative integer split budget required")
    rng = np.random.default_rng(seed)
    blocks = initial
    trace = []
    eligible = dict(discovery)
    eligible["counts"] = np.minimum(discovery["counts"], estimation_support["counts"])
    eligible["users"] = np.minimum(discovery["users"], estimation_support["users"])
    for step in range(steps):
        candidates = supported_witnesses(eligible, blocks, min_cases=min_cases, min_users=min_users)
        if exact_count_gate:
            from fractions import Fraction
            from .neutral_response import exact_response_gap

            if "positives" not in discovery:
                raise ValueError("Exact count gating requires success counts")
            threshold = Fraction.from_float(float(minimum_probability_gap))
            candidates = [
                w
                for w in candidates
                if exact_response_gap(
                    discovery["counts"], discovery["positives"], w["states"], w["environment"]
                )
                > threshold
            ]
        else:
            candidates = [
                w
                for w in candidates
                if w["js_nats"] > 0 and w["probability_gap"] > minimum_probability_gap
            ]

        if not candidates:
            break
        if policy == "targeted":
            chosen = candidates[0]
        elif policy == "random_witness":
            chosen = candidates[int(rng.integers(len(candidates)))]
        elif policy in {"random_environment", "fixed_grid"}:
            environments = sorted({w["environment"] for w in candidates})
            env = (
                int(rng.choice(environments))
                if policy == "random_environment"
                else environments[step % len(environments)]
            )
            chosen = next(w for w in candidates if w["environment"] == env)
        else:

            def uncertainty(w):
                a, b = w["states"]
                e = w["environment"]
                p, q = discovery["probabilities"][[a, b], e]
                return (
                    p * (1 - p) / discovery["counts"][a, e]
                    + q * (1 - q) / discovery["counts"][b, e]
                )

            chosen = max(candidates, key=uncertainty)
        blocks = split_response_witness(blocks, discovery["probabilities"], chosen)
        trace.append(
            dict(**chosen, eligible_pair_scores=len(candidates), states_after_split=len(blocks))
        )
    return blocks, trace


def response_frontier(
    discovery,
    estimation_support,
    initial,
    budgets,
    seed,
    *,
    min_cases=2000,
    min_users=50,
    minimum_probability_gap=0.0,
    exact_count_gate=False,
):
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(not isinstance(b, int) or b < 1 for b in budgets)
    ):
        raise ValueError("Distinct positive split budgets required")
    k = len(discovery["counts"])
    rows = [
        dict(name="unrefined", blocks=initial, trace=[], split_budget=0, random_fill_states=0),
        dict(
            name="full_parent",
            blocks=tuple((s,) for s in range(k)),
            trace=[],
            split_budget=0,
            random_fill_states=0,
        ),
    ]
    for budget in budgets:
        target, trace = select_partition(
            discovery,
            estimation_support,
            initial,
            budget,
            "targeted",
            seed,
            min_cases=min_cases,
            min_users=min_users,
            minimum_probability_gap=minimum_probability_gap,
            exact_count_gate=exact_count_gate,
        )
        rows.append(
            dict(
                name=f"targeted-b{budget}",
                blocks=target,
                trace=trace,
                split_budget=budget,
                random_fill_states=0,
            )
        )
        for policy in [
            "random_environment",
            "fixed_grid",
            "uncertainty",
            "random_witness",
            "random_capacity",
        ]:
            blocks, trace = (
                (initial, [])
                if policy == "random_capacity"
                else select_partition(
                    discovery,
                    estimation_support,
                    initial,
                    len(target) - len(initial),
                    policy,
                    seed + budget,
                    min_cases=min_cases,
                    min_users=min_users,
                    minimum_probability_gap=minimum_probability_gap,
                    exact_count_gate=exact_count_gate,
                )
            )
            fill = max(0, len(target) - len(blocks))
            blocks = random_capacity_refinement(blocks, k, len(target), seed + budget)
            rows.append(
                dict(
                    name=f"{policy}-b{budget}",
                    blocks=blocks,
                    trace=trace,
                    split_budget=budget,
                    random_fill_states=fill,
                )
            )
    return rows
