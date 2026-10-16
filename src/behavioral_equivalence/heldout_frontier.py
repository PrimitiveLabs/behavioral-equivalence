"""Neutral-certified, capacity-matched refinement for response-group holdouts."""

from .neutral_response import (
    certified_neutral_partition,
    repair_neutral_partition,
    random_neutral_capacity,
    response_gap_witness,
)
from .heldout_response import query_only_table
from .response_frontier import select_partition


def heldout_refinement_frontier(
    neutral,
    discovery,
    estimation,
    query_groups,
    budgets,
    seed,
    *,
    epsilon=0.01,
    min_cases=2000,
    min_users=50,
):
    if (
        not budgets
        or len(set(budgets)) != len(budgets)
        or any(not isinstance(b, int) or b < 1 for b in budgets)
    ):
        raise ValueError("Distinct positive refinement budgets required")
    initial, initial_certificate = certified_neutral_partition(neutral, epsilon)
    visible_discovery = query_only_table(discovery, query_groups)
    visible_estimation = query_only_table(estimation, query_groups)
    full = tuple((s,) for s in range(len(neutral)))
    _, full_certificate = repair_neutral_partition(neutral, full, epsilon)
    rows = [
        dict(
            name="unrefined",
            blocks=initial,
            trace=[],
            split_budget=0,
            neutral_repair_states=0,
            capacity_fill_states=0,
            neutral_certificate=initial_certificate,
        ),
        dict(
            name="full_parent",
            blocks=full,
            trace=[],
            split_budget=0,
            neutral_repair_states=0,
            capacity_fill_states=0,
            neutral_certificate=full_certificate,
        ),
    ]
    policies = ["targeted", "random_environment", "fixed_grid", "uncertainty", "random_witness"]
    for budget in budgets:
        candidates = []
        for index, policy in enumerate(policies):
            blocks, trace = select_partition(
                visible_discovery,
                visible_estimation,
                initial,
                budget,
                policy,
                seed + index,
                min_cases=min_cases,
                min_users=min_users,
                minimum_probability_gap=2 * epsilon,
                exact_count_gate=True,
            )
            certificates = [
                response_gap_witness(
                    visible_discovery["counts"],
                    visible_discovery["positives"],
                    w["states"],
                    w["environment"],
                    epsilon,
                )
                for w in trace
            ]
            if any(not c["exceeds_budget"] for c in certificates):
                raise ArithmeticError("Selected witness does not exclude shared-row precision")
            repaired, certificate = repair_neutral_partition(neutral, blocks, epsilon)
            candidates.append(
                dict(
                    name=f"{policy}-b{budget}",
                    blocks=repaired,
                    trace=trace,
                    witness_certificates=certificates,
                    split_budget=budget,
                    neutral_repair_states=len(repaired) - len(blocks),
                    neutral_certificate=certificate,
                )
            )
        common = max(len(c["blocks"]) for c in candidates)
        for index, candidate in enumerate(candidates):
            before = len(candidate["blocks"])
            blocks, certificate = random_neutral_capacity(
                neutral, candidate["blocks"], common, seed + 1000 * budget + index, epsilon
            )
            candidate.update(
                blocks=blocks, capacity_fill_states=common - before, neutral_certificate=certificate
            )
            rows.append(candidate)
        blocks, certificate = random_neutral_capacity(
            neutral, initial, common, seed + 1000 * budget + len(policies), epsilon
        )
        rows.append(
            dict(
                name=f"random_neutral_capacity-b{budget}",
                blocks=blocks,
                trace=[],
                witness_certificates=[],
                split_budget=budget,
                neutral_repair_states=0,
                capacity_fill_states=common - len(initial),
                neutral_certificate=certificate,
            )
        )
    return rows
