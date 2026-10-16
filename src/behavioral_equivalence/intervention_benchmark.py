"""Synthetic environment learning and matched-capacity abstraction controls."""

from fractions import Fraction
import time
import numpy as np
from .core import exact_partition, approximate_partition, quotient
from .synthetic import generate
from .refinement import outcome_distribution
from .intervention_space import environment_chain, environment_catalogues, interpolate_environment
from .catalogue_search import (
    search_catalogue,
    maximum_within_block_js,
    random_refinement_to_count,
    SEARCH_METHODS,
)
from .integer_bounds import certify_abstraction, verify_abstraction


def run_intervention_world(
    family: str,
    size: int,
    seed: int,
    surface: str,
    budget: int = 8,
    horizons: tuple[int, ...] = (5, 10, 25, 50),
) -> tuple[dict, dict]:
    started = time.monotonic()
    world = generate(family, size, seed)
    original = exact_partition([world.chain])
    points, heldout = environment_catalogues()
    # Only the scoring loop below accesses these ground-truth held-out kernels.
    hidden = [environment_chain(world, tuple(point), surface) for point in heldout]
    truth = {h: [outcome_distribution(chain, h) for chain in hidden] for h in horizons}
    records, searches, certificates, retained = [], [], [], []
    for method in SEARCH_METHODS:
        observed = {}
        witnesses = {}

        def evaluate(index):
            chain = environment_chain(world, tuple(points[index]), surface)
            observed[index] = chain
            score, pair = maximum_within_block_js(outcome_distribution(chain, 10), original)
            witnesses[index] = pair
            return score

        search = search_catalogue(points, evaluate, method, budget, seed)
        winner = search["winner"]
        random_winner = int(np.random.default_rng([seed, 301]).choice(search["indices"]))
        chosen = approximate_partition([world.chain, observed[winner]], 0.01, original)
        random_witness = approximate_partition(
            [world.chain, observed[random_winner]], 0.01, original
        )
        matched = random_refinement_to_count(original, len(chosen), seed + 902)
        all_observed = approximate_partition(
            [world.chain] + [observed[i] for i in search["indices"]], 0.01, original
        )
        partitions = {
            "unchanged": original,
            "targeted_witness": chosen,
            "random_witness_same_data": random_witness,
            "random_capacity": matched,
            "all_observed": all_observed,
            "identity": tuple((i,) for i in range(size)),
        }
        observed_points = np.vstack([np.zeros((1, 2)), points[search["indices"]]])
        observed_models = [world.chain] + [observed[i] for i in search["indices"]]
        searches.append(
            dict(
                method=method,
                **search,
                witness_states=list(witnesses[winner]),
                witness_point=points[winner].tolist(),
                random_witness_index=random_winner,
                partition_maps={
                    name: [list(b) for b in blocks] for name, blocks in partitions.items()
                },
            )
        )
        for index, point in enumerate(heldout):
            estimate = interpolate_environment(observed_points, observed_models, tuple(point))
            full = {h: outcome_distribution(estimate, h) for h in horizons}
            retained.append(
                dict(
                    method=method,
                    heldout_index=index,
                    estimated_transition=estimate.transitions.tolist(),
                    true_transition=hidden[index].transitions.tolist(),
                )
            )
            for name, blocks in partitions.items():
                reduced, radius = quotient(estimate, blocks)
                mapping = np.array(
                    [next(i for i, block in enumerate(blocks) if s in block) for s in range(size)]
                )
                if name == "targeted_witness":
                    certificate = certify_abstraction(
                        estimate, reduced, blocks, list(horizons), bits=48
                    )
                    verify_abstraction(estimate, reduced, certificate)
                    certificates.append(
                        dict(
                            method=method,
                            heldout_index=index,
                            certificate=certificate,
                            abstract_transition=reduced.transitions.tolist(),
                        )
                    )
                for h in horizons:
                    predicted = outcome_distribution(reduced, h)[mapping]
                    environment_error = float(np.abs(truth[h][index] - full[h]).max())
                    abstraction_error = float(np.abs(full[h] - predicted).max())
                    overall_error = float(np.abs(truth[h][index] - predicted).max())
                    row = dict(
                        method=method,
                        control=name,
                        heldout_index=index,
                        horizon=h,
                        states=len(blocks),
                        environment_model_max_error=environment_error,
                        abstraction_max_error=abstraction_error,
                        total_prediction_max_error=overall_error,
                        initial_completion_error=float(
                            abs(
                                np.sum(
                                    world.chain.initial * (truth[h][index][:, 0] - predicted[:, 0])
                                )
                            )
                        ),
                        nominal_radius_for_display=radius,
                    )
                    if name == "targeted_witness":
                        delta = certificate["deltas"][str(h)]
                        exact_delta = Fraction(
                            int(delta["numerator_hex"], 16), int(delta["denominator_hex"], 16)
                        )
                        row["certified_estimated_model_delta_for_display"] = float(exact_delta)
                        if abstraction_error > float(exact_delta) + 1e-12:
                            raise AssertionError(
                                "Floating implementation contradicts integer abstraction certificate"
                            )
                    records.append(row)
        retained.append(
            dict(
                method=method,
                search_indices=search["indices"],
                observed_points=observed_points.tolist(),
                observed_transitions=[c.transitions.tolist() for c in observed_models],
            )
        )
    means = []
    for method in SEARCH_METHODS:
        for control in partitions:
            for h in horizons:
                rows = [
                    r
                    for r in records
                    if r["method"] == method and r["control"] == control and r["horizon"] == h
                ]
                means.append(
                    dict(
                        method=method,
                        control=control,
                        horizon=h,
                        states=rows[0]["states"],
                        heldout_environments=len(rows),
                        mean_total_prediction_max_error=float(
                            np.mean([r["total_prediction_max_error"] for r in rows])
                        ),
                        mean_abstraction_max_error=float(
                            np.mean([r["abstraction_max_error"] for r in rows])
                        ),
                        mean_environment_model_max_error=float(
                            np.mean([r["environment_model_max_error"] for r in rows])
                        ),
                        mean_initial_completion_error=float(
                            np.mean([r["initial_completion_error"] for r in rows])
                        ),
                    )
                )
    report = dict(
        status="complete",
        spec=dict(family=family, size=size, seed=seed, surface=surface),
        searches=searches,
        records=records,
        means=means,
        original_states=len(original),
        world_digest=world.chain.digest(),
        labels=list(world.chain.labels),
        initial=world.chain.initial.tolist(),
        search_catalogue=points.tolist(),
        heldout_catalogue=heldout.tolist(),
        environment_budget=budget,
        verified_estimated_model_certificates=len(certificates),
        wall_seconds=time.monotonic() - started,
        scope="Known synthetic kernel queries with a shared convex response-surface learner. Held-out true kernels enter scoring only. Formal certificates compare estimated full and reduced kernels, not the unknown generating environment response. No human inference.",
        controls_scope="All controls within a search method receive the same queried kernels. Random-capacity control matches targeted state count. Identity isolates environment-model error. All-observed refinement uses the full available query set; random-witness control uses one uniformly selected observed environment.",
    )
    return report, dict(certificates=certificates, retained_models=retained)
