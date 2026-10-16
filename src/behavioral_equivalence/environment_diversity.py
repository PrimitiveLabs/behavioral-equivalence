"""Exploratory fixed-query environment-span ablation with known synthetic worlds."""

from fractions import Fraction
import time
import numpy as np
from .synthetic import generate
from .core import exact_partition, approximate_partition, quotient
from .refinement import outcome_distribution
from .intervention_space import environment_chain, environment_catalogues, interpolate_environment
from .catalogue_search import search_catalogue, maximum_within_block_js, random_refinement_to_count
from .integer_bounds import certify_abstraction, verify_abstraction


def diversity_catalogues():
    points, heldout = environment_catalogues()
    near = points[(np.abs(points[:, 0]) <= 1 / 7 + 1e-12) & (np.abs(points[:, 1]) <= 3 / 7 + 1e-12)]
    medium = points[(np.abs(points) <= 3 / 7 + 1e-12).all(axis=1)]
    if (len(near), len(medium), len(points)) != (8, 16, 64):
        raise ValueError("Frozen8x8catalogue geometry changed")
    return {"narrow": near, "medium": medium, "broad": points}, heldout


def run_diversity_world(family: str, size: int, seed: int, surface: str):
    started = time.monotonic()
    world = generate(family, size, seed)
    initial = exact_partition([world.chain])
    catalogues, heldout = diversity_catalogues()
    horizon = 10
    hidden = [environment_chain(world, tuple(point), surface) for point in heldout]
    truth = [outcome_distribution(c, horizon) for c in hidden]
    records = []
    certificates = []
    searches = []
    arrays = {}
    partitions = {}
    for span, points in catalogues.items():
        observed = {}

        def evaluate(index):
            chain = environment_chain(world, tuple(points[index]), surface)
            observed[index] = chain
            return maximum_within_block_js(outcome_distribution(chain, horizon), initial)[0]

        search = search_catalogue(points, evaluate, "maximin_grid", 8, seed)
        winner = search["winner"]
        selected = points[search["indices"]]
        targeted = approximate_partition([world.chain, observed[winner]], 0.01, initial)
        controls = {
            "unchanged": initial,
            "targeted": targeted,
            "random_capacity": random_refinement_to_count(initial, len(targeted), seed + 902),
            "all_observed": approximate_partition(
                [world.chain] + [observed[i] for i in search["indices"]], 0.01, initial
            ),
            "identity": tuple((s,) for s in range(size)),
        }
        partitions[span] = {name: [list(b) for b in blocks] for name, blocks in controls.items()}
        searches.append(
            dict(
                span=span,
                query_count=len(observed),
                selected_points=selected.tolist(),
                candidate_pool_size=len(points),
                winner_point=points[winner].tolist(),
                diameter=float(
                    np.sqrt(((selected[:, None] - selected[None, :]) ** 2).sum(axis=2)).max()
                ),
            )
        )
        x = np.vstack([np.zeros((1, 2)), selected])
        models = [world.chain] + [observed[i] for i in search["indices"]]
        for index, point in enumerate(heldout):
            estimate = interpolate_environment(x, models, tuple(point))
            full = outcome_distribution(estimate, horizon)
            arrays[f"{span}-e{index}-parent"] = estimate.transitions
            for control, blocks in controls.items():
                abstract, _ = quotient(estimate, blocks)
                mapping = np.empty(size, dtype=np.int64)
                for b, states in enumerate(blocks):
                    mapping[list(states)] = b
                prediction = outcome_distribution(abstract, horizon)[mapping]
                abstraction_error = float(np.abs(full - prediction).max())
                if control == "targeted":
                    certificate = certify_abstraction(
                        estimate, abstract, blocks, [horizon], bits=48
                    )
                    verify_abstraction(estimate, abstract, certificate)
                    value = certificate["deltas"][str(horizon)]
                    bound = Fraction(
                        int(value["numerator_hex"], 16), int(value["denominator_hex"], 16)
                    )
                    if abstraction_error > float(bound) + 1e-12:
                        raise AssertionError(
                            "Abstraction error exceeds exact-normalized model allowance"
                        )
                    certificates.append(
                        dict(span=span, heldout_index=index, certificate=certificate)
                    )
                    arrays[f"{span}-e{index}-targeted"] = abstract.transitions
                records.append(
                    dict(
                        span=span,
                        control=control,
                        heldout_index=index,
                        horizon=horizon,
                        states=len(blocks),
                        environment_model_max_error=float(np.abs(truth[index] - full).max()),
                        abstraction_max_error=abstraction_error,
                        total_prediction_max_error=float(np.abs(truth[index] - prediction).max()),
                        initial_completion_error=float(
                            abs(world.chain.initial @ (truth[index][:, 0] - prediction[:, 0]))
                        ),
                    )
                )
    report = dict(
        status="complete",
        family=family,
        size=size,
        seed=seed,
        surface=surface,
        searches=searches,
        partitions=partitions,
        records=records,
        certificates=certificates,
        labels=world.chain.labels,
        initial=world.chain.initial.tolist(),
        wall_seconds=time.monotonic() - started,
        scope="Exploratory after original synthetic intervention results. Equal budgets of8nonneutral queries, paired world/heldout points; targeted certificates cover estimated kernels, not environment-learning error or humans.",
    )
    return report, arrays
