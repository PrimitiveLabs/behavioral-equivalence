import numpy as np
from behavioral_equivalence.benchmark import (
    probability_tables,
    partition_properties,
    evaluate_world,
)
from behavioral_equivalence.core import reachability, exact_partition
from behavioral_equivalence.synthetic import generate


def test_common_properties_against_independent_single_property_calculations():
    world = generate("belief", 16, 81)
    horizons = [5, 10]
    table = probability_tables(world.chain, horizons)
    for h in horizons:
        np.testing.assert_allclose(
            table[h][:, 0], reachability(world.chain, "complete", h), atol=1e-14
        )
    reduced, epsilon, properties = partition_properties(
        world.chain, exact_partition([world.chain]), horizons, table, 1e-10
    )
    assert epsilon == 0 and reduced.size <= 4
    assert len(properties) == 8
    assert not any(p["violated"] for p in properties)


def test_all_six_method_families_use_the_same_properties():
    protocol = {
        "horizons": [5],
        "epsilon_grid": [0.03],
        "state_divisors": [4],
        "predictive_feature_horizons": [1, 5],
        "numerical_tolerance": 1e-10,
    }
    result = evaluate_world(
        {"family": "memory", "size": 16, "seed": 90, "perturbation": 0.01}, protocol
    )
    assert len({m["method"] for m in result["methods"]}) == 6
    assert result["bound_violations"] == 0
    assert all(len(m["properties"]) == 4 for m in result["methods"])
