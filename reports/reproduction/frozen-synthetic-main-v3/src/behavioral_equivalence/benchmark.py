"""Common synthetic benchmark for six abstraction-method families."""

import hashlib
import json
import resource
import sys
import time
import numpy as np
from .core import Chain, Partition, approximate_partition, exact_partition, quotient, coupling_bound
from .clustering import feature_partition, predictive_features, random_partition
from .synthetic import GENERATOR_REVISION, generate


def probability_tables(chain: Chain, horizons: list[int]) -> dict[int, np.ndarray]:
    if not horizons or any(not isinstance(h, int) or h < 1 for h in horizons):
        raise ValueError("Benchmark horizons must be positive integers")
    labels = np.array(chain.labels)
    goals = np.column_stack([labels == "complete", labels == "disengage"])
    values = goals.astype(float)
    selected = set(horizons) | {h - 1 for h in horizons}
    tables = {0: values.copy()} if 0 in selected else {}
    for horizon in range(1, max(horizons) + 1):
        values = np.where(goals, 1, np.einsum("ij,jk->ik", chain.transitions, values))
        if horizon in selected:
            tables[horizon] = values.copy()
    return tables


def partition_properties(
    chain: Chain,
    blocks: Partition,
    horizons: list[int],
    truth: dict[int, np.ndarray],
    tolerance: float,
) -> tuple[Chain, float, list[dict]]:
    reduced, epsilon = quotient(chain, blocks)
    predicted = probability_tables(reduced, horizons)
    mapping = np.empty(chain.size, dtype=int)
    for i, block in enumerate(blocks):
        mapping[list(block)] = i
    active = np.array([s == "active" for s in chain.labels])
    q_active = np.array([s == "active" for s in reduced.labels])
    d = chain.transitions[:, active].sum(axis=1)
    d_hat = reduced.transitions[:, q_active].sum(axis=1)
    records = []
    for h in horizons:
        delta = coupling_bound(epsilon, h)
        pairs = [
            ("complete", truth[h][:, 0], predicted[h][:, 0]),
            ("disengage", truth[h][:, 1], predicted[h][:, 1]),
            (
                "survival",
                np.clip(1 - truth[h].sum(axis=1), 0, 1),
                np.clip(1 - predicted[h].sum(axis=1), 0, 1),
            ),
        ]
        for target, full, abstract in pairs:
            differences = np.abs(full - abstract[mapping])
            s = int(differences.argmax())
            records.append(
                {
                    "target": target,
                    "horizon": h,
                    "max_error": float(differences[s]),
                    "bound": delta,
                    "violated": bool(differences[s] > delta + tolerance),
                    "witness_state": s,
                    "concrete_value": float(full[s]),
                    "abstract_value": float(abstract[mapping[s]]),
                    "states_checked": chain.size,
                }
            )
        numerator = np.einsum("ij,j->i", chain.transitions[:, active], truth[h - 1][active, 0])
        numerator_hat = np.einsum(
            "ij,j->i", reduced.transitions[:, q_active], predicted[h - 1][q_active, 0]
        )
        eligible = active & (d > 0) & (d_hat[mapping] > 0)
        full = np.divide(numerator, d, out=np.zeros_like(d), where=d > 0)
        abstract = np.divide(numerator_hat, d_hat, out=np.zeros_like(d_hat), where=d_hat > 0)
        denominator_lower = d_hat[mapping] - epsilon
        conditional_bound = np.ones(chain.size)
        supported = denominator_lower > 0
        conditional_bound[supported] = np.minimum(
            1, (delta + epsilon) / denominator_lower[supported]
        )
        differences = np.abs(full - abstract[mapping])
        witnesses = np.flatnonzero(eligible)
        if not len(witnesses):
            raise ValueError("No supported conditional property origins")
        s = int(witnesses[np.argmax(differences[witnesses])])
        records.append(
            {
                "target": "complete_given_active_next",
                "horizon": h,
                "max_error": float(differences[s]),
                "bound": float(conditional_bound[s]),
                "violated": bool(
                    np.any(differences[eligible] > conditional_bound[eligible] + tolerance)
                ),
                "witness_state": s,
                "concrete_value": float(full[s]),
                "abstract_value": float(abstract[mapping[s]]),
                "states_checked": int(eligible.sum()),
            }
        )
    return reduced, epsilon, records


def evaluate_world(spec: dict, protocol: dict, verify=None) -> dict:
    world = generate(spec["family"], spec["size"], spec["seed"], spec["perturbation"])
    chain = world.chain
    horizons = protocol["horizons"]
    tolerance = protocol["numerical_tolerance"]
    truth = probability_tables(chain, horizons)
    embedding = np.repeat(np.asarray(world.features, dtype=float), 4, axis=0)

    def embedding_partition(k: int) -> Partition:
        normalized = (embedding - embedding.mean(axis=0)) / np.maximum(embedding.std(axis=0), 1e-12)
        return feature_partition(chain, normalized, k, spec["seed"])

    def prediction_partition(k: int) -> Partition:
        features = predictive_features(chain, tuple(protocol["predictive_feature_horizons"]))
        return feature_partition(chain, features, k, spec["seed"])

    original_verification = verify(chain, -1, "original") if verify else {"status": "not_requested"}
    variants = [("exact_rational", {}, lambda: exact_partition([chain]))]
    for epsilon in protocol["epsilon_grid"]:
        variants.append(
            (
                "approximate_bisimulation",
                {"epsilon_budget": epsilon},
                lambda e=epsilon: approximate_partition([chain], e),
            )
        )
    for divisor in protocol["state_divisors"]:
        k = chain.size // divisor
        variants.extend(
            [
                (
                    "feature_embedding_clustering",
                    {"requested_states": k},
                    lambda k=k: embedding_partition(k),
                ),
                (
                    "predictive_clustering",
                    {"requested_states": k},
                    lambda k=k: prediction_partition(k),
                ),
                (
                    "random_merging",
                    {"requested_states": k},
                    lambda k=k: random_partition(chain, k, spec["seed"]),
                ),
            ]
        )
        for epsilon in protocol["epsilon_grid"]:
            variants.append(
                (
                    "feature_initialized_verified_refinement",
                    {"requested_states": k, "epsilon_budget": epsilon},
                    lambda k=k, e=epsilon: approximate_partition(
                        [chain], e, embedding_partition(k)
                    ),
                )
            )
    methods = []
    for index, (name, parameters, factory) in enumerate(variants):
        start = time.monotonic()
        blocks = factory()
        partition_seconds = time.monotonic() - start
        start = time.monotonic()
        reduced, epsilon, properties = partition_properties(
            chain, blocks, horizons, truth, tolerance
        )
        property_seconds = time.monotonic() - start
        independent = verify(reduced, index, name) if verify else {"status": "not_requested"}
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        methods.append(
            {
                "method": name,
                "parameters": parameters,
                "states": reduced.size,
                "compression_ratio": chain.size / reduced.size,
                "epsilon_measured": epsilon,
                "transition_edges": int(np.count_nonzero(reduced.transitions)),
                "partition_seconds": partition_seconds,
                "property_seconds": property_seconds,
                "process_cumulative_peak_rss_kib": rss / 1024 if sys.platform == "darwin" else rss,
                "properties": properties,
                "independent_verification": independent,
                "partition": [list(b) for b in blocks],
            }
        )
    for method in methods:
        if "epsilon_budget" in method["parameters"]:
            if method["epsilon_measured"] > method["parameters"]["epsilon_budget"] + tolerance:
                raise AssertionError("Selected partition exceeds prescribed numerical tolerance")
    return {
        "spec": spec,
        "generator_revision": GENERATOR_REVISION,
        "mechanism_parameters": dict(world.parameters),
        "world_digest": chain.digest(),
        "ground_truth_partition": [list(b) for b in world.partition],
        "methods": methods,
        "method_variants": len(methods),
        "original_verification": original_verification,
        "bound_violations": sum(p["violated"] for m in methods for p in m["properties"]),
        "numeric_status": "floating checks with explicit tolerance; exact partition uses rational signatures",
        "memory_scope": "process cumulative high-water mark, not isolated per-method memory",
        "input_digest": hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(),
    }
