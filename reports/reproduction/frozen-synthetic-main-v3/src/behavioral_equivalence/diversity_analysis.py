"""Complete paired-world, seed-cluster summaries for exploratory diversity tests."""

from itertools import product
import math
import numpy as np
from .environment_diversity import diversity_catalogues

SPANS = ("narrow", "medium", "broad")
CONTROLS = ("unchanged", "targeted", "random_capacity", "all_observed", "identity")
METRICS = (
    "initial_completion_error",
    "total_prediction_max_error",
    "environment_model_max_error",
    "abstraction_max_error",
)


def diversity_world_means(report):
    if report["status"] != "complete":
        raise ValueError("Complete world required")
    expected = set(product(SPANS, CONTROLS, range(16)))
    rows = {(r["span"], r["control"], r["heldout_index"]): r for r in report["records"]}
    if set(rows) != expected or len(rows) != len(report["records"]):
        raise ValueError("Complete unique span/control/heldout grid required")
    for row in rows.values():
        if (
            row["horizon"] != 10
            or not isinstance(row["states"], int)
            or not 1 <= row["states"] <= report["size"]
            or any(not math.isfinite(row[m]) or not 0 <= row[m] <= 1 for m in METRICS)
        ):
            raise ValueError("Valid fixed-horizon probability errors and state counts required")
        if report["surface"] == "negative_control" and row["total_prediction_max_error"] > 1e-12:
            raise ValueError("Invariant negative control changed predictions")
    keys = {(r["span"], r["heldout_index"]) for r in report["certificates"]}
    if keys != set(product(SPANS, range(16))) or len(report["certificates"]) != 48:
        raise ValueError("Complete targeted certificate bindings required")
    pools, _ = diversity_catalogues()
    searches = {s["span"]: s for s in report["searches"]}
    if set(searches) != set(SPANS) or len(report["searches"]) != 3:
        raise ValueError("One query set per span required")
    for span, search in searches.items():
        points = np.asarray(search["selected_points"])
        if (
            search["query_count"] != 8
            or points.shape != (8, 2)
            or len(np.unique(points, axis=0)) != 8
            or any(not np.any(np.all(pools[span] == point, axis=1)) for point in points)
        ):
            raise ValueError("Eight distinct queries from frozen span pool required")
    means = []
    for span, control in product(SPANS, CONTROLS):
        group = [rows[(span, control, i)] for i in range(16)]
        states = {r["states"] for r in group}
        if len(states) != 1:
            raise ValueError("Partition must be fixed across heldout points")
        if control == "identity" and next(iter(states)) != report["size"]:
            raise ValueError("Identity control must preserve all states")
        if control == "random_capacity" and any(
            r["states"] != rows[(span, "targeted", r["heldout_index"])]["states"] for r in group
        ):
            raise ValueError("Random control must match targeted capacity")
        means.append(
            dict(
                span=span,
                control=control,
                states=next(iter(states)),
                **{"mean_" + m: float(np.mean([r[m] for r in group])) for m in METRICS},
            )
        )
    return means


def summarize_diversity(summaries, protocol, source_digest):
    expected = set(
        product(protocol["families"], protocol["sizes"], protocol["seeds"], protocol["surfaces"])
    )
    if protocol["expected_worlds"] != len(expected) or len(expected) != len(summaries):
        raise ValueError("Frozen complete world matrix required")
    worlds = {}
    for row in summaries:
        key = tuple(row["spec"][k] for k in ["family", "size", "seed", "surface"])
        if (
            row.get("status") != "complete"
            or key in worlds
            or key not in expected
            or row["source_digest"] != source_digest
            or row["certificate_replay_count"] != 48
        ):
            raise ValueError("Unique source-bound fully replayed worlds required")
        means = {(r["span"], r["control"]): r for r in row["means"]}
        if len(row["means"]) != 15 or set(means) != set(product(SPANS, CONTROLS)):
            raise ValueError("Complete world means required")
        for r in means.values():
            if any(
                not math.isfinite(r["mean_" + m]) or not 0 <= r["mean_" + m] <= 1 for m in METRICS
            ):
                raise ValueError("Finite probability-error means required")
        for span in SPANS:
            for control in CONTROLS:
                states = means[(span, control)]["states"]
                if not isinstance(states, int) or not 1 <= states <= key[1]:
                    raise ValueError("Feasible latent capacity required")
            if means[(span, "identity")]["states"] != key[1]:
                raise ValueError("Identity capacity differs from generating model")
            if means[(span, "targeted")]["states"] != means[(span, "random_capacity")]["states"]:
                raise ValueError("Refinement comparison is not capacity matched")
        worlds[key] = means
    seeds = sorted(protocol["seeds"])
    changing = [k for k in sorted(worlds) if k[3] != "negative_control"]
    if len(seeds) < 2 or not changing:
        raise ValueError("Multiple seed clusters and changing worlds required")
    rng = np.random.default_rng(20261015)
    bootstrap = rng.integers(0, len(seeds), (5000, len(seeds)))
    contrasts = (
        [("span", ("broad", c), ("narrow", c)) for c in CONTROLS]
        + [("span", ("medium", c), ("narrow", c)) for c in CONTROLS]
        + [("capacity_matched_refinement", (s, "targeted"), (s, "random_capacity")) for s in SPANS]
    )
    results = []
    for scope, candidate, reference in contrasts:
        for metric in METRICS:
            effects = []
            for seed in seeds:
                values = [
                    worlds[k][candidate]["mean_" + metric] - worlds[k][reference]["mean_" + metric]
                    for k in changing
                    if k[2] == seed
                ]
                effects.append(
                    dict(seed=seed, paired_worlds=len(values), difference=float(np.mean(values)))
                )
            values = np.array([r["difference"] for r in effects])
            lo, hi = np.quantile(values[bootstrap].mean(axis=1), [0.025, 0.975])
            results.append(
                dict(
                    scope=scope,
                    candidate=list(candidate),
                    reference=list(reference),
                    metric=metric,
                    paired_mean_difference=float(values.mean()),
                    descriptive_seed_cluster_ci95=[float(lo), float(hi)],
                    seed_effects=effects,
                )
            )
    negatives = [
        r["mean_total_prediction_max_error"]
        for k, means in worlds.items()
        if k[3] == "negative_control"
        for r in means.values()
    ]
    if any(v > 1e-12 for v in negatives):
        raise ValueError("Invariant negative control failed")
    return dict(
        status="complete",
        phase="exploratory",
        source_digest=source_digest,
        worlds=len(worlds),
        response_changing_worlds=len(changing),
        negative_control_worlds=len(worlds) - len(changing),
        seed_clusters=len(seeds),
        bootstrap_replications=5000,
        bootstrap_seed=20261015,
        comparisons=results,
        negative_control_max_mean_prediction_error=max(negatives, default=None),
        certificates_replayed=48 * len(worlds),
        scope="Conditional exploratory summaries for the frozen generator and fixed heldout catalogue; limited-cluster bootstrap intervals are descriptive, not confirmatory or distribution-free. Cross-span targeted comparisons can change capacity; identity isolates environment-learning error. Within-span targeted/random comparisons match capacity. No human inference.",
    )


def replay_diversity_certificates(report, arrays):
    """Reconstruct saved point kernels and replay every span/context certificate."""
    from .core import Chain
    from .integer_bounds import verify_abstraction

    keys = {(r["span"], r["heldout_index"]) for r in report["certificates"]}
    if keys != set(product(SPANS, range(16))) or len(report["certificates"]) != 48:
        raise ValueError("Complete unique persisted certificate matrix required")
    replayed = 0
    for record in report["certificates"]:
        span, index = record["span"], record["heldout_index"]
        blocks = tuple(tuple(b) for b in report["partitions"][span]["targeted"])
        parent = Chain(
            arrays[f"{span}-e{index}-parent"],
            tuple(report["labels"]),
            np.asarray(report["initial"]),
        )
        abstract = Chain(
            arrays[f"{span}-e{index}-targeted"],
            tuple(parent.labels[b[0]] for b in blocks),
            np.array([parent.initial[list(b)].sum() for b in blocks]),
        )
        verify_abstraction(parent, abstract, record["certificate"])
        replayed += 1
    return replayed
