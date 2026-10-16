"""Paired seed-cluster analysis for the complete synthetic intervention design."""

from itertools import product
import math
import numpy as np
from .catalogue_search import SEARCH_METHODS

CONTROLS = (
    "unchanged",
    "targeted_witness",
    "random_witness_same_data",
    "random_capacity",
    "all_observed",
    "identity",
)


def summarize_interventions(manifest: dict, protocol: dict) -> dict:
    if (
        manifest["status"] != "complete"
        or manifest["failures"]
        or manifest["phase"] != "frozen multidimensional synthetic intervention"
        or manifest["source_digest"] != protocol["source_digest"]
        or manifest["protocol_digest"] != protocol["protocol_digest"]
    ):
        raise ValueError("Complete source-bound intervention campaign required")
    expected = set(
        product(protocol["families"], protocol["sizes"], protocol["seeds"], protocol["surfaces"])
    )
    if (
        protocol["expected_worlds"] != len(expected)
        or manifest["worlds"] != len(expected)
        or manifest["expected_worlds"] != len(expected)
    ):
        raise ValueError("World count differs from frozen design")
    worlds = {}
    metrics = (
        "mean_total_prediction_max_error",
        "mean_abstraction_max_error",
        "mean_environment_model_max_error",
        "mean_initial_completion_error",
    )
    for report in manifest["results"]:
        spec = report["spec"]
        key = tuple(spec[k] for k in ("family", "size", "seed", "surface"))
        if (
            key not in expected
            or key in worlds
            or report["source_digest"] != protocol["source_digest"]
            or report["protocol_digest"] != protocol["protocol_digest"]
        ):
            raise ValueError("Duplicate, off-design or source-mismatched world")
        means = {(r["method"], r["control"], r["horizon"]): r for r in report["means"]}
        expected_means = set(product(SEARCH_METHODS, CONTROLS, protocol["horizons"]))
        if set(means) != expected_means or len(report["means"]) != len(expected_means):
            raise ValueError("Missing or duplicate method/control/horizon means")
        for row in means.values():
            if row["heldout_environments"] != 16 or any(
                not math.isfinite(row[m]) or not 0 <= row[m] <= 1 for m in metrics
            ):
                raise ValueError(
                    "Finite probability-error means over sixteen environments required"
                )
            if key[3] == "negative_control" and row["mean_total_prediction_max_error"] > 1e-12:
                raise ValueError("Structural negative control changed observable behavior")
        for method in SEARCH_METHODS:
            for h in protocol["horizons"]:
                if (
                    means[(method, "targeted_witness", h)]["states"]
                    != means[(method, "random_capacity", h)]["states"]
                ):
                    raise ValueError("Random-capacity comparator is not capacity matched")
                if (
                    means[(method, "identity", h)]["states"] != spec["size"]
                    or means[(method, "unchanged", h)]["states"] != report["original_states"]
                ):
                    raise ValueError("Identity or original comparator has wrong state count")
        searches = {r["method"]: r for r in report["searches"]}
        if set(searches) != set(SEARCH_METHODS) or len(report["searches"]) != len(SEARCH_METHODS):
            raise ValueError("Missing or duplicate search methods")
        for search in searches.values():
            if (
                len(search["indices"]) != protocol["environment_budget"]
                or len(set(search["indices"])) != len(search["indices"])
                or search["environment_evaluations"] != protocol["environment_budget"]
                or search["winner"] not in search["indices"]
            ):
                raise ValueError("Actual query budget differs from design")
        if report["verified_estimated_model_certificates"] != 16 * len(SEARCH_METHODS):
            raise ValueError("Missing targeted estimated-model certificates")
        worlds[key] = (report, means)
    if set(worlds) != expected or manifest["verified_estimated_model_certificates"] != len(
        expected
    ) * 16 * len(SEARCH_METHODS):
        raise ValueError("Incomplete world or certificate matrix")
    included = [key for key in sorted(worlds) if key[3] != "negative_control"]
    seeds = sorted(protocol["seeds"])
    if len(seeds) < 2 or not included:
        raise ValueError("At least two seed clusters and a response-changing surface required")
    rng = np.random.default_rng(20261015)
    bootstrap = rng.integers(0, len(seeds), (5000, len(seeds)))
    comparisons = [
        ("bundled_primary", ("rbf_ucb", "targeted_witness"), ("random", "random_capacity"), 0.05)
    ] + [
        (
            "same_data_and_capacity",
            (method, "targeted_witness"),
            (method, "random_capacity"),
            0.05 / len(SEARCH_METHODS),
        )
        for method in SEARCH_METHODS
    ]
    results = []
    h = protocol["primary_horizon"]
    metric = "mean_total_prediction_max_error"
    for scope, candidate, reference, alpha in comparisons:
        effects = []
        for seed in seeds:
            pairs = [
                (worlds[key][1][(*candidate, h)], worlds[key][1][(*reference, h)])
                for key in included
                if key[2] == seed
            ]
            effects.append(
                dict(
                    seed=seed,
                    world_cells=len(pairs),
                    paired_mean_difference=float(
                        np.mean([a[metric] - b[metric] for a, b in pairs])
                    ),
                    candidate_mean_states=float(np.mean([a["states"] for a, _ in pairs])),
                    reference_mean_states=float(np.mean([b["states"] for _, b in pairs])),
                )
            )
        values = np.array([r["paired_mean_difference"] for r in effects])
        lo, hi = np.quantile(values[bootstrap].mean(axis=1), [alpha / 2, 1 - alpha / 2])
        results.append(
            dict(
                scope=scope,
                candidate=list(candidate),
                reference=list(reference),
                horizon=h,
                metric=metric,
                paired_mean_difference=float(values.mean()),
                seed_cluster_bootstrap_interval=[float(lo), float(hi)],
                interval_level=1 - alpha,
                seed_effects=effects,
                independent_seed_clusters=len(seeds),
                interpretation="Negative differences favor the candidate. Bundled search comparison can differ in capacity; only within-method targeted/random-capacity comparisons isolate matched-capacity refinement.",
            )
        )
    negative = [
        r
        for key, (report, _) in worlds.items()
        if key[3] == "negative_control"
        for r in report["means"]
    ]
    return dict(
        schema_version="intervention-seed-cluster-analysis-v1",
        status="complete",
        campaign_id=protocol["campaign_id"],
        protocol_digest=protocol["protocol_digest"],
        response_worlds=len(worlds),
        primary_response_changing_worlds=len(included),
        negative_control_worlds=len(worlds) - len(included),
        seed_clusters=len(seeds),
        bootstrap_replications=5000,
        bootstrap_seed=20261015,
        comparisons=results,
        negative_control_max_mean_prediction_error=max((r[metric] for r in negative), default=None),
        scope="Conditional on the fixed synthetic generator and held-out catalogue. Ten planned shared seed clusters, not states, horizons, coordinates or surfaces, determine uncertainty. Limited-cluster bootstrap summaries are not distribution-free certificates. No human generalization inference.",
    )
