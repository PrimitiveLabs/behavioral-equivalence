"""Paired secondary ablation summaries over fixed-world calibration datasets."""

from itertools import product
import math
from .coverage_analysis import coverage_interval
from .repeated_analysis import METHODS, SUPPORTS, summarize_repetitions


def summarize_nominal(original: dict, nominal: dict, protocol: dict, source_digest: str) -> dict:
    """Validate complete pairing before computing conditional Monte Carlo summaries."""
    validated = summarize_repetitions(original, protocol)
    if (
        nominal["status"] != "complete"
        or nominal["failures"]
        or nominal["phase"] != "secondary paired nominal ablation"
        or nominal["source_digest"] != source_digest
    ):
        raise ValueError("Complete source-bound nominal ablation required")
    worlds = set(
        product(protocol["families"], protocol["sizes"], protocol["seeds"], protocol["regimes"])
    )
    expected = set(product(worlds, range(protocol["repetitions"]), SUPPORTS))
    original_chunks = {
        tuple(c["spec"][k] for k in ("family", "size", "seed", "regime"))
        + (c["spec"]["first_repetition"],): c
        for c in original["chunks"]
    }
    reference = {}
    for c in original["chunks"]:
        world = tuple(c["spec"][k] for k in ("family", "size", "seed", "regime"))
        for row in c["records"]:
            reference[(world, row["repetition"], row["support"], row["method"])] = row
    observed = {}
    seen_chunks = set()
    for chunk in nominal["chunks"]:
        spec = chunk["spec"]
        world = tuple(spec[k] for k in ("family", "size", "seed", "regime"))
        chunk_key = world + (spec["first_repetition"],)
        if chunk_key in seen_chunks:
            raise ValueError("Duplicate nominal chunk")
        seen_chunks.add(chunk_key)
        parent = original_chunks.get(chunk_key)
        if (
            parent is None
            or spec != parent["spec"]
            or chunk["source_digest"] != source_digest
            or chunk["original_chunk_digest"] != parent["chunk_digest"]
        ):
            raise ValueError("Nominal chunk differs from original design")
        for row in chunk["records"]:
            key = (world, row["repetition"], row["support"])
            if (
                key not in expected
                or key in observed
                or row["method"] != "empirical_point"
                or not spec["first_repetition"] <= row["repetition"] < spec["last_repetition"]
            ):
                raise ValueError("Duplicate or off-design nominal observation")
            if not isinstance(row["kernel_membership"], bool):
                raise ValueError("Boolean kernel membership required")
            properties = row["properties"]
            expected_properties = set(
                product(("complete", "disengage", "survival"), protocol["horizons"])
            )
            if (
                len(properties) != len(expected_properties)
                or {(p["property"], p["horizon"]) for p in properties} != expected_properties
            ):
                raise ValueError("Incomplete nominal property suite")
            for p in properties:
                counts = [
                    p[k]
                    for k in (
                        "enclosure_containment_count",
                        "proven_noncoverage_count",
                        "unresolved_count",
                    )
                ]
                if (
                    any(not isinstance(n, int) or isinstance(n, bool) or n < 0 for n in counts)
                    or sum(counts) != world[1]
                ):
                    raise ValueError("Invalid probability coverage counts")
            widths = row["completion_widths"]
            if sorted(w["horizon"] for w in widths) != sorted(protocol["horizons"]) or any(
                not math.isfinite(w[k]) or not 0 <= w[k] <= 1
                for w in widths
                for k in ("mean_active_width", "initial_weighted_width")
            ):
                raise ValueError("Invalid nominal width data")
            observed[key] = row
    if (
        set(observed) != expected
        or seen_chunks != set(original_chunks)
        or nominal["point_evaluations"] != len(expected)
    ):
        raise ValueError("Missing nominal calibration evaluations")
    comparisons = (
        len(SUPPORTS) * len(protocol["regimes"]) * (2 + len(METHODS) * len(protocol["horizons"]))
    )
    coverage, pairs = [], []
    for support, regime in product(SUPPORTS, protocol["regimes"]):
        keys = sorted(k for k in expected if k[2] == support and k[0][3] == regime)
        n = len(keys)
        for metric in ("kernel_membership", "all_probability_enclosures_contained"):
            successes = sum(
                bool(observed[k]["kernel_membership"])
                if metric == "kernel_membership"
                else all(
                    p["enclosure_containment_count"] == k[0][1] for p in observed[k]["properties"]
                )
                for k in keys
            )
            coverage.append(
                dict(
                    support=support,
                    regime=regime,
                    metric=metric,
                    successes=successes,
                    independent_datasets=n,
                    simultaneous_ci=coverage_interval(successes, n, 0.05, comparisons),
                )
            )
        for method, horizon in product(METHODS, protocol["horizons"]):
            differences, weighted = [], []
            for key in keys:
                point = next(
                    w for w in observed[key]["completion_widths"] if w["horizon"] == horizon
                )
                ref = next(
                    w
                    for w in reference[(*key, method)]["completion_widths"]
                    if w["horizon"] == horizon
                )
                differences.append(point["mean_active_width"] - ref["mean_active_width"])
                weighted.append(point["initial_weighted_width"] - ref["initial_weighted_width"])
            mean = sum(differences) / n
            radius = 2 * math.sqrt(math.log(2 * comparisons / 0.05) / (2 * n))
            pairs.append(
                dict(
                    support=support,
                    regime=regime,
                    reference_method=method,
                    horizon=horizon,
                    independent_paired_datasets=n,
                    point_minus_reference_mean_active_width=mean,
                    simultaneous_ci=[
                        max(-1, math.nextafter(mean - radius, -math.inf)),
                        min(1, math.nextafter(mean + radius, math.inf)),
                    ],
                    point_minus_reference_initial_weighted_width=sum(weighted) / n,
                )
            )
    return dict(
        schema_version="nominal-ablation-summary-v1",
        status="complete",
        source_digest=source_digest,
        original_protocol_digest=protocol["protocol_digest"],
        independent_datasets=validated["independent_datasets"],
        point_support_evaluations=len(observed),
        family_confidence=0.95,
        simultaneous_intervals=comparisons,
        point_coverage=coverage,
        paired_completion_widths=pairs,
        statistical_coverage_claim="No confidence guarantee for the empirical point estimator",
        interval_scope="Numerically evaluated Hoeffding/Bonferroni Monte Carlo intervals conditional on fixed worlds. Paired width differences lie in [-1,1]. Initial-weighted differences are descriptive only.",
        evidence_scope="Complete design and pairing checked. Original certificates were separately replayed; this structural summary does not replay nominal arithmetic.",
    )
