"""Conservative Monte Carlo uncertainty over transition datasets, not states."""

import math


def coverage_interval(
    successes: int, datasets: int, alpha: float, comparisons: int = 1
) -> list[float]:
    """Hoeffding interval for an average of independent Bernoulli means.

    Success probabilities may differ across fixed synthetic worlds. Calibration
    datasets must be independent conditional on those worlds. Bonferroni gives
    simultaneous coverage across the declared comparisons without requiring
    independence between methods or metrics. Floating evaluation is separate
    from the exact model-level certificates.
    """
    if (
        any(
            not isinstance(x, int) or isinstance(x, bool)
            for x in [successes, datasets, comparisons]
        )
        or datasets < 1
        or comparisons < 1
        or not 0 <= successes <= datasets
    ):
        raise ValueError("Valid integer counts of independent datasets required")
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("Alpha must lie in (0,1)")
    radius = math.sqrt(math.log(2 * comparisons / alpha) / (2 * datasets))
    p = successes / datasets
    return [
        max(0.0, math.nextafter(p - radius, -math.inf)),
        min(1.0, math.nextafter(p + radius, math.inf)),
    ]


def summarize_coverage_uncertainty(summary: dict, alpha: float = 0.05) -> dict:
    records = summary["coverage"]
    if summary["status"] != "complete" or not records:
        raise ValueError("Complete coverage report required")
    keys = [(r["method"], r["size"], r["samples_per_row"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate Monte Carlo cells")
    methods = sorted({r["method"] for r in records})
    metrics = ["kernel_inclusion_count", "all_probability_enclosures_contained_count"]
    comparisons = len(metrics) * (len(records) + len(methods))
    intervals = []
    for r in records:
        for metric in metrics:
            intervals.append(
                {
                    "scope": "size and sample-count cell",
                    "method": r["method"],
                    "size": r["size"],
                    "samples_per_row": r["samples_per_row"],
                    "metric": metric,
                    "successes": r[metric],
                    "independent_datasets": r["sampled_datasets"],
                    "simultaneous_ci": coverage_interval(
                        r[metric], r["sampled_datasets"], alpha, comparisons
                    ),
                }
            )
    for method in methods:
        cells = [r for r in records if r["method"] == method]
        for metric in metrics:
            k, n = sum(r[metric] for r in cells), sum(r["sampled_datasets"] for r in cells)
            intervals.append(
                {
                    "scope": "all fixed worlds and sample-count designs",
                    "method": method,
                    "metric": metric,
                    "successes": k,
                    "independent_datasets": n,
                    "simultaneous_ci": coverage_interval(k, n, alpha, comparisons),
                }
            )
    return {
        "schema_version": "monte-carlo-coverage-v1",
        "source_campaign": summary["campaign_id"],
        "source_protocol_digest": summary["protocol_digest"],
        "family_confidence": 1 - alpha,
        "reported_intervals": comparisons,
        "intervals": intervals,
        "estimand": "mean coverage probability over the fixed simulated world/sample-count designs, conditional on those worlds",
        "independence_scope": "independently drawn calibration datasets within each method; methods share datasets and are not pooled as independent trials",
        "method": "Hoeffding plus Bonferroni, numerically evaluated; no identical-success-probability assumption",
        "limitations": [
            "Wide finite-Monte-Carlo intervals do not establish worst-case model coverage.",
            "No human confidence guarantee follows.",
            "World generation, fitted models and numerical model verification have separate sources of uncertainty.",
        ],
    }
