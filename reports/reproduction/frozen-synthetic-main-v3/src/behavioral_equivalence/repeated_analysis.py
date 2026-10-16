"""Fixed-world calibration summaries that retain paired method/support units."""

from itertools import product
import math
from .coverage_analysis import coverage_interval

METHODS = ("certified_hoeffding", "clopper_pearson_float_quantiles")
SUPPORTS = ("unknown_terminal_rows", "known_closed_terminal_labels")


def summarize_repetitions(manifest: dict, protocol: dict) -> dict:
    """Require the complete frozen design before reporting coverage estimates.

    One calibration dataset is a repetition within a world and sampling regime.
    Four method/support evaluations of that dataset are paired, not four trials.
    This validates aggregate structure, not the underlying integer certificates.
    """
    if (
        manifest["status"] != "complete"
        or manifest["failures"]
        or manifest["phase"] != "frozen repeated synthetic calibration"
        or manifest["source_digest"] != protocol["source_digest"]
        or manifest["protocol_digest"] != protocol["protocol_digest"]
    ):
        raise ValueError("Complete source-matched full manifest required")
    worlds = set(
        product(protocol["families"], protocol["sizes"], protocol["seeds"], protocol["regimes"])
    )
    expected = set(product(worlds, range(protocol["repetitions"]), METHODS, SUPPORTS))
    observations = {}
    for chunk in manifest["chunks"]:
        spec = chunk["spec"]
        world = tuple(spec[k] for k in ("family", "size", "seed", "regime"))
        if (
            chunk["source_digest"] != protocol["source_digest"]
            or chunk["protocol_digest"] != protocol["protocol_digest"]
        ):
            raise ValueError("Chunk source or protocol mismatch")
        for record in chunk["records"]:
            key = (world, record["repetition"], record["method"], record["support"])
            if (
                key in observations
                or key not in expected
                or not spec["first_repetition"] <= record["repetition"] < spec["last_repetition"]
            ):
                raise ValueError("Duplicate or off-design repeated evaluation")
            properties = record["properties"]
            expected_properties = set(
                product(("complete", "disengage", "survival"), protocol["horizons"])
            )
            property_keys = [(r["property"], r["horizon"]) for r in properties]
            if (
                len(property_keys) != len(expected_properties)
                or set(property_keys) != expected_properties
            ):
                raise ValueError("Missing or duplicate property/horizon checks")
            widths = record["completion_widths"]
            if sorted(r["horizon"] for r in widths) != sorted(protocol["horizons"]) or any(
                not math.isfinite(r["mean_active_width"]) or not 0 <= r["mean_active_width"] <= 1
                for r in widths
            ):
                raise ValueError("Valid completion widths required")
            observations[key] = record
    if set(observations) != expected:
        raise ValueError("Missing repeated evaluations")
    datasets = len(worlds) * protocol["repetitions"]
    if manifest["datasets"] != datasets or manifest["method_dataset_evaluations"] != len(expected):
        raise ValueError("Manifest counts do not match independent datasets")
    comparisons = len(worlds) * len(METHODS) * len(SUPPORTS) * 2
    cells = []
    for world, method, support in product(sorted(worlds), METHODS, SUPPORTS):
        rows = [observations[(world, r, method, support)] for r in range(protocol["repetitions"])]
        membership = sum(bool(r["kernel_membership"]) for r in rows)
        contained = sum(
            all(p["enclosure_containment_count"] == world[1] for p in r["properties"]) for r in rows
        )
        cells.append(
            {
                "family": world[0],
                "size": world[1],
                "seed": world[2],
                "regime": world[3],
                "method": method,
                "support": support,
                "independent_datasets": len(rows),
                "kernel_inclusion_count": membership,
                "all_probability_enclosures_contained_count": contained,
                "kernel_coverage_simultaneous_ci": coverage_interval(
                    membership, len(rows), 0.05, comparisons
                ),
                "probability_coverage_simultaneous_ci": coverage_interval(
                    contained, len(rows), 0.05, comparisons
                ),
                "completion_mean_active_width": {
                    str(h): sum(
                        next(
                            w["mean_active_width"]
                            for w in r["completion_widths"]
                            if w["horizon"] == h
                        )
                        for r in rows
                    )
                    / len(rows)
                    for h in protocol["horizons"]
                },
            }
        )
    return {
        "schema_version": "repeated-calibration-summary-v1",
        "status": "complete",
        "campaign_id": protocol["campaign_id"],
        "protocol_digest": protocol["protocol_digest"],
        "independent_datasets": datasets,
        "paired_method_support_evaluations": len(expected),
        "simultaneous_coverage_intervals": comparisons,
        "family_confidence": 0.95,
        "cells": cells,
        "scope": "Coverage conditional on each fixed world and sampling regime. Methods/support controls share calibration datasets and are never pooled as independent trials. Summary structure checks do not replace certificate verification.",
    }
