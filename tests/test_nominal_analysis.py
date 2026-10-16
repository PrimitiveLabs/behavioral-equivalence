from copy import deepcopy
import pytest
from behavioral_equivalence.nominal_analysis import summarize_nominal
from behavioral_equivalence.repeated_analysis import METHODS
from test_repeated_analysis import fixture


def setup():
    original, protocol = fixture()
    parent = original["chunks"][0]
    parent["chunk_digest"] = "c" * 64
    for row in parent["records"]:
        for width in row["completion_widths"]:
            width["initial_weighted_width"] = width["mean_active_width"]
    point = deepcopy(parent)
    point["source_digest"] = "new"
    point["original_chunk_digest"] = parent["chunk_digest"]
    point["records"] = [r for r in point["records"] if r["method"] == METHODS[0]]
    for row in point["records"]:
        row["method"] = "empirical_point"
        for prop in row["properties"]:
            prop["proven_noncoverage_count"] = 0
            prop["unresolved_count"] = 0
        for width in row["completion_widths"]:
            width["mean_active_width"] /= 2
            width["initial_weighted_width"] /= 2
    nominal = dict(
        status="complete",
        failures=[],
        phase="secondary paired nominal ablation",
        source_digest="new",
        chunks=[point],
        point_evaluations=4,
    )
    return original, nominal, protocol


def test_paired_summary_retains_independent_dataset_denominator():
    original, nominal, protocol = setup()
    result = summarize_nominal(original, nominal, protocol, "new")
    assert result["independent_datasets"] == 2
    assert result["point_support_evaluations"] == 4
    assert result["simultaneous_intervals"] == 12
    assert all(x["independent_paired_datasets"] == 2 for x in result["paired_completion_widths"])
    assert sorted(
        set(
            x["point_minus_reference_mean_active_width"] for x in result["paired_completion_widths"]
        )
    ) == [-0.4, -0.1]
    assert len(result["point_coverage"]) == 4


@pytest.mark.parametrize(
    "change",
    [
        "source",
        "missing",
        "duplicate",
        "property",
        "count",
        "width",
        "kernel",
        "parent",
        "empty_duplicate",
        "total",
    ],
)
def test_invalid_pairing_or_metrics_rejected(change):
    original, nominal, protocol = setup()
    chunk = nominal["chunks"][0]
    row = chunk["records"][0]
    if change == "source":
        nominal["source_digest"] = "bad"
    elif change == "missing":
        chunk["records"].pop()
    elif change == "duplicate":
        chunk["records"].append(deepcopy(row))
    elif change == "property":
        row["properties"].pop()
    elif change == "count":
        row["properties"][0]["unresolved_count"] = 100
    elif change == "width":
        row["completion_widths"][0]["initial_weighted_width"] = float("nan")
    elif change == "kernel":
        row["kernel_membership"] = "false"
    elif change == "empty_duplicate":
        extra = deepcopy(chunk)
        extra["records"] = []
        nominal["chunks"].append(extra)
    elif change == "parent":
        chunk["original_chunk_digest"] = "bad"
    else:
        nominal["point_evaluations"] = 5
    with pytest.raises(ValueError):
        summarize_nominal(original, nominal, protocol, "new")
