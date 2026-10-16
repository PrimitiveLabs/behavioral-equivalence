from copy import deepcopy
from itertools import product
import pytest
from behavioral_equivalence.repeated_analysis import summarize_repetitions, METHODS, SUPPORTS


def fixture():
    protocol = dict(
        families=["trust"],
        sizes=[32],
        seeds=[7],
        regimes=["uniform_active"],
        repetitions=2,
        horizons=[5, 10],
        source_digest="s",
        protocol_digest="p",
        campaign_id="c",
    )
    records = [
        dict(
            repetition=r,
            method=m,
            support=s,
            kernel_membership=r == 0,
            properties=[
                dict(property=p, horizon=h, enclosure_containment_count=32)
                for p, h in product(["complete", "disengage", "survival"], [5, 10])
            ],
            completion_widths=[
                dict(horizon=h, mean_active_width=0.2 if s == SUPPORTS[1] else 0.8) for h in [5, 10]
            ],
        )
        for r, m, s in product(range(2), METHODS, SUPPORTS)
    ]
    chunk = dict(
        source_digest="s",
        protocol_digest="p",
        spec=dict(
            family="trust",
            size=32,
            seed=7,
            regime="uniform_active",
            first_repetition=0,
            last_repetition=2,
        ),
        records=records,
    )
    return dict(
        status="complete",
        phase="frozen repeated synthetic calibration",
        failures=[],
        source_digest="s",
        protocol_digest="p",
        chunks=[chunk],
        datasets=2,
        method_dataset_evaluations=8,
    ), protocol


def test_paired_controls_do_not_inflate_independent_dataset_count():
    manifest, protocol = fixture()
    result = summarize_repetitions(manifest, protocol)
    assert result["independent_datasets"] == 2
    assert result["paired_method_support_evaluations"] == 8
    assert result["simultaneous_coverage_intervals"] == 8
    assert len(result["cells"]) == 4
    for cell in result["cells"]:
        assert cell["independent_datasets"] == 2
        assert cell["kernel_inclusion_count"] == 1
        assert cell["all_probability_enclosures_contained_count"] == 2
        assert cell["completion_mean_active_width"]["5"] == (
            0.2 if cell["support"] == SUPPORTS[1] else 0.8
        )


@pytest.mark.parametrize(
    "mutation", ["duplicate", "missing", "properties", "width", "count", "source", "smoke"]
)
def test_reject_incomplete_or_invalid_evidence(mutation):
    manifest, protocol = fixture()
    manifest = deepcopy(manifest)
    chunk = manifest["chunks"][0]
    if mutation == "duplicate":
        chunk["records"].append(deepcopy(chunk["records"][0]))
    elif mutation == "missing":
        chunk["records"].pop()
    elif mutation == "properties":
        chunk["records"][0]["properties"].pop()
    elif mutation == "width":
        chunk["records"][0]["completion_widths"][0]["mean_active_width"] = float("nan")
    elif mutation == "count":
        manifest["datasets"] = 8
    elif mutation == "source":
        chunk["source_digest"] = "changed"
    else:
        manifest["phase"] = "smoke"
    with pytest.raises(ValueError):
        summarize_repetitions(manifest, protocol)
