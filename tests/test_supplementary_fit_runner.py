import importlib.util
from pathlib import Path
import sys
import pytest

tools = Path(__file__).parents[1] / "tools"
spec = importlib.util.spec_from_file_location(
    "supplementary_runner", tools / "run_supplementary_fits.py"
)
module = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(tools))
try:
    spec.loader.exec_module(module)
finally:
    sys.path.pop(0)


def report():
    return dict(
        status="complete",
        source_digest=module.DIGEST,
        protocol="known_users",
        architecture="context",
        seed=1,
        records=[
            dict(
                endpoint=name,
                status="complete",
                selected_C=0.1,
                candidates=[dict(C=C, validation=dict(nll=0.5)) for C in [0.1, 1.0, 10.0]],
            )
            for name in [
                "is_like",
                "is_follow",
                "is_comment",
                "is_forward",
                "is_hate",
                "watch_category",
            ]
        ],
    )


def test_complete_grid_and_tied_selection_and_unavailable_endpoint():
    r = report()
    module.validate_report(r, "known_users", "context", 1)
    r["records"][0] = dict(endpoint="is_like", status="unavailable_no_observed_training_labels")
    for row in r["records"][1:]:
        for candidate in row["candidates"]:
            candidate["validation"]["nll"] = None
    module.validate_report(r, "known_users", "context", 1)


@pytest.mark.parametrize(
    "change",
    [
        "digest",
        "protocol",
        "architecture",
        "seed",
        "status",
        "missing_endpoint",
        "partial_grid",
        "wrong_selection",
    ],
)
def test_frozen_fit_report_rejects_identity_or_selection_drift(change):
    r = report()
    if change in ["protocol", "architecture", "status"]:
        r[change] = "changed"
    elif change == "digest":
        r["source_digest"] = "changed"
    elif change == "seed":
        r["seed"] = 2
    elif change == "missing_endpoint":
        r["records"].pop()
    elif change == "partial_grid":
        r["records"][0]["candidates"].pop()
    else:
        r["records"][0]["selected_C"] = 10.0
    with pytest.raises(ValueError):
        module.validate_report(r, "known_users", "context", 1)
