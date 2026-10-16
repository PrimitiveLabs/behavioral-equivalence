"""Pilot controller input guards and local synthetic refinement integration."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from test_cloud_summary_boundaries import body

ROOT = Path(__file__).parents[1]


@pytest.mark.parametrize(
    "controller", ["refinement_pilot.py", "uncertainty_pilot.py", "compare_uncertainty.py"]
)
@pytest.mark.parametrize("run_id", ["", "../escape", "a/b", "with_underscore"])
def test_pilot_run_identifiers_reject_path_escape_before_writes(controller, run_id):
    run = body(controller, "run")
    with pytest.raises(ValueError, match="Invalid run ID"):
        run(run_id)


def test_refinement_pilot_records_equal_search_budget_and_retains_limitations(
    tmp_path, monkeypatch
):
    import behavioral_equivalence.synthetic as synthetic

    monkeypatch.setattr(synthetic, "FAMILIES", ("memory",))
    (tmp_path / "artifacts").mkdir()

    def local_path(value):
        text = str(value)
        if text == "/artifacts":
            return tmp_path / "artifacts"
        if text == "/research/src/behavioral_equivalence":
            return ROOT / "src/behavioral_equivalence"
        return Path(value)

    calls = []
    run = body(
        "refinement_pilot.py",
        "run",
        Path=local_path,
        volume=SimpleNamespace(commit=lambda: calls.append("commit")),
    )
    result = run("fixture")
    report = json.loads((tmp_path / "artifacts/fixture/report.json").read_text())
    assert result["records"] == len(report["records"]) == 27
    assert report["phase"] == "engineering_pilot"
    assert "no human intervention evidence" in report["limitations"]
    assert all(
        r["search_budget"] == r["actual_environment_evaluations"] == 8 for r in report["records"]
    )
    assert all(r["states_after"] >= r["states_before"] for r in report["records"])
    assert all(
        np.isfinite(r[k]) and 0 <= r[k] <= 1
        for r in report["records"]
        for k in ["error_before", "error_after", "epsilon_before", "epsilon_after"]
    )
    assert calls == ["commit"]
    with pytest.raises(FileExistsError):
        run("fixture")
