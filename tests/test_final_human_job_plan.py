from collections import Counter
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "freeze_plan", Path(__file__).parents[1] / "tools/freeze_human_analysis.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_complete_job_matrix_and_unique_reservation_identities():
    jobs = module.final_jobs({phase: {"source_digest": "a" * 64} for phase in module.PHASES})
    assert len(jobs) == 50
    assert len({j["reservation_id"] for j in jobs}) == 50
    assert Counter(j["phase"] for j in jobs) == {
        "one_step": 30,
        "direct_horizon": 10,
        "response_all_groups": 5,
        "response_heldout_groups": 5,
    }
    assert {j["embargo_hours"] for j in jobs if j["phase"] == "one_step"} == {0, 24, 48}
    assert sum(j["reservation_usd"] for j in jobs) == pytest.approx(21.56144)


def test_missing_phase_or_invalid_source_rejected():
    with pytest.raises(ValueError):
        module.final_jobs({})
    with pytest.raises(ValueError):
        module.final_jobs({phase: {"source_digest": "bad"} for phase in module.PHASES})
