import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "human_primary", Path(__file__).parents[1] / "tools/summarize_human_primary.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_all_seeds_required_and_identity_cannot_establish_strict_improvement():
    rows = [dict(seed=s, ci95=[-0.01, -0.001]) for s in range(5)]
    assert module.response_decision(rows, range(5))["scientific_outcome"] == "supported"
    rows[-1]["ci95"] = [0, 0]
    assert module.response_decision(rows, range(5))["scientific_outcome"] == "unsupported"
    assert module.response_decision(rows[:-1], range(5))["scientific_outcome"] == "not_evaluated"
    with pytest.raises(ValueError):
        module.response_decision(rows + rows[:1], range(5))
