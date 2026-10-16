import importlib.util
from pathlib import Path
import numpy as np
import pytest
from behavioral_equivalence.integer_bounds import verify_abstraction

spec = importlib.util.spec_from_file_location(
    "supplementary_abstraction", Path(__file__).parents[1] / "tools/supplementary_abstraction.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_exact_binary_equivalence_preserves_forecasts_and_replays_certificate():
    e = np.array([[0.8, 0.2], [0.8, 0.2], [0.3, 0.7]])
    result = module.certify_endpoint(e, ((0, 1, 2),), 0)
    assert len(result["blocks"]) == 2
    assert np.array_equal(result["probabilities"], e)
    verify_abstraction(result["parent"], result["abstract"], result["certificate"])


def test_approximate_four_category_merge_has_measured_bound():
    e = np.array([[0.2, 0.3, 0.1, 0.4], [0.21, 0.29, 0.1, 0.4]])
    result = module.certify_endpoint(e, ((0, 1),), 0.006)
    assert len(result["blocks"]) == 1
    assert np.allclose(result["probabilities"].sum(axis=1), 1)
    assert np.max(np.abs(result["probabilities"] - e)) <= 0.006
    assert result["trace"][-1]["radius"] <= 0.006
    verify_abstraction(result["parent"], result["abstract"], result["certificate"])
    strict = module.certify_endpoint(e, ((0, 1),), 0.001)
    assert len(strict["blocks"]) == 2


@pytest.mark.parametrize(
    "emissions,blocks,epsilon",
    [
        (np.array([[0.8, 0.8]]), ((0,),), 0.01),
        (np.array([[0.5, 0.5]]), ((1,),), 0.01),
        (np.array([[0.5, 0.5]]), ((0,),), -1),
    ],
)
def test_invalid_endpoint_model_or_partition_rejected(emissions, blocks, epsilon):
    with pytest.raises(ValueError):
        module.certify_endpoint(emissions, blocks, epsilon)
