import numpy as np
import pytest
from behavioral_equivalence.statistics import paired_user_nll


def test_paired_user_weighting_differs_from_interaction_weighting():
    users = np.array(["a"] * 100 + ["b"])
    target = np.ones(101)
    reference = np.full(101, 0.5)
    candidate = np.r_[np.full(100, 0.6), 0.25]
    result = paired_user_nll(target, reference, candidate, users, replications=200)
    expected = (np.log(0.5 / 0.6) + np.log(0.5 / 0.25)) / 2
    assert result["candidate_minus_reference_nats"] == pytest.approx(expected)
    assert result["users"] == 2
    assert not result["noninferior_at_margin"]


def test_identical_models_and_invalid_probability():
    target = np.array([0, 1, 0, 1])
    p = np.array([0.2, 0.7, 0.4, 0.8])
    users = np.arange(4)
    result = paired_user_nll(target, p, p, users, replications=100)
    assert result["ci95"] == [0.0, 0.0]
    assert result["noninferior_at_margin"]
    with pytest.raises(ValueError):
        paired_user_nll(target, p, np.ones(4) * 1.2, users)
