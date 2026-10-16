import numpy as np
import pytest
from behavioral_equivalence.precision_planning import user_nll_differences, normal_precision_plan


def test_user_mean_loss_difference_respects_equal_user_weighting():
    y = np.array([1, 1, 0])
    ref = np.array([0.5, 0.5, 0.5])
    candidate = np.array([0.25, 0.25, 0.75])
    users = np.array([1, 1, 2])
    assert np.allclose(user_nll_differences(y, ref, candidate, users), np.log(2))
    assert np.array_equal(user_nll_differences(y, ref, ref, users), [0, 0])


def test_normal_planning_scales_with_variance_and_squared_gap():
    report = normal_precision_plan([-1.0, 1.0, -1.0, 1.0], (0.01, 0.005))
    assert report["validation_user_variance"] == pytest.approx(4 / 3)
    a, b = [r["normal_approximate_users"] for r in report["required_users"]]
    assert abs(b - 4 * a) <= 3
    assert normal_precision_plan([0.0, 0.0])["required_users"][0]["normal_approximate_users"] == 2


@pytest.mark.parametrize(
    "d,gaps", [([1.0], (0.01,)), ([0.0, float("nan")], (0.01,)), ([0.0, 1.0], (0.0,))]
)
def test_bad_precision_inputs(d, gaps):
    with pytest.raises(ValueError):
        normal_precision_plan(d, gaps)


def test_bad_case_alignment_or_single_user_is_rejected():
    with pytest.raises(ValueError):
        user_nll_differences([1], [0.5], [0.4], [1])
    with pytest.raises(ValueError):
        user_nll_differences([1, 0], [0.5], [0.4, 0.5], [1, 2])


def test_structural_identity_restricts_to_declared_context_family():
    from behavioral_equivalence.precision_planning import forecast_identity

    c = np.array([[0.1, 0.2], [0.3, 0.4]])
    r = c.copy()
    r[0, 0] = 0.9
    assert forecast_identity(c, r, [1])
    assert not forecast_identity(c, r, [0, 1])
    with pytest.raises(ValueError):
        forecast_identity(c, r, [2])


def test_identical_forecasts_cannot_support_strict_improvement_power():
    from behavioral_equivalence.precision_planning import improvement_precision_plan

    p = improvement_precision_plan([0.0, 0.0], [0.005], structurally_identical=True)
    assert p["required_users"][0]["normal_approximate_users"] is None
    assert p["improvement_status"] == "impossible_for_identical_forecasts"
    with pytest.raises(ValueError):
        improvement_precision_plan([0.0, 0.1], [0.005], structurally_identical=True)
    p = improvement_precision_plan([0.0, 0.0], [0.005], structurally_identical=False)
    assert p["improvement_status"] == "conditional_variance_proxy_only"


def test_repeated_heldout_forecasts_keep_user_cluster_variance():
    y = np.array([1, 0, 1, 0])
    r = np.full(4, 0.5)
    c = np.array([0.6, 0.2, 0.8, 0.7])
    users = np.array([1, 1, 2, 3])
    original = user_nll_differences(y, r, c, users)
    repeated = user_nll_differences(*(np.tile(a, 3) for a in (y, r, c, users)))
    assert np.allclose(repeated, original)
    assert len(repeated) == 3
    assert normal_precision_plan(repeated)["validation_user_variance"] == pytest.approx(
        normal_precision_plan(original)["validation_user_variance"]
    )
