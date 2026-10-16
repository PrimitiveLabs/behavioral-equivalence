import numpy as np
import pytest
from behavioral_equivalence.heldout_response import (
    response_group_folds,
    teacher_response_prior,
    query_only_table,
    heldout_point_model,
)


def test_all_six_folds_partition_known_groups_without_unknown():
    folds = response_group_folds()
    assert len(folds) == 6
    assert len({f["query"] for f in folds}) == 6
    for f in folds:
        assert not set(f["query"]) & set(f["heldout"])
        assert set(f["query"]) | set(f["heldout"]) == {0, 1, 2, 3}


def test_training_teacher_prior_uses_environment_then_global_fallback():
    result = teacher_response_prior(
        np.array([0, 0, 1]), np.array([0, 0, 1]), np.array([0.2, 0.4, 0.8]), 3
    )
    assert result["probabilities"][0, 0] == pytest.approx(0.3)
    assert result["probabilities"][2, 0] == pytest.approx(0.3)
    assert result["probabilities"][2, 1] == pytest.approx(0.8)
    assert result["probabilities"][2, 2] == pytest.approx(1.4 / 3)
    with pytest.raises(ValueError):
        teacher_response_prior(np.array([3]), np.array([0]), np.array([0.5]), 3)


def test_heldout_calibration_labels_cannot_change_forecasts_or_discovery():
    prior = np.full((4, 75), 0.3)
    n = np.full((4, 75), 10, dtype=np.int64)
    y = np.zeros_like(n)
    users = np.full_like(n, 5)
    first = dict(counts=n, positives=y, users=users)
    query = (0, 1)
    blocks = ((0, 1), (2, 3))
    a = heldout_point_model(prior, first, blocks, query)
    hidden = ~np.isin(np.arange(75) % 5, query)
    y[:, hidden] = 10
    b = heldout_point_model(prior, first, blocks, query)
    assert np.array_equal(a["parent"], b["parent"])
    assert np.array_equal(a["abstract"], b["abstract"])
    assert np.all(a["parent"][:, hidden] == prior[:, hidden])
    visible = query_only_table(first, query)
    assert np.all(visible["counts"][:, hidden] == 0)
    assert np.isnan(visible["probabilities"][:, hidden]).all()
    y[:, 0] = 10
    c = heldout_point_model(prior, first, blocks, query)
    assert not np.array_equal(a["parent"], c["parent"])
    assert np.array_equal(c["parent"][:, hidden], a["parent"][:, hidden])


def test_invalid_group_sets_and_prior_mass_rejected():
    n = np.ones((4, 75), dtype=np.int64)
    table = dict(counts=n, positives=n, users=n)
    for groups in [(0, 0), (0, 4), (0,)]:
        with pytest.raises(ValueError):
            query_only_table(table, groups)
    with pytest.raises(ValueError):
        heldout_point_model(np.full((4, 75), 0.5), table, ((0, 1), (2, 3)), (0, 1), 0)
