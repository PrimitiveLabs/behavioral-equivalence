import numpy as np
import pytest
from behavioral_equivalence.random_response import (
    calibration_discovery_mask,
    response_table,
    supported_witnesses,
)


def test_user_split_is_disjoint_deterministic_and_order_independent():
    users = np.repeat(np.arange(100), 3)
    mask = calibration_discovery_mask(users, 17)
    assert mask.any() and not mask.all()
    assert np.all(mask.reshape(-1, 3) == mask.reshape(-1, 3)[:, :1])
    assert np.array_equal(mask[::-1], calibration_discovery_mask(users[::-1], 17))
    assert not np.array_equal(mask, calibration_discovery_mask(users, 18))
    with pytest.raises(ValueError):
        calibration_discovery_mask(np.array([-1]), 0)
    with pytest.raises(ValueError):
        calibration_discovery_mask(users, -1)


def test_supported_witnesses_hold_scenario_fixed_and_exclude_unknown():
    states = np.repeat([0, 1], 8)
    env = np.tile([0, 0, 1, 1, 4, 4, 5, 5], 2)
    users = np.tile([0, 1, 0, 1, 0, 1, 0, 0], 2)
    targets = np.repeat([0, 1], 8)
    table = response_table(states, env, users, targets, 3)
    assert table["users"][0, 5] == 1
    assert np.isnan(table["probabilities"][2, 0])
    witnesses = supported_witnesses(table, ((0, 1), (2,)), min_cases=2, min_users=2)
    assert [w["environment"] for w in witnesses] == [0, 1]
    assert all(
        w["states"] == (0, 1) and w["scenario"] == 0 and w["probability_gap"] == 1
        for w in witnesses
    )
    assert witnesses[0]["js_nats"] == pytest.approx(np.log(2))
    assert supported_witnesses(table, ((0,), (1,), (2,)), min_cases=2, min_users=2) == []
    with pytest.raises(ValueError):
        supported_witnesses(table, ((0, 1), (1, 2)))
    with pytest.raises(ValueError):
        supported_witnesses(table, ((0, 1, 2),), min_cases=0)


@pytest.mark.parametrize(
    "states,env,users,y",
    [
        ([-1], [0], [0], [1]),
        ([0], [75], [0], [1]),
        ([0], [0], [-1], [1]),
        ([0], [0], [0], [2]),
        ([0, 1], [0], [0], [1]),
    ],
)
def test_invalid_observations(states, env, users, y):
    with pytest.raises(ValueError):
        response_table(np.array(states), np.array(env), np.array(users), np.array(y), 2)


def test_empty_table_and_invalid_table():
    empty = np.array([], dtype=np.int64)
    table = response_table(empty, empty, empty, empty, 2)
    assert table["counts"].sum() == 0
    assert supported_witnesses(table, ((0, 1),)) == []
    table["counts"][0, 0] = 1
    with pytest.raises(ValueError):
        supported_witnesses(table, ((0, 1),))
