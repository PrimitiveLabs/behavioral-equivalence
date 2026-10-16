import numpy as np
import pytest

from behavioral_equivalence.horizon_archive import restore_horizon_forecasts
from behavioral_equivalence.horizon_targets import origin_iid_event_probabilities


def test_lossless_direct_iid_and_finite_reconstruction():
    horizons = [1, 5]
    direct = np.arange(30).reshape(3, 10) / 30
    p = np.array([0.1, 0.4, 0.9])
    states = np.array([1, 0, 1])
    table = np.arange(20).reshape(2, 10) / 20
    arrays = {
        "targets": np.zeros((3, 10)),
        "state_k256": states,
        "direct.gru": direct,
        "iid_origin.markov.iid": p,
    }
    result = restore_horizon_forecasts(arrays, {"parent": table}, horizons)
    np.testing.assert_array_equal(result["gru.direct"], direct)
    np.testing.assert_array_equal(result["markov.iid"], origin_iid_event_probabilities(p, horizons))
    np.testing.assert_array_equal(result["parent"], table[states])


def test_bad_state_table_and_probability_rejected():
    arrays = {
        "targets": np.zeros((1, 5)),
        "state_k256": np.array([1]),
        "direct.gru": np.zeros((1, 5)),
    }
    for table in [np.zeros((1, 5)), np.zeros((2, 4))]:
        with pytest.raises(ValueError):
            restore_horizon_forecasts(arrays, {"parent": table}, [1])
    arrays["state_k256"] = np.array([0])
    with pytest.raises(ValueError, match="Duplicate"):
        restore_horizon_forecasts(arrays, {"gru.direct": np.zeros((1, 5))}, [1])
    arrays["direct.gru"] = np.full((1, 5), 1.1)
    with pytest.raises(ValueError, match="probability"):
        restore_horizon_forecasts(arrays, {}, [1])
