import numpy as np
import pytest
from behavioral_equivalence.intrinsic_context import duration_context


def test_duration_is_used_only_with_positive_value_and_strictly_prior_upload_day():
    duration = np.array([7000, 7000, 7000, 0, np.nan, 7000])
    upload = np.array([9, 10, 11, 9, 9, np.iinfo(np.int32).min])
    features, known = duration_context(duration, upload, np.full(6, 10 * 86400000))
    assert known.tolist() == [True, False, False, False, False, False]
    assert features[0, 0] == pytest.approx(np.log(8))
    np.testing.assert_array_equal(features[:, 1], [0, 1, 1, 1, 1, 1])
    assert np.isfinite(features).all()


def test_context_arrays_must_align_and_dates_cannot_be_target_like_floats():
    with pytest.raises(ValueError, match="Aligned"):
        duration_context(np.array([1, 2]), np.array([0]), np.array([3]))
    with pytest.raises(ValueError, match="integer"):
        duration_context(np.array([1]), np.array([0.2]), np.array([3]))
