from copy import deepcopy
from fractions import Fraction
import numpy as np
import pytest
from behavioral_equivalence.confidence import certified_hoeffding, verify_hoeffding
from behavioral_equivalence.integer_bounds import interval_kernel


def test_confidence_has_exact_tail_evidence_and_outward_coordinate_endpoints():
    counts = np.array([[200, 800], [0, 1000]])
    model, cert = certified_hoeffding(counts, ("active", "goal"))
    verify_hoeffding(counts, model.labels, cert)
    interval_kernel(model)
    row = cert["rows"][0]
    exp_lower = Fraction(
        int(row["exp_lower_numerator_hex"], 16), int(row["exp_lower_denominator_hex"], 16)
    )
    assert exp_lower >= Fraction(2 * 4, 1) / Fraction(1, 20)
    radius = Fraction(row["radius_numerator"], cert["radius_denominator"])
    assert Fraction.from_float(model.lower[0, 0]) <= Fraction(1, 5) - radius
    assert Fraction.from_float(model.upper[0, 0]) >= Fraction(1, 5) + radius
    damaged = deepcopy(cert)
    damaged["rows"][0]["radius_numerator"] = 0
    with pytest.raises(ValueError, match="recomputation"):
        verify_hoeffding(counts, model.labels, damaged)


def test_unsampled_and_tiny_rows_use_valid_whole_simplex():
    counts = np.array([[0, 0], [0, 1]])
    model, cert = certified_hoeffding(counts, ("active", "goal"))
    assert np.array_equal(model.lower, np.zeros((2, 2)))
    assert np.array_equal(model.upper, np.ones((2, 2)))
    verify_hoeffding(counts, model.labels, cert)
    with pytest.raises(ValueError, match="integer"):
        certified_hoeffding(counts.astype(float), model.labels)
    with pytest.raises(ValueError, match="Rational alpha"):
        certified_hoeffding(counts, model.labels, 0.05)
