from fractions import Fraction
import numpy as np
import pytest
from behavioral_equivalence.response_chain import binary_response_chain, response_chain_pair
from behavioral_equivalence.integer_bounds import certify_abstraction, verify_abstraction


def test_binary_point_certificate_and_observable_labels():
    p, q, blocks = response_chain_pair([0.125, 0.375, 0.5, 0.5], [0.25, 0.5], ((0, 1), (2, 3)))
    assert p.labels == ("pending",) * 4 + ("positive", "negative")
    assert np.array_equal(q.initial, [0.5, 0.5, 0, 0])
    certificate = certify_abstraction(
        p, q, blocks, [1], properties={"positive": ("positive",), "negative": ("negative",)}
    )
    epsilon = certificate["epsilon"]
    assert Fraction(
        int(epsilon["numerator_hex"], 16), int(epsilon["denominator_hex"], 16)
    ) == Fraction(1, 8)
    verify_abstraction(p, q, certificate)
    certificate["epsilon"]["numerator_hex"] = "0x0"
    with pytest.raises(ValueError):
        verify_abstraction(p, q, certificate)


def test_identity_has_zero_error_and_nonuniform_class_mass_matches_parent():
    p, q, blocks = response_chain_pair([0.1, 0.2, 0.3, 0.4], [0.2, 0.4], ((0, 1, 2), (3,)))
    assert np.array_equal(q.initial, [0.75, 0.25, 0, 0])
    p, q, blocks = response_chain_pair([0.0, 1.0], [0.0, 1.0], ((0,), (1,)))
    cert = certify_abstraction(p, q, blocks, [1], properties={"positive": ("positive",)})
    assert int(cert["epsilon"]["numerator_hex"], 16) == 0


@pytest.mark.parametrize("p", [[], [float("nan")], [-0.1], [1.1]])
def test_invalid_probabilities(p):
    with pytest.raises(ValueError):
        binary_response_chain(p)


def test_bad_initial_shape_is_rejected():
    with pytest.raises(ValueError):
        binary_response_chain([0.2, 0.8], 0.5)
