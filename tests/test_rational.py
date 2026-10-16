from copy import deepcopy
from fractions import Fraction
import pytest
from behavioral_equivalence.synthetic import generate
from behavioral_equivalence.rational import certify, verify


def test_exact_certificate_and_corruption_detection():
    world = generate("memory", 16, 8)
    certificate = certify(world.chain, world.partition, "complete", 10)
    assert certificate["epsilon"] == "0"
    assert certificate["observed_max_error"] == "0"
    verify(world.chain, certificate)
    corrupt = deepcopy(certificate)
    corrupt["intervals"][0][1] = "1"
    with pytest.raises(ValueError, match="exact recomputation"):
        verify(world.chain, corrupt)


def test_approximate_certificate_uses_actual_quotient_and_rational_enclosure():
    world = generate("trust", 16, 4, 0.01)
    certificate = certify(world.chain, world.partition, "disengage", 5)
    assert Fraction(certificate["epsilon"]) > 0
    assert Fraction(certificate["observed_max_error"]) <= Fraction(certificate["delta"])
    verify(world.chain, certificate)


def test_every_released_fixture_certificate_recomputes():
    from pathlib import Path
    import json
    import numpy as np
    from behavioral_equivalence.core import Chain

    for world_path in Path("data/synthetic/fixtures").rglob("*-world.json"):
        model = json.loads(world_path.read_text())
        certificate = json.loads(
            world_path.with_name(world_path.name.replace("-world", "-certificate")).read_text()
        )
        chain = Chain(np.array(model["p"]), tuple(model["labels"]), np.array(model["initial"]))
        verify(chain, certificate)
