import pytest
from pydantic import ValidationError
from behavioral_equivalence.human_comparison import HumanPrimaryResult, decide_primary


def result(protocol="known_users", seed=1, **changes):
    values = dict(
        protocol=protocol,
        seed=seed,
        phase="confirmatory",
        parent_latent_states=256,
        reduced_latent_states=128,
        candidate_minus_reference_nats=0,
        ci95=(-0.001, 0.009),
        users=100,
        cases=1000,
        report_sha256="a" * 64,
    )
    values.update(changes)
    return HumanPrimaryResult(**values)


def test_all_seeds_and_protocols_required_and_one_failure_rejects():
    rows = tuple(result(p, s) for p in ("known_users", "unseen_users") for s in range(1, 6))
    decision = decide_primary(rows, tuple(range(1, 6)))
    assert decision["scientific_outcome"] == "supported"
    assert decision["required_fits"] == 10
    assert decide_primary(rows[:-1], tuple(range(1, 6)))["scientific_outcome"] == "not_evaluated"
    failed = rows[:-1] + (result("unseen_users", 5, reduced_latent_states=256),)
    decision = decide_primary(failed, tuple(range(1, 6)))
    assert decision["scientific_outcome"] == "unsupported"
    assert decision["observed_compression_failures"] == 1
    boundary = rows[:-1] + (result("unseen_users", 5, ci95=(0, 0.01)),)
    assert decide_primary(boundary, tuple(range(1, 6)))["observed_noninferiority_failures"] == 1


def test_matrix_and_summary_validation():
    with pytest.raises(ValueError, match="Duplicate"):
        decide_primary((result(), result()), (1,))
    with pytest.raises(ValueError, match="outside"):
        decide_primary((result(seed=2),), (1,))
    for seeds, margin in (((), 0.01), ((1, 1), 0.01), ((True,), 0.01), ((1,), 0)):
        with pytest.raises(ValueError, match="seeds"):
            decide_primary((), seeds, margin)
    for changes in (
        dict(ci95=(1, 0)),
        dict(ci95=(0, float("nan"))),
        dict(users=1001),
        dict(phase="exploratory"),
        dict(candidate_minus_reference_nats=float("inf")),
    ):
        with pytest.raises(ValidationError):
            result(**changes)
