from dataclasses import replace
import pytest
from behavioral_equivalence.data import (
    Interaction,
    PredictionCase,
    audit_cases,
    chronological_cases,
    disjoint_user_cases,
)


def events():
    return [
        Interaction(f"{u}:{t}", str(u), t * 1000, (u + t) % 2, t * 1000)
        for u in range(20)
        for t in range(40)
    ]


def test_chronological_and_embargo():
    cases = chronological_cases(events(), horizon=3)
    report = audit_cases(cases, "chronological")
    assert report["counts"] == {"train": 500, "validation": 40, "calibration": 40, "test": 40}
    assert all(c.event not in c.history for c in cases)
    assert max(c.event.timestamp_ms for c in cases if c.split == "train") == 25000


def test_user_protocol_and_cold_start():
    cases = disjoint_user_cases(events(), seed=3)
    report = audit_cases(cases, "users")
    assert report["users"] == 20
    assert report["counts"] == {"train": 560, "validation": 80, "calibration": 80, "test": 80}
    assert len([c for c in cases if not c.history]) == 20
    assert cases == disjoint_user_cases(list(reversed(events())), seed=3)


def test_detect_adversarial_leaks():
    ev = events()
    with pytest.raises(ValueError, match="after prediction"):
        replace(ev[0], context_available_ms=1)
    with pytest.raises(ValueError, match="History must precede"):
        PredictionCase(ev[0], (ev[1],), "test")
    with pytest.raises(ValueError, match="same user"):
        PredictionCase(ev[1], (ev[40],), "test")
    with pytest.raises(ValueError, match="Duplicate"):
        chronological_cases(ev + [ev[0]])
    with pytest.raises(ValueError, match="timestamp ties"):
        chronological_cases(ev + [replace(ev[0], case_id="new-id")])
    cases = chronological_cases(ev)
    altered = [replace(c, split="test") if c.event.case_id == "0:1" else c for c in cases]
    with pytest.raises(ValueError, match="future targets"):
        audit_cases(altered, "chronological")
    with pytest.raises(ValueError, match="disjoint"):
        audit_cases(cases, "users")


def test_bundle_history_excludes_current_and_late_outcomes():
    from behavioral_equivalence.data import TimestampBundle, available_bundle_history

    earlier = TimestampBundle("u", 100, 150, (0, 1, 1))
    late = TimestampBundle("u", 200, 1000, (1,))
    current = TimestampBundle("u", 300, 350, (0,))
    other = TimestampBundle("other", 100, 150, (1,))
    assert available_bundle_history([current, late, earlier, other], "u", 300) == (earlier,)
    permuted = TimestampBundle("u", 100, 150, (1, 0, 1))
    assert sum(earlier.outcomes) == sum(permuted.outcomes)
    with pytest.raises(ValueError, match="unordered bundle"):
        available_bundle_history([earlier, permuted], "u", 300)
