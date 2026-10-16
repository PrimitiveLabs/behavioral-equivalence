import hashlib
import numpy as np
import pytest
from behavioral_equivalence.human_protocol import identity_fingerprints, confirmatory_cohort


def test_identity_is_order_invariant_and_preserves_duplicate_observations():
    users = np.array([8, 2, 8, 2, 8])
    times = np.array([20, 7, 10, 7, 10])
    tabs = np.array([1, 2, 3, 2, 3])
    records = identity_fingerprints(users, times, tabs)
    expected = hashlib.sha256(
        np.array([[10, 3], [10, 3], [20, 1]], dtype="<i8").tobytes()
    ).hexdigest()
    assert records[1] == {"user_id": 8, "rows": 3, "sha256": expected}
    assert records == identity_fingerprints(users[::-1], times[::-1], tabs[::-1])
    assert records != identity_fingerprints(users[:-1], times[:-1], tabs[:-1])
    with pytest.raises(ValueError, match="integer"):
        identity_fingerprints(users.astype(float), times, tabs)
    assert (
        identity_fingerprints(
            np.array([], dtype=int), np.array([], dtype=int), np.array([], dtype=int)
        )
        == []
    )


def test_pilot_exclusion_matches_reindexed_users_without_outcomes():
    final = [{"user_id": i, "rows": 3, "sha256": f"identity-{i}"} for i in range(20)]
    pilot = [
        {"pilot_user_id": 100, "rows": 3, "sha256": "identity-8"},
        {"pilot_user_id": 101, "rows": 3, "sha256": "identity-2"},
    ]
    result = confirmatory_cohort(pilot, final, 20261015)
    assert result["excluded_user_ids"] == [2, 8]
    assert result["eligible_users"] == 18
    assert not {"2", "8"} & result["user_assignment"].keys()
    assert result == confirmatory_cohort(pilot[::-1], final[::-1], 20261015)
    assert set(result["user_assignment"].values()) == {"train", "validation", "calibration", "test"}
    with pytest.raises(ValueError, match="missing"):
        confirmatory_cohort(pilot, final[:5], 1)
    with pytest.raises(ValueError, match="unique"):
        confirmatory_cohort(pilot, final + final[:1], 1)
    changed = [dict(pilot[0], rows=4)]
    with pytest.raises(ValueError, match="row counts"):
        confirmatory_cohort(changed, final, 1)


def test_observed_27k_archive_contract_requires_all_five_shards_and_every_column():
    import json
    from pathlib import Path
    from copy import deepcopy
    from behavioral_equivalence.human_protocol import validate_27k_logs

    report = Path(__file__).resolve().parents[1] / "reports/preflight/kuairand-27k-inspection.json"
    entries = json.loads(report.read_text())["entries"]
    logs = validate_27k_logs(entries)
    assert len(logs) == 5
    with pytest.raises(ValueError, match="five"):
        validate_27k_logs([e for e in entries if e["name"] != logs[0]])
    changed = deepcopy(entries)
    next(e for e in changed if e["name"] == logs[0])["columns"].remove("is_rand")
    with pytest.raises(ValueError, match="source-column"):
        validate_27k_logs(changed)
