import numpy as np
import pytest
from behavioral_equivalence.case_audit import audit_cases
from behavioral_equivalence.calendar_splits import EvaluationCalendar, ROLE_NAMES
from behavioral_equivalence.confirmatory_preparation import prepare_sample

pl = pytest.importorskip("polars", reason="case audit integration runs in the pinned Modal image")


def test_independent_case_audit_reconstructs_selection_and_causal_histories_and_detects_corruption():
    events, bundles = [], []
    for user in [1, 2]:
        for i, t in enumerate(range(10, 450, 10)):
            # Later bundles sometimes arrive before an earlier delayed bundle.
            bundles.append(
                {
                    "user_id": user,
                    "time_ms": t,
                    "available_sum_ms": t + (110 if i % 5 == 0 else 0),
                    "clicks": i % 3,
                    "exposures": 3,
                    "user_split": "train" if user == 1 else "test",
                }
            )
            for j in range(3):
                events.append(
                    {
                        "case_id": len(events),
                        "user_id": user,
                        "time_ms": t,
                        "is_click": int(j < i % 3),
                        "video_id": i,
                        "tab": j,
                        "is_rand": j % 2,
                    }
                )
    events, bundles = pl.DataFrame(events), pl.DataFrame(bundles)
    calendar = EvaluationCalendar(100, 200, 300)
    caps = dict.fromkeys(ROLE_NAMES, 15)
    selected, tokens = prepare_sample(events, bundles, calendar, 0, caps, 22)
    config = {
        "calendar": {"train_end": 100, "validation_end": 200, "calibration_end": 300},
        "sampling_seed": 22,
        "per_user_role_caps": caps,
    }
    e = {c: events[c].to_numpy() for c in events.columns}
    b = {c: bundles[c].to_numpy() for c in bundles.columns}
    c = {
        k: selected[v].to_numpy()
        for k, v in {
            "case_id": "case_id",
            "user_id": "user_id",
            "time_ms": "time_ms",
            "available_sum_ms": "available_sum_ms",
            "tab": "tab",
            "video_id": "video_id",
            "target": "is_click",
            "is_random": "is_rand",
            "bundle_index": "feature_id",
            "role_code": "role_code",
        }.items()
    }
    c["case_id"] = c["case_id"].astype(np.uint64)
    c["user_role_code"] = (
        selected["user_split"].replace_strict({r: i for i, r in enumerate(ROLE_NAMES)}).to_numpy()
    )
    c["tokens"] = tokens
    assert audit_cases(e, b, c, 0, config, 0)["cases_audited"] == len(selected)
    for key, message in [
        ("target", "physical source"),
        ("tokens", "history"),
        ("role_code", "Calendar"),
    ]:
        broken = {k: v.copy() for k, v in c.items()}
        broken[key].flat[0] += 1
        with pytest.raises(ValueError, match=message):
            audit_cases(e, b, broken, 0, config, 0)
    missing = {k: v[1:] for k, v in c.items()}
    with pytest.raises(ValueError, match="priority/cap"):
        audit_cases(e, b, missing, 0, config, 0)
