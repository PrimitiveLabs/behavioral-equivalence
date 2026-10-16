import numpy as np
import pytest
from behavioral_equivalence.calendar_splits import EvaluationCalendar
from behavioral_equivalence.confirmatory_preparation import prepare_sample

pl = pytest.importorskip("polars", reason="confirmatory preparation runs in the pinned Modal image")


def fixture():
    bundles, events = [], []
    for user in [1, 2]:
        for i, time in enumerate([10, 20, 90, 100, 110, 150, 200, 220, 300, 310, 350]):
            bundles.append(
                {
                    "user_id": user,
                    "time_ms": time,
                    "available_sum_ms": time + 4,
                    "user_split": "train" if user == 1 else "test",
                    "clicks": i % 2,
                    "exposures": 2,
                }
            )
            for j in range(2):
                events.append(
                    {
                        "case_id": len(events),
                        "user_id": user,
                        "time_ms": time,
                        "tab": j,
                        "video_id": j + i,
                        "is_rand": 0,
                        "is_click": j % 2,
                    }
                )
    return pl.DataFrame(events), pl.DataFrame(bundles)


def test_sampling_is_paired_target_blind_bounded_and_history_is_causal():
    events, bundles = fixture()
    caps = {r: 3 for r in ["train", "validation", "calibration", "test"]}
    selected, history = prepare_sample(
        events, bundles, EvaluationCalendar(100, 200, 300), 5, caps, 77
    )
    flipped, changed = prepare_sample(
        events.with_columns((1 - pl.col("is_click")).alias("is_click")),
        bundles,
        EvaluationCalendar(100, 200, 300),
        5,
        caps,
        77,
    )
    assert selected["case_id"].to_list() == flipped["case_id"].to_list()
    np.testing.assert_array_equal(history, changed)
    assert selected.group_by(["user_id", "role_code"]).len()["len"].max() == 3
    for i, row in enumerate(selected.to_dicts()):
        prior = bundles.filter(
            (pl.col("user_id") == row["user_id"])
            & (pl.col("available_sum_ms") + 5 < row["time_ms"])
        ).sort("time_ms")
        expected = np.zeros(32, dtype=np.uint8)
        tokens = (1 + np.minimum(15, np.floor(16 * prior["clicks"].to_numpy() / 2))).astype(
            np.uint8
        )
        if len(tokens):
            expected[-len(tokens) :] = tokens
        else:
            expected[-1] = 17
        np.testing.assert_array_equal(history[i], expected)
        if row["role_code"] < 3:
            assert row["available_sum_ms"] + 5 <= [100, 200, 300][row["role_code"]]
    assert 90 in selected["time_ms"].to_list()


def test_case_mapping_rejects_duplicate_ids_or_missing_bundle():
    events, bundles = fixture()
    args = (
        EvaluationCalendar(100, 200, 300),
        0,
        dict.fromkeys(["train", "validation", "calibration", "test"], 2),
        1,
    )
    with pytest.raises(ValueError, match="Unique case"):
        prepare_sample(pl.concat([events, events.head(1)]), bundles, *args)
    with pytest.raises(ValueError, match="complete bundle"):
        prepare_sample(events, bundles.filter(pl.col("time_ms") != 10), *args)
