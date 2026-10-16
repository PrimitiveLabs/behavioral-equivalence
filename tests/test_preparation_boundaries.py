"""Preparation constructor failures using real, small Polars tables."""

import pytest
from behavioral_equivalence.calendar_splits import EvaluationCalendar, ROLE_NAMES
from behavioral_equivalence.confirmatory_preparation import prepare_sample

pl = pytest.importorskip("polars", reason="pinned Polars required for preparation boundaries")


def tables():
    bundles = pl.DataFrame(
        dict(
            user_id=[1],
            time_ms=[10],
            available_sum_ms=[10],
            user_split=["train"],
            clicks=[0],
            exposures=[1],
        )
    )
    events = pl.DataFrame(
        dict(
            case_id=[1], user_id=[1], time_ms=[10], tab=[0], video_id=[1], is_rand=[0], is_click=[0]
        )
    )
    return events, bundles


@pytest.mark.parametrize(
    "caps",
    [
        {},
        {"train": 1},
        {**dict.fromkeys(ROLE_NAMES, 1), "extra": 1},
        {**dict.fromkeys(ROLE_NAMES, 1), "train": 0},
        {**dict.fromkeys(ROLE_NAMES, 1), "train": -1},
        {**dict.fromkeys(ROLE_NAMES, 1), "train": True},
        {**dict.fromkeys(ROLE_NAMES, 1), "train": 1.5},
    ],
)
def test_preparation_rejects_missing_extra_or_invalid_role_caps(caps):
    events, bundles = tables()
    with pytest.raises(ValueError, match="Positive per-user cap"):
        prepare_sample(events, bundles, EvaluationCalendar(100, 200, 300), 0, caps, 1)


def test_preparation_rejects_duplicate_timestamp_bundles_before_join():
    events, bundles = tables()
    with pytest.raises(ValueError, match="Timestamp bundles must be unique"):
        prepare_sample(
            events,
            pl.concat([bundles, bundles]),
            EvaluationCalendar(100, 200, 300),
            0,
            dict.fromkeys(ROLE_NAMES, 1),
            1,
        )


def test_preparation_empty_events_produces_no_invented_prediction_cases():
    events, bundles = tables()
    selected, histories = prepare_sample(
        events.head(0),
        bundles,
        EvaluationCalendar(100, 200, 300),
        0,
        dict.fromkeys(ROLE_NAMES, 1),
        1,
    )
    assert len(selected) == 0 and histories.shape == (0, 32)
