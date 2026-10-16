import importlib.util
from pathlib import Path
import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "supplementary", Path(__file__).parents[1] / "tools/supplementary_outcomes.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def columns(watch):
    return {
        **{name: np.zeros(len(watch)) for name in module.FEEDBACK_FIELDS},
        "play_time_ms": watch,
    }


def test_watch_boundaries_zero_and_missingness():
    result = module.supplementary_targets(
        columns([0, 999, 1000, 4999, 5000, 29999, 30000, -1, np.nan, np.inf])
    )
    categories = np.column_stack([result[f"watch_category_{k}"] for k in range(4)])
    assert np.array_equal(categories[:7], np.eye(4, dtype=int)[[0, 0, 1, 1, 2, 2, 3]])
    assert np.all(categories[7:] == -1)
    assert np.array_equal(categories[:7].sum(axis=1), np.ones(7))


def test_feedback_is_field_specific_and_invalid_is_unknown():
    data = columns([0] * 7)
    data["is_like"] = [0, 1, -1, 2, np.nan, np.inf, 0.5]
    data["is_hate"] = [1, 0, 0, 0, 0, 0, 0]
    result = module.supplementary_targets(data)
    assert result["is_like"].tolist() == [0, 1, -1, -1, -1, -1, -1]
    assert result["is_hate"].tolist() == [1, 0, 0, 0, 0, 0, 0]
    assert np.all(result["watch_category_0"] == 1)


@pytest.mark.parametrize(
    "change,message", [("missing", "required"), ("length", "Aligned"), ("rank", "dimensional")]
)
def test_invalid_column_contract(change, message):
    data = columns([0, 1000])
    if change == "missing":
        del data["is_like"]
    elif change == "length":
        data["is_like"] = [0]
    else:
        data["is_like"] = [[0, 1]]
    with pytest.raises(ValueError, match=message):
        module.supplementary_targets(data)


def test_empty_aligned_columns_have_empty_targets():
    assert all(a.shape == (0,) for a in module.supplementary_targets(columns([])).values())


def source_and_cases():
    source = columns([100, 40000, 8000])
    source.update(
        user_id=np.array([1, 1, 2]),
        time_ms=np.array([10, 10, 20]),
        video_id=np.array([7, 7, 8]),
        tab=np.array([1, 1, 2]),
        is_click=np.array([1, 1, 0]),
    )
    source["is_like"] = np.array([0, 1, 0])
    cases = {name: source[name][[1, 0]] for name in ["user_id", "time_ms", "video_id", "tab"]}
    cases.update(
        target=np.array([1, 1]), case_id=np.array([(3 << 32) + 1, 3 << 32], dtype=np.uint64)
    )
    return source, cases


def test_physical_identity_distinguishes_same_event_keys_and_preserves_case_order():
    source, cases = source_and_cases()
    result = module.source_bound_targets(source, cases, 3)
    assert result["is_like"].tolist() == [1, 0]
    assert result["watch_category_3"].tolist() == [1, 0]
    assert result["watch_category_0"].tolist() == [0, 1]


@pytest.mark.parametrize(
    "change",
    [
        "bucket",
        "signed",
        "duplicate",
        "outside",
        "identity",
        "shape",
        "source_shape",
        "invalid_bucket",
    ],
)
def test_physical_join_rejects_wrong_source_or_case_contract(change):
    source, cases = source_and_cases()
    bucket = 3
    if change == "bucket":
        bucket = 2
    elif change == "signed":
        cases["case_id"] = cases["case_id"].astype(np.int64)
    elif change == "duplicate":
        cases["case_id"][1] = cases["case_id"][0]
    elif change == "outside":
        cases["case_id"][0] = (3 << 32) + 100
    elif change == "identity":
        cases["video_id"][0] = 999
    elif change == "shape":
        cases["user_id"] = np.array([1])
    elif change == "source_shape":
        source["is_like"] = np.array([0])
    elif change == "invalid_bucket":
        bucket = True
    with pytest.raises(ValueError):
        module.source_bound_targets(source, cases, bucket)


def test_six_endpoints_use_one_categorical_watch_target():
    indicators = module.supplementary_targets(columns([0, 1000, 5000, 30000, -1]))
    targets = module.categorical_targets(indicators)
    assert targets.shape == (5, 6)
    assert targets[:, -1].tolist() == [0, 1, 2, 3, -1]
    indicators["watch_category_0"][1] = 1
    with pytest.raises(ValueError, match="one-hot"):
        module.categorical_targets(indicators)


def test_role_sample_is_exact_reproducible_and_independent_of_bucket_layout():
    counts = np.array([7, 0, 13])
    sample = module.fixed_role_sample(counts, maximum=8)
    joined = np.concatenate([sample[0], sample[1] + 7, sample[2] + 7])
    assert len(joined) == 8 and len(np.unique(joined)) == 8
    assert np.array_equal(joined, module.fixed_role_sample(np.array([20]), maximum=8)[0])
    assert all(
        np.array_equal(a, b) for a, b in zip(sample, module.fixed_role_sample(counts, maximum=8))
    )
    assert [x.tolist() for x in module.fixed_role_sample(np.array([0, 0]))] == [[], []]


@pytest.mark.parametrize(
    "counts,maximum",
    [(np.array([-1]), 1), (np.array([1.5]), 1), (np.array([1]), 0), (np.array([1]), True)],
)
def test_role_sample_bad_contract(counts, maximum):
    with pytest.raises(ValueError):
        module.fixed_role_sample(counts, maximum)
