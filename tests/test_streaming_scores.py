import numpy as np
import pytest
from behavioral_equivalence.statistics import prediction_scores, paired_user_nll
from behavioral_equivalence.streaming_scores import (
    metric_block,
    merge_metric_blocks,
    shared_user_bootstrap,
)


def test_streamed_metrics_match_full_case_metrics_with_unequal_users_and_bins():
    ids = np.arange(7)
    users = np.array([1, 1, 1, 2, 3, 3, 4])
    targets = np.array([1, 0, 1, 0, 1, 0, 1])
    probabilities = np.array([1.0, 0.0, 0.85, 0.4, 0.3, 0.7, 0.6])
    blocks = [
        metric_block(ids[:4], users[:4], targets[:4], probabilities[:4]),
        metric_block(ids[4:], users[4:], targets[4:], probabilities[4:]),
    ]
    actual = merge_metric_blocks(blocks)
    expected = prediction_scores(targets, probabilities, users)
    for name in [
        "nll_case_mean",
        "brier_case_mean",
        "nll_equal_user_mean",
        "brier_equal_user_mean",
    ]:
        assert actual["metrics"][name] == pytest.approx(expected[name], abs=1e-14)
    for a, e in zip(
        actual["metrics"]["calibration_bins"], expected["calibration_bins"], strict=True
    ):
        for key in ["cases", "users", "equal_user_weight", "predicted", "observed"]:
            if e[key] is None:
                assert a[key] is None
            else:
                assert a[key] == pytest.approx(e[key])
    assert np.array_equal(actual["private_user_statistics"]["case_counts"], [3, 1, 2, 1])


def test_split_user_is_rejected_rather_than_getting_multiple_votes():
    block = metric_block(np.array([1]), np.array([3]), np.array([1]), np.array([0.5]))
    with pytest.raises(ValueError, match="multiple"):
        merge_metric_blocks([block, block])


def test_empty_cohort_has_no_invented_score():
    empty = metric_block(
        np.array([], dtype=int), np.array([], dtype=int), np.array([]), np.array([])
    )
    report = merge_metric_blocks([empty])["metrics"]
    assert report["users"] == report["cases"] == 0
    assert report["nll_equal_user_mean"] is None
    assert merge_metric_blocks([])["metrics"]["nll_case_mean"] is None


def test_shared_bootstrap_matches_explicit_user_resampling_and_preserves_sign():
    ids = np.arange(6)
    users = np.array([7, 7, 7, 2, 9, 9])
    y = np.array([1, 0, 1, 0, 1, 0])
    reference = np.full(6, 0.5)
    candidate = np.array([0.9, 0.1, 0.8, 0.2, 0.4, 0.7])
    a = merge_metric_blocks([metric_block(ids, users, y, candidate)])["private_user_statistics"]
    b = merge_metric_blocks([metric_block(ids, users, y, reference)])["private_user_statistics"]
    actual = shared_user_bootstrap(
        a["user_ids"],
        a["case_counts"],
        {"a": a["user_nll"], "b": b["user_nll"]},
        {"forward": ("a", "b"), "reverse": ("b", "a"), "identity": ("a", "a")},
        replications=257,
        seed=31,
        batch_size=11,
    )
    expected = paired_user_nll(y, reference, candidate, users, replications=257, seed=31)
    assert actual["forward"]["ci95"] == pytest.approx(expected["ci95"], abs=1e-14)
    assert actual["forward"]["candidate_minus_reference_nats"] == pytest.approx(
        expected["candidate_minus_reference_nats"], abs=1e-14
    )
    assert actual["reverse"]["ci95"] == pytest.approx([-expected["ci95"][1], -expected["ci95"][0]])
    assert actual["identity"]["ci95"] == [0, 0]
    assert not actual["identity"]["improved"]
    assert actual["identity"]["noninferior_at_margin"]


def test_invalid_case_or_probability_alignment_is_rejected():
    with pytest.raises(ValueError):
        metric_block(np.array([1, 1]), np.array([1, 2]), np.array([1, 0]), np.array([0.5, 0.5]))
    with pytest.raises(ValueError):
        metric_block(np.array([1]), np.array([1]), np.array([1]), np.array([1.1]))
    with pytest.raises(ValueError):
        shared_user_bootstrap([1, 2], [1, 1], {"a": [0.1]}, {"test": ("a", "a")})
    with pytest.raises(ValueError):
        shared_user_bootstrap([1, 1], [1, 1], {"a": [0.1, 0.2]}, {"test": ("a", "a")})


def test_replicated_forecasts_average_losses_not_probabilities_and_keep_original_users():
    ids = np.arange(4)
    users = np.array([1, 1, 1, 2])
    y = np.array([1, 0, 1, 0])
    p = np.array([[0.1, 0.9, 0.6], [0.3, 0.7, 0.2], [0.4, 0.6, 0.5], [0.8, 0.2, 0.4]])
    actual = merge_metric_blocks([metric_block(ids, users, y, p)])
    individual = [merge_metric_blocks([metric_block(ids, users, y, p[:, j])]) for j in range(3)]
    for key in ["user_nll", "user_brier"]:
        np.testing.assert_allclose(
            actual["private_user_statistics"][key],
            np.mean([s["private_user_statistics"][key] for s in individual], axis=0),
        )
    assert actual["metrics"]["cases"] == 4
    assert actual["metrics"]["users"] == 2
    assert actual["metrics"]["prediction_copies"] == 12
    assert sum(b["cases"] for b in actual["metrics"]["calibration_bins"]) == 12
    assert sum(
        b["equal_user_weight"] for b in actual["metrics"]["calibration_bins"]
    ) == pytest.approx(1)
    pooled = merge_metric_blocks([metric_block(ids, users, y, p.mean(axis=1))])
    assert actual["metrics"]["nll_equal_user_mean"] > pooled["metrics"]["nll_equal_user_mean"]
    np.testing.assert_array_equal(actual["private_user_statistics"]["case_counts"], [3, 1])


def test_replicated_empty_and_mixed_bucket_rules():
    empty = np.array([], dtype=int)
    block = metric_block(empty, empty, empty, np.empty((0, 3)))
    assert merge_metric_blocks([block])["metrics"]["prediction_copies"] == 0
    flat = metric_block(empty, empty, empty, np.empty(0))
    with pytest.raises(ValueError, match="replication"):
        merge_metric_blocks([block, flat])
    with pytest.raises(ValueError):
        metric_block(empty, empty, empty, np.empty((0, 0)))
