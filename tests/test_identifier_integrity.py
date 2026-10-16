"""Exact identifiers across accepted integer widths, with signed-path compatibility."""

import hashlib
import json
from pathlib import Path
import struct
import numpy as np
import pytest
from behavioral_equivalence.random_response import response_table
from behavioral_equivalence.item_context import encode_items
from behavioral_equivalence.human_protocol import identity_fingerprints


def legacy(fixture, function):
    record = json.loads(Path("data/synthetic/fixtures/release", fixture).read_text())
    assert hashlib.sha256(record["source_module"].encode()).hexdigest() == record["source_sha256"]
    scope = {"__package__": "behavioral_equivalence"}
    exec(compile(record["source_module"], "frozen-project-source", "exec"), scope)
    return scope[function]


@pytest.mark.parametrize("unsigned", ["users", "environments"])
def test_response_keys_preserve_unsigned_identifiers_and_outcome_statistics(unsigned):
    states, environments, users = np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([1, 2, 1])
    if unsigned == "users":
        users = np.array([2**64 - 1, 2**64 - 2, 2**64 - 1], dtype=np.uint64)
    else:
        environments = environments.astype(np.uint64)
    table = response_table(states, environments, users, [0, 1, 1], 2)
    assert table["counts"][0, 1] == 3 and table["users"][0, 1] == 2
    assert table["positives"][0, 1] == 2 and table["probabilities"][0, 1] == pytest.approx(2 / 3)
    with pytest.raises(TypeError):
        legacy("response-integer-identities-v1.json", "response_table")(
            states, environments, users, [0, 1, 1], 2
        )


@pytest.mark.parametrize(
    "v_dtype,x_dtype", [(np.int64, np.uint64), (np.uint64, np.int64), (np.uint64, np.uint64)]
)
def test_item_lookup_preserves_adjacent_ids_above_float_exactness(v_dtype, x_dtype):
    vocabulary = np.array([0, 2**53, 2**53 + 1, 2**63 - 1], dtype=v_dtype)
    items = np.array([2**53 + 1, 2**53, 2**53 + 2, 0, 2**63 - 1], dtype=x_dtype)
    np.testing.assert_array_equal(encode_items(vocabulary, items), [3, 2, 0, 1, 4])


def test_item_lookup_handles_entire_unsigned_range_and_preserves_signed_callers():
    v = np.array([1, 2**64 - 2, 2**64 - 1], dtype=np.uint64)
    np.testing.assert_array_equal(
        encode_items(v, np.array([2**64 - 1, 2**64 - 2, 0], dtype=np.uint64)), [3, 2, 0]
    )
    old = legacy("item-integer-identities-v1.json", "encode_items")
    assert old(
        np.array([2**53, 2**53 + 1], dtype=np.int64), np.array([2**53 + 1], dtype=np.uint64)
    ).tolist() != [2]
    rng = np.random.default_rng(23)
    for _ in range(20):
        v = np.unique(rng.integers(0, 100, 20, dtype=np.int64))
        items = rng.integers(0, 100, 30, dtype=np.int64)
        np.testing.assert_array_equal(encode_items(v, items), old(v, items))


def test_fingerprint_mixed_columns_do_not_merge_distinct_signed64_timestamps():
    old = legacy("fingerprint-integer-identities-v1.json", "identity_fingerprints")
    users, tabs = np.array([1]), np.array([0], dtype=np.int64)
    a, b = np.array([2**53], dtype=np.uint64), np.array([2**53 + 1], dtype=np.uint64)
    assert old(users, a, tabs) == old(users, b, tabs)
    first, second = identity_fingerprints(users, a, tabs), identity_fingerprints(users, b, tabs)
    assert first != second
    assert first[0]["sha256"] == hashlib.sha256(struct.pack("<qq", 2**53, 0)).hexdigest()
    assert second[0]["sha256"] == hashlib.sha256(struct.pack("<qq", 2**53 + 1, 0)).hexdigest()


@pytest.mark.parametrize("column", ["timestamps", "tabs"])
def test_fixed_signed64_fingerprint_codec_rejects_unrepresentable_values(column):
    times, tabs = np.array([1]), np.array([0])
    if column == "timestamps":
        times = np.array([2**63], dtype=np.uint64)
    else:
        tabs = np.array([2**63], dtype=np.uint64)
    with pytest.raises(ValueError, match="signed int64 source schema"):
        identity_fingerprints(np.array([1]), times, tabs)


def test_signed_pipeline_fingerprints_and_response_tables_remain_identical():
    old_identity = legacy("fingerprint-integer-identities-v1.json", "identity_fingerprints")
    old_response = legacy("response-integer-identities-v1.json", "response_table")
    rng = np.random.default_rng(20261016)
    users = rng.integers(0, 30, 500, dtype=np.int64)
    times = rng.integers(10**12, 2 * 10**12, 500, dtype=np.int64)
    tabs = rng.integers(0, 15, 500, dtype=np.int64)
    assert identity_fingerprints(users, times, tabs) == old_identity(users, times, tabs)
    states = rng.integers(0, 8, 500, dtype=np.int64)
    environments = rng.integers(0, 75, 500, dtype=np.int16)
    outcomes = rng.integers(0, 2, 500, dtype=np.int8)
    before = old_response(states, environments, users, outcomes, 8)
    after = response_table(states, environments, users, outcomes, 8)
    for key in before:
        np.testing.assert_array_equal(before[key], after[key])


@pytest.mark.parametrize("values", [[], [-1], [[1]]])
def test_item_vocabulary_rejects_unsupported_training_ids(values):
    from behavioral_equivalence.item_context import fit_item_vocabulary

    with pytest.raises(ValueError):
        fit_item_vocabulary(np.asarray(values))


@pytest.mark.parametrize("maximum", [True, 1.5])
def test_vocabulary_capacity_must_be_an_integer(maximum):
    from behavioral_equivalence.item_context import fit_item_vocabulary

    with pytest.raises(ValueError, match="Positive vocabulary"):
        fit_item_vocabulary(np.array([1]), maximum)


@pytest.mark.parametrize(
    "vocabulary,items", [([2, 1], [1]), ([-1], [1]), ([1], [-1]), ([1], [[1]])]
)
def test_item_encoding_rejects_unsupported_dictionary_or_query(vocabulary, items):
    with pytest.raises(ValueError, match="Sorted unique vocabulary"):
        encode_items(np.asarray(vocabulary), np.asarray(items))


@pytest.mark.parametrize(
    "duration,upload,exposure", [([[1]], [0], [1]), ([1], [0], [1, 2]), ([1], [0], [1.5])]
)
def test_intrinsic_context_shape_and_integer_clock_boundaries(duration, upload, exposure):
    from behavioral_equivalence.intrinsic_context import duration_context

    with pytest.raises(ValueError, match="Aligned duration"):
        duration_context(np.asarray(duration), np.asarray(upload), np.asarray(exposure))


@pytest.mark.parametrize(
    "scaling", [dict(mean=0, scale=0), dict(mean=np.inf, scale=1), dict(mean=0, scale=np.nan)]
)
def test_duration_normalization_rejects_invalid_training_statistics(scaling):
    from behavioral_equivalence.content_scaling import scale_duration

    with pytest.raises(ValueError, match="Finite normalization"):
        scale_duration(np.array([[1, 0]]), scaling)


def test_empty_duration_features_and_exposure_selection_retain_no_invented_cases():
    from behavioral_equivalence.content_scaling import fit_duration_scaling, scale_duration
    from behavioral_equivalence.intrinsic_context import duration_context
    from behavioral_equivalence.item_metadata import duration_groups
    from behavioral_equivalence.randomized_cohort import randomized_role_indices

    empty = np.array([], dtype=np.int64)
    features, known = duration_context(empty, empty, empty)
    assert features.shape == (0, 2) and not known.size
    assert duration_groups(empty).size == 0
    assert fit_duration_scaling(features) == dict(mean=0, scale=1)
    assert scale_duration(features, dict(mean=0, scale=1)).shape == (0, 2)
    selected = randomized_role_indices(empty, empty, empty, empty, empty, empty, "known_users")
    assert all(v.size == 0 for v in selected.values())


def test_randomized_exposures_reject_reverse_time_but_keep_simultaneous_cases():
    from behavioral_equivalence.randomized_cohort import randomized_role_indices

    args = [
        np.array([1, 1]),
        np.array([2, 1]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([3, 3]),
        np.array([3, 3]),
    ]
    with pytest.raises(ValueError, match="Chronological exposures"):
        randomized_role_indices(*args, "known_users")
    args[1] = np.array([1, 1])
    assert randomized_role_indices(*args, "known_users")["test"].tolist() == [0, 1]
