"""Distinct freeze mutations must fail before final prediction work."""

from copy import deepcopy
import importlib.util
from pathlib import Path
import pytest
from behavioral_equivalence.analysis_freeze import verify_analysis_freeze

spec = importlib.util.spec_from_file_location(
    "release_freeze_fixture", Path(__file__).with_name("test_analysis_freeze.py")
)
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)


@pytest.mark.parametrize(
    "change",
    [
        "schema",
        "missing_status",
        "uninspected_flag",
        "component_order",
        "duplicate_fit",
        "wrong_fit_seed",
        "empty_artifacts",
        "absolute_path",
        "nonstring_path",
        "missing_role",
        "hash_characters",
        "hash_type",
    ],
)
def test_each_independent_freeze_identity_boundary_rejects_mutation(change):
    m, c, payload = fixtures.fixture()
    if change == "schema":
        m["schema_version"] = "other"
    if change == "missing_status":
        m.pop("status")
    if change == "uninspected_flag":
        m["test_results_inspected"] = None
    if change == "component_order":
        m["components"].reverse()
    if change == "duplicate_fit":
        m["fits"][-1] = deepcopy(m["fits"][0])
    if change == "wrong_fit_seed":
        m["fits"][-1]["seed"] = 0
    if change == "empty_artifacts":
        m["artifacts"] = []
    if change == "absolute_path":
        m["artifacts"][0]["path"] = "/private"
    if change == "nonstring_path":
        m["artifacts"][0]["path"] = 4
    if change == "missing_role":
        m["artifacts"][0]["role"] = "unknown"
    if change == "hash_characters":
        m["controllers"]["one_step"]["source_digest"] = "g" * 64
    if change == "hash_type":
        m["controllers"]["one_step"]["config_sha256"] = None
    with pytest.raises(ValueError):
        verify_analysis_freeze(
            m, source_digest="a" * 64, config_bytes=c, read_artifact=lambda name: payload
        )


def test_unknown_phase_and_configuration_byte_change_do_not_read_bound_artifacts():
    m, c, payload = fixtures.fixture()
    reads = []

    def read(name):
        reads.append(name)
        return payload

    with pytest.raises(ValueError):
        verify_analysis_freeze(
            m, source_digest="a" * 64, config_bytes=c, read_artifact=read, phase="unknown"
        )
    with pytest.raises(ValueError):
        verify_analysis_freeze(m, source_digest="a" * 64, config_bytes=c + b" ", read_artifact=read)
    assert reads == []
