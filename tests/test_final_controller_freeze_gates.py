"""Actual evaluator bodies must reject changed locks before private case access."""

from copy import deepcopy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from test_analysis_freeze import fixture
from test_cloud_summary_boundaries import body


@pytest.mark.parametrize(
    "controller,phase,args",
    [
        ("evaluate_human_one_step.py", "one_step", ("known_users", 20261015)),
        ("evaluate_human_horizons.py", "direct_horizon", ("known_users", 20261015)),
        ("evaluate_human_responses.py", "response_all_groups", ("response_all_groups", 20261015)),
        (
            "evaluate_human_responses.py",
            "response_heldout_groups",
            ("response_heldout_groups", 20261015),
        ),
    ],
)
@pytest.mark.parametrize("mutation", ["status", "source", "config"])
def test_changed_joint_freeze_stops_actual_evaluator_before_private_reads(
    tmp_path, monkeypatch, controller, phase, args, mutation
):
    manifest, raw_config, _ = fixture()
    config = json.loads(raw_config)
    config["freeze_path"] = "freeze.json"
    raw_config = json.dumps(config).encode()
    manifest = deepcopy(manifest)
    manifest["controllers"][phase]["config_sha256"] = hashlib.sha256(raw_config).hexdigest()
    if mutation == "status":
        manifest["status"] = "proposed"
    else:
        field = "source_digest" if mutation == "source" else "config_sha256"
        manifest["controllers"][phase][field] = "b" * 64
    freeze = tmp_path / "freeze.json"
    freeze.write_text(json.dumps(manifest))
    reads = []

    class OnlyFreeze:
        def __truediv__(self, name):
            reads.append(name)
            if name != "freeze.json":
                pytest.fail("Evaluator accessed a private artifact after invalid lock")
            return freeze

    def path(value):
        if str(value) == "/artifacts":
            return OnlyFreeze()
        pytest.fail("Evaluator accessed a path outside the analysis lock")

    def no_inference(*args, **kwargs):
        pytest.fail("Invalid analysis lock reached model inference")

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "behavioral_equivalence.content_neural",
        SimpleNamespace(build_content_model=no_inference),
    )
    reloads = []
    run = body(
        controller,
        "evaluate",
        Path=path,
        snapshot=lambda *args: ({"config.json": raw_config.decode()}, "a" * 64),
        volume=SimpleNamespace(reload=lambda: reloads.append(True)),
    )
    with pytest.raises(ValueError, match="Complete matching final analysis freeze"):
        run(*args)
    assert reads == ["freeze.json"]
    assert reloads == [True]
    assert list(tmp_path.iterdir()) == [freeze]
