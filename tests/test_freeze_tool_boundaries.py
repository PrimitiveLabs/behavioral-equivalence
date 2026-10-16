"""Analysis-lock job identities and supplementary freeze overwrite guards."""

import json

import pytest

from test_final_human_job_plan import module as human
from test_supplementary_freeze import module as supplementary, fixture


@pytest.mark.parametrize("digest", ["A" * 64, "a" * 63, "g" * 64, "a" * 65])
def test_job_bindings_require_full_lowercase_source_digest(digest):
    bindings = {phase: {"source_digest": "a" * 64} for phase in human.PHASES}
    bindings["one_step"]["source_digest"] = digest
    with pytest.raises(ValueError, match="Full source digest"):
        human.final_jobs(bindings)


@pytest.mark.parametrize(
    "change",
    [
        "architecture",
        "source",
        "endpoint_status",
        "calibration_source",
        "calibration_status",
        "certificate_count",
    ],
)
def test_supplementary_freeze_requires_complete_bound_reports(tmp_path, change):
    head = fixture(tmp_path)
    path = (
        (tmp_path / "reports/supplementary-calibration/known_users-20261015.json")
        if change.startswith("calibration") or change == "certificate_count"
        else head
    )
    r = json.loads(path.read_text())
    if change == "architecture":
        r["architecture"] = "other"
    elif change in ["source", "calibration_source"]:
        r["source_digest"] = "other"
    elif change == "endpoint_status":
        r["records"][0]["status"] = "unavailable_no_observed_training_labels"
    elif change == "calibration_status":
        r["status"] = "partial"
    else:
        r["certificates_replayed"] = 119
    path.write_text(json.dumps(r))
    with pytest.raises(ValueError):
        supplementary.build_config(tmp_path)


def test_supplementary_main_idempotent_and_refuses_existing_different_freeze(tmp_path, monkeypatch):
    fixture(tmp_path)
    (tmp_path / "reports/source-snapshots").mkdir()
    monkeypatch.setattr(supplementary, "ROOT", tmp_path)
    supplementary.main()
    path = tmp_path / "configs/human-supplementary-final-v1.json"
    raw = path.read_bytes()
    supplementary.main()
    assert path.read_bytes() == raw
    path.write_text("historical freeze must remain unchanged")
    with pytest.raises(ValueError, match="Refuse to overwrite"):
        supplementary.main()
    assert path.read_text() == "historical freeze must remain unchanged"
