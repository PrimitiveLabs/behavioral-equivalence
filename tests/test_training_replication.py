"""The supplemental seed matrix must preserve the original experiment."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_supplement_adds_two_seeds_without_changing_data_or_training():
    base = json.loads((ROOT / "configs/human-training-calendar-v3.json").read_text())
    extra = json.loads((ROOT / "configs/human-training-calendar-v4-replicates.json").read_text())
    allowed = {"seeds", "resource_budget", "frozen_at", "replication_scope"}
    assert {k: v for k, v in base.items() if k not in allowed} == {
        k: v for k, v in extra.items() if k not in allowed
    }
    assert not set(base["seeds"]) & set(extra["seeds"])
    assert sorted(base["seeds"] + extra["seeds"]) == list(range(20261015, 20261020))
    budget_changes = {"ceiling_usd", "training_jobs"}
    assert {k: v for k, v in base["resource_budget"].items() if k not in budget_changes} == {
        k: v for k, v in extra["resource_budget"].items() if k not in budget_changes
    }
    assert (
        extra["resource_budget"]["training_jobs"]
        == len(extra["seeds"]) * len(extra["architectures"]) * len(extra["protocols"])
        == 12
    )
    assert (
        base["resource_budget"]["training_jobs"] + extra["resource_budget"]["training_jobs"] == 30
    )
