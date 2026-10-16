import json
from copy import deepcopy
from pathlib import Path
import pytest
from behavioral_equivalence.__main__ import plan


def test_full_campaign_plan_and_rejection_of_broken_dependencies():
    config = json.loads(Path("configs/program.json").read_text())
    summary = plan(config)
    assert summary["deliverable_count"] == 9
    assert summary["task_count"] == 37
    assert summary["submission"] == "disabled in planner"
    broken = deepcopy(config)
    broken["tasks"][0]["depends_on"] = ["R04"]
    with pytest.raises(ValueError, match="cycle"):
        plan(broken)
    broken["tasks"][0]["depends_on"] = ["missing"]
    with pytest.raises(ValueError, match="Missing"):
        plan(broken)


def test_phase_execution_freezes_keep_human_and_final_claim_gates_intact():
    config = json.loads(Path("configs/program.json").read_text())
    tasks = {t["id"]: t for t in config["tasks"]}
    assert tasks["E1"]["depends_on"] == ["P03S"]
    assert tasks["E3S"]["depends_on"] == ["P03U"]
    assert tasks["E4S"]["depends_on"] == ["P03I"]
    assert all(tasks[gate]["status"] == "complete" for gate in ("P03S", "P03U", "P03I"))
    assert "P03" in tasks["E2"]["depends_on"] and "P03" in tasks["E4H"]["depends_on"]
    assert tasks["P03"]["status"] != "complete"
    assert {"A02", "P02"}.issubset(tasks["D08B"]["depends_on"])
    assert {"A02", "P02", "D08B", "R03"}.issubset(config["claim_gates"])
