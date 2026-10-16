"""Resource quotes and the actual frozen synthetic-protocol boundary."""

import ast
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pytest
from behavioral_equivalence.budget import BudgetLedger, resource_reservation

ROOT = Path(__file__).parents[1]


def quote():
    return {
        "resource_budget": {
            "rates_per_second": {"cpu_core": 2.0, "memory_gib": 3.0},
            "worker": {"timeout_seconds": 5, "cpu_cores": 2, "memory_gib": 4},
            "coordinator": {"timeout_seconds": 7, "cpu_cores": 1, "memory_gib": 1},
            "startup_buffer_usd": 0.5,
        }
    }


def test_quote_prices_all_workers_coordinator_and_startup_once():
    assert resource_reservation(quote(), 2) == 195.5


@pytest.mark.parametrize("worlds", [0, -1])
def test_quote_rejects_nonpositive_world_count(worlds):
    with pytest.raises(ValueError):
        resource_reservation(quote(), worlds)


@pytest.mark.parametrize("rate", [0.0, -1.0, float("nan"), float("inf")])
def test_quote_rejects_nonfinite_or_nonpositive_prices(rate):
    p = quote()
    p["resource_budget"]["rates_per_second"]["cpu_core"] = rate
    with pytest.raises(ValueError):
        resource_reservation(p, 1)


@pytest.mark.parametrize("startup", [float("nan"), float("inf"), -1000.0])
def test_quote_rejects_nonfinite_or_nonpositive_total(startup):
    p = quote()
    p["resource_budget"]["startup_buffer_usd"] = startup
    with pytest.raises(ValueError):
        resource_reservation(p, 1)


@pytest.mark.parametrize("value", [0.0, -1.0, float("nan"), float("inf")])
def test_ledger_rejects_invalid_ceiling_and_amount(tmp_path, value):
    with pytest.raises(ValueError):
        BudgetLedger(tmp_path / "invalid.sqlite", value)
    ledger = BudgetLedger(tmp_path / "valid.sqlite", 1)
    with pytest.raises(ValueError):
        ledger.reserve("invalid", value)
    assert ledger.reserve("valid", 1)["total_reserved_usd"] == 1


def test_money_rounding_is_conservative_and_failed_transactions_do_not_consume_budget(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.sqlite", 0.0000039)
    assert ledger.reserve("first", 0.0000011)["total_reserved_usd"] == 0.000002
    with pytest.raises(ValueError):
        ledger.reserve("overflow", 0.0000011)
    with pytest.raises(ValueError):
        ledger.reserve("first", 0.0000021)
    assert (
        BudgetLedger(tmp_path / "ledger.sqlite", 0.0000039).reserve("last", 0.000001)[
            "total_reserved_usd"
        ]
        == 0.000003
    )


def validator():
    path = ROOT / "cloud/synthetic_campaign.py"
    tree = ast.parse(path.read_text())
    function = next(
        x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "validate_protocol"
    )
    namespace = {"Path": Path}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["validate_protocol"]


def refingerprint(p):
    semantic = {k: v for k, v in p.items() if k not in ["campaign_id", "protocol_digest"]}
    digest = hashlib.sha256(json.dumps(semantic, sort_keys=True).encode()).hexdigest()
    p["protocol_digest"] = digest
    p["campaign_id"] = "synthetic-main-v3-" + digest[:12]


def test_frozen_protocol_boundary_accepts_original_design():
    validator()(json.loads((ROOT / "configs/synthetic-main-v3.json").read_text()))


@pytest.mark.parametrize(
    "change",
    [
        "fingerprint",
        "campaign",
        "no_source",
        "source_path",
        "source_suffix",
        "worker",
        "coordinator",
        "startup",
        "allowance",
        "price",
    ],
)
def test_protocol_rejects_corruption_or_resources_inconsistent_with_executable_limits(change):
    p = deepcopy(json.loads((ROOT / "configs/synthetic-main-v3.json").read_text()))
    if change == "fingerprint":
        p["protocol_digest"] = "0" * 64
    elif change == "campaign":
        p["campaign_id"] = "wrong"
    else:
        if change == "no_source":
            p["source_files"] = []
        if change == "source_path":
            p["source_files"] = ["../core.py"]
        if change == "source_suffix":
            p["source_files"] = ["core.txt"]
        if change == "worker":
            p["resource_budget"]["worker"]["cpu_cores"] = -1
        if change == "coordinator":
            p["resource_budget"]["coordinator"]["memory_gib"] = 2
        if change == "startup":
            p["resource_budget"]["startup_buffer_usd"] = 1
        if change == "allowance":
            p["resource_budget"]["compute_allowance_usd"] = 0
        if change == "price":
            p["resource_budget"]["rates_per_second"]["cpu_core"] = 0
        refingerprint(p)
    with pytest.raises(ValueError):
        validator()(p)
