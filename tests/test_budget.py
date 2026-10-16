from concurrent.futures import ThreadPoolExecutor
import pytest
from behavioral_equivalence.budget import BudgetLedger


def test_reservations_are_idempotent_and_cannot_overrun(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.sqlite", 10)
    assert ledger.reserve("one", 6)["total_reserved_usd"] == 6
    assert ledger.reserve("one", 6)["total_reserved_usd"] == 6
    with pytest.raises(ValueError, match="exceeded"):
        ledger.reserve("two", 5)
    with pytest.raises(ValueError, match="cannot change"):
        ledger.reserve("one", 3)
    with pytest.raises(ValueError, match="different ceiling"):
        BudgetLedger(tmp_path / "ledger.sqlite", 20)


def test_competing_reservations_are_atomic(tmp_path):
    ledger = BudgetLedger(tmp_path / "ledger.sqlite", 15)

    def reserve(name):
        try:
            ledger.reserve(name, 10)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=2) as pool:
        result = list(pool.map(reserve, ["a", "b"]))
    assert sum(result) == 1
