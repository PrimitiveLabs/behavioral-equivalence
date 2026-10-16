"""Atomic conservative reservations for bounded cloud execution."""

import math
from pathlib import Path
import sqlite3


def resource_reservation(protocol: dict, worlds: int) -> float:
    budget = protocol["resource_budget"]
    rates = budget["rates_per_second"]
    if worlds < 1 or any(not math.isfinite(v) or v <= 0 for v in rates.values()):
        raise ValueError("Positive workload and finite positive prices required")
    worker = budget["worker"]
    coordinator = budget["coordinator"]

    def cost(r):
        return r["timeout_seconds"] * (
            r["cpu_cores"] * rates["cpu_core"] + r["memory_gib"] * rates["memory_gib"]
        )

    amount = worlds * cost(worker) + cost(coordinator) + budget["startup_buffer_usd"]
    if not math.isfinite(amount) or amount <= 0:
        raise ValueError("Invalid resource reservation")
    return amount


class BudgetLedger:
    """Keep worst-case reservations until authoritative billing reconciles them."""

    def __init__(self, path: Path, ceiling_usd: float):
        if not math.isfinite(ceiling_usd) or ceiling_usd <= 0:
            raise ValueError("Finite positive ceiling required")
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.ceiling = math.floor(ceiling_usd * 1_000_000)
        with sqlite3.connect(path) as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS ceiling (id INTEGER PRIMARY KEY, micros INTEGER)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS reservation (run_id TEXT PRIMARY KEY, micros INTEGER)"
            )
            db.execute("INSERT OR IGNORE INTO ceiling VALUES (1,?)", (self.ceiling,))
            if db.execute("SELECT micros FROM ceiling WHERE id=1").fetchone()[0] != self.ceiling:
                raise ValueError("Existing ledger has a different ceiling")

    def reserve(self, run_id: str, amount_usd: float) -> dict:
        if not run_id or not math.isfinite(amount_usd) or amount_usd <= 0:
            raise ValueError("Run identity and finite positive amount required")
        amount = math.ceil(amount_usd * 1_000_000)
        with sqlite3.connect(self.path, timeout=30) as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT micros FROM reservation WHERE run_id=?", (run_id,)).fetchone()
            if old is not None and old[0] != amount:
                raise ValueError("Reservation cannot change under an existing run ID")
            used = db.execute("SELECT COALESCE(SUM(micros),0) FROM reservation").fetchone()[0]
            if old is None:
                if used + amount > self.ceiling:
                    raise ValueError("Campaign allowance would be exceeded")
                db.execute("INSERT INTO reservation VALUES (?,?)", (run_id, amount))
                used += amount
        return {
            "run_id": run_id,
            "reserved_usd": amount / 1_000_000,
            "total_reserved_usd": used / 1_000_000,
            "ceiling_usd": self.ceiling / 1_000_000,
            "actual_billing_usd": None,
            "status": "reserved; idempotent run identity",
        }


def check_attempt_allowance(
    source_digest, prior_sources, *, maximum_versions, maximum_calls, new_call
):
    """Bound manually submitted source/phase attempts under a shared reservation.

    One prior entry denotes one coordinator call, including failed/canceled
    calls. Idempotent attachment consumes no additional allowance. Provider
    preemptions/restarts are outside this local submission guard.
    """
    versions = set(prior_sources)
    if source_digest not in versions and len(versions) >= maximum_versions:
        raise ValueError("Program source-attempt allowance exhausted")
    if new_call and len(prior_sources) >= maximum_calls:
        raise ValueError("Program coordinator-attempt allowance exhausted")
