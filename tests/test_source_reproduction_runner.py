"""Exercise the actual local entrypoint without making provider calls."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import pytest

ROOT = Path(__file__).parents[1]
SOURCE = ROOT / "cloud/reproduce_source_package.py"
spec = importlib.util.spec_from_file_location(
    "release_runner_helpers", ROOT / "tools/run_final_human_jobs.py"
)
helpers = importlib.util.module_from_spec(spec)
spec.loader.exec_module(helpers)


@pytest.fixture
def runner(tmp_path, monkeypatch):
    (tmp_path / "reports/preflight").mkdir(parents=True)
    archive = tmp_path / "source.tar.gz"
    archive.write_bytes(b"source fixture")
    (tmp_path / "reports/current-source-package.json").write_text(
        json.dumps(
            dict(path="source.tar.gz", sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        )
    )
    state = {"spawn": 0, "attach": 0, "fail": False}

    class Call:
        object_id = "fc-fixture"

        def get(self):
            if state["fail"]:
                raise TimeoutError("Observation expired, remote outcome unknown")
            return {"status": "complete", "scope": "fixture response, not a real scientific run"}

    call = Call()

    def spawn(*args):
        state["spawn"] += 1
        return call

    def attach(identity):
        assert identity == "fc-fixture"
        state["attach"] += 1
        return call

    monkeypatch.setitem(__import__("sys").modules, "run_final_human_jobs", helpers)
    tree = ast.parse(SOURCE.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    function.decorator_list = []
    namespace = dict(
        ROOT=tmp_path,
        Path=Path,
        __file__=str(SOURCE),
        modal=SimpleNamespace(FunctionCall=SimpleNamespace(from_id=attach)),
        reproduce=SimpleNamespace(spawn=spawn),
    )
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace["main"], state, tmp_path


def test_success_and_idempotent_attachment_use_one_submission(runner):
    run, state, root = runner
    run()
    run()
    assert state["spawn"] == 1 and state["attach"] == 1
    saved = next((root / ".campaign").glob("source-package-*.json"))
    reservation = json.loads(saved.read_text())["reservation"]
    assert reservation["reserved_usd"] == 0.14
    assert reservation["total_reserved_usd"] == 0.14
    assert reservation["ceiling_usd"] == 40
    reports = list((root / "reports/preflight").glob("*.json"))
    assert len(reports) == 1
    assert json.loads(reports[0].read_text())["status"] == "complete"


def test_observation_timeout_keeps_handle_and_never_replaces_call(runner):
    run, state, root = runner
    state["fail"] = True
    with pytest.raises(TimeoutError):
        run()
    assert len(list((root / ".campaign").glob("source-package-*.json"))) == 1
    state["fail"] = False
    run()
    assert state["spawn"] == 1 and state["attach"] == 1


def test_lifetime_lock_prevents_second_launcher_before_reservation(runner):
    run, state, root = runner
    with helpers.launcher_lock(root / ".campaign/source-package-launcher.lock"):
        with pytest.raises(RuntimeError, match="holds the local lock"):
            run()
    assert state["spawn"] == 0
    assert not (root / ".campaign/nested-human-budget.sqlite").exists()


def test_ambiguous_submission_marker_without_handle_never_authorizes_retry(runner):
    run, state, root = runner
    run()
    saved = next((root / ".campaign").glob("source-package-*.json"))
    saved.unlink()
    with pytest.raises(FileExistsError):
        run()
    assert state["spawn"] == 1


def test_changed_archive_is_rejected_before_cloud_submission(runner):
    run, state, root = runner
    (root / "source.tar.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Source pointer changed"):
        run()
    assert state["spawn"] == 0


def test_remaining_allowance_below_quote_rejects_before_submission(runner):
    from behavioral_equivalence.budget import BudgetLedger

    run, state, root = runner
    BudgetLedger(root / ".campaign/nested-human-budget.sqlite", 40).reserve("prior", 39.87)
    with pytest.raises(ValueError, match="allowance would be exceeded"):
        run()
    assert state["spawn"] == state["attach"] == 0
    assert not list((root / ".campaign").glob("*.submission-attempt"))


def test_quote_covers_full_runtime_and_twenty_minutes_overhead():
    from decimal import Decimal

    tree = ast.parse(SOURCE.read_text())
    reproduce = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reproduce"
    )
    limits = {
        k.arg: ast.literal_eval(k.value)
        for k in reproduce.decorator_list[0].keywords
        if k.arg in {"cpu", "memory", "timeout", "retries", "max_containers"}
    }
    assert limits == {
        "cpu": (2, 2),
        "memory": (8192, 8192),
        "timeout": 1800,
        "retries": 0,
        "max_containers": 1,
    }
    per_second = Decimal("0.0000131") * 2 + Decimal("0.00000222") * 8
    assert per_second * 1800 == Decimal("0.07912800")
    assert per_second * (1800 + 1200) == Decimal("0.13188000")
    assert Decimal("0.14") > per_second * (1800 + 1200)
