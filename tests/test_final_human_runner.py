import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location(
    "final_runner", Path(__file__).parents[1] / "tools/run_final_human_jobs.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_final_report_must_match_source_lock_seed_protocol_and_cohort():
    job = dict(phase="one_step", protocol="known_users", seed=20261015, embargo_hours=24)
    report = dict(
        status="complete",
        source_digest="source",
        analysis_freeze_digest="freeze",
        seed=20261015,
        protocol="known_users",
        split="test",
        feedback_embargo_hours=24,
    )
    module.validate_result(job, report, "source", "freeze")
    for key, value in [
        ("source_digest", "changed"),
        ("seed", 20261016),
        ("feedback_embargo_hours", 0),
        ("split", "validation"),
    ]:
        with pytest.raises(ValueError):
            module.validate_result(job, dict(report, **{key: value}), "source", "freeze")
    response = dict(job, phase="response_heldout_groups")
    with pytest.raises(ValueError):
        module.validate_result(
            response, dict(report, phase="response_all_groups"), "source", "freeze"
        )


@pytest.mark.parametrize(
    "key,value",
    [("status", "failed"), ("analysis_freeze_digest", "other"), ("protocol", "unseen_users")],
)
def test_remaining_identity_fields_rejected(key, value):
    job = dict(phase="direct_horizon", protocol="known_users", seed=1)
    report = dict(
        status="complete",
        source_digest="s",
        analysis_freeze_digest="f",
        seed=1,
        protocol="known_users",
    )
    with pytest.raises(ValueError, match="identities"):
        module.validate_result(job, dict(report, **{key: value}), "s", "f")


@pytest.mark.parametrize(
    "phase", ["direct_horizon", "response_all_groups", "response_heldout_groups"]
)
def test_other_valid_result_phases(phase):
    job = dict(phase=phase, protocol="known_users", seed=1)
    report = dict(
        status="complete",
        source_digest="s",
        analysis_freeze_digest="f",
        seed=1,
        protocol="known_users",
        phase=phase,
    )
    module.validate_result(job, report, "s", "f")


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("observe"),
        ConnectionError("network"),
        ValueError("identity"),
        OSError("download"),
    ],
)
def test_observation_error_never_claims_remote_failure(error):
    result = module.observation_error(error)
    assert result == dict(
        status="needs_inspection", error_type=type(error).__name__, error=str(error)
    )


def test_submission_claim_persists_and_blocks_retry(tmp_path):
    path = tmp_path / "campaign/job.attempt"
    module.claim_submission(path)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        module.claim_submission(path)
    assert path.read_bytes() == original


def test_launcher_lock_excludes_other_process_and_releases(tmp_path):
    import subprocess
    import sys

    path = tmp_path / "launcher.lock"
    script = (
        "import fcntl,sys; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)"
    )
    with module.launcher_lock(path):
        blocked = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True)
        assert blocked.returncode != 0
        assert b"BlockingIOError" in blocked.stderr
        with pytest.raises(RuntimeError, match="Another"):
            with module.launcher_lock(path):
                pytest.fail("concurrent launcher acquired lock")
    released = subprocess.run([sys.executable, "-c", script, str(path)], capture_output=True)
    assert released.returncode == 0, released.stderr


@pytest.mark.parametrize(
    "failure,submission",
    [
        (None, "existing"),
        ("concurrent", "existing"),
        ("observe", "existing"),
        ("download", "existing"),
        ("malformed", "existing"),
        ("identity", "existing"),
        (None, "fresh"),
        (None, "partial"),
        ("submission", "exit"),
        ("submission", "claimed"),
    ],
)
def test_matrix_recovery_retains_original_handle_and_never_resubmits(
    tmp_path, monkeypatch, failure, submission
):
    import hashlib
    import json
    import subprocess
    import sys
    from types import SimpleNamespace

    reports = tmp_path / "reports"
    (reports / "preflight").mkdir(parents=True)
    campaign = tmp_path / ".campaign"
    campaign.mkdir()
    freeze = dict(controllers={"one_step": {"source_digest": "s"}})
    raw = json.dumps(freeze).encode()
    (reports / "final-analysis-freeze.json").write_bytes(raw)
    jobs = [
        dict(
            reservation_id=f"job-{i}",
            phase="one_step",
            protocol="known_users",
            seed=i,
            embargo_hours=24,
            state_file=f"{i}.json",
            controller="unused",
            arguments=[],
        )
        for i in ([1, 2, 3] if failure == "concurrent" else [1, 2])
    ]
    (reports / "final-human-job-plan.json").write_text(
        json.dumps(dict(freeze_sha256=hashlib.sha256(raw).hexdigest(), jobs=jobs))
    )
    for i in [1, 2] if submission == "existing" else [2]:
        (campaign / f"{i}.json").write_text(json.dumps(dict(function_call_id=f"original-{i}")))
    observed = []
    downloads = []
    import threading

    barrier = threading.Barrier(2)

    def from_id(identity):
        observed.append(identity)

        def get():
            if failure == "concurrent":
                import time

                barrier.wait(timeout=5)
                if identity == "original-1":
                    raise TimeoutError("one sibling observation failed")
                time.sleep(0.05)
            if failure == "observe":
                raise TimeoutError("original call observation unavailable")
            return dict(root=f"/artifacts/{identity}")

        return SimpleNamespace(get=get)

    def read_file(path):
        downloads.append(path)
        if failure == "download":
            raise ConnectionError("volume unavailable")
        if failure == "malformed":
            return [b"invalid json"]
        seed = int(path.split("/")[0].split("-")[1])
        return [
            json.dumps(
                dict(
                    status="complete",
                    source_digest="wrong" if failure == "identity" else "s",
                    analysis_freeze_digest=hashlib.sha256(
                        json.dumps(freeze, sort_keys=True).encode()
                    ).hexdigest(),
                    seed=seed,
                    protocol="known_users",
                    split="test",
                    feedback_embargo_hours=24,
                )
            ).encode()
        ]

    fake_modal = SimpleNamespace(
        Volume=SimpleNamespace(from_name=lambda name: SimpleNamespace(read_file=read_file)),
        FunctionCall=SimpleNamespace(from_id=from_id),
    )
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setattr(module, "ROOT", tmp_path)

    def forbidden_submission(*args, **kwargs):
        pytest.fail("Existing handle must never submit a replacement")

    original_popen = subprocess.Popen
    children = []
    if submission == "claimed":
        module.claim_submission(campaign / "1.json.submission-attempt")

    def local_submission(command, **kwargs):
        assert command[1:3] == ["run", "--detach"]
        assert (campaign / "1.json.submission-attempt").exists()
        script = "import pathlib,json,sys,time; p=pathlib.Path(sys.argv[1]); "
        if submission == "exit":
            script += "sys.exit(7)"
        else:
            if submission == "partial":
                script += "p.write_text('{'); time.sleep(0.35); "
            script += "p.write_text(json.dumps({'function_call_id':'original-1'}))"
        child = original_popen([sys.executable, "-c", script, str(campaign / "1.json")], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(
        subprocess,
        "Popen",
        local_submission if submission in {"fresh", "partial", "exit"} else forbidden_submission,
    )
    try:
        module.run_matrix(2 if failure == "concurrent" else 1)
    finally:
        for child in children:
            child.wait(timeout=5)
    checkpoint = json.loads((reports / "preflight/final-human-job-progress.json").read_text())
    first = checkpoint["jobs"]["job-1"]
    if failure == "submission":
        assert "function_call_id" not in first
        assert not (campaign / "1.json").exists()
        assert (campaign / "1.json.submission-attempt").exists()
    else:
        assert first["function_call_id"] == "original-1"
        assert json.loads((campaign / "1.json").read_text())["function_call_id"] == "original-1"
    if failure == "concurrent":
        assert first["status"] == "needs_inspection"
        assert checkpoint["complete"] == 1 and checkpoint["needs_inspection"] == 1
        assert checkpoint["jobs"]["job-2"]["status"] == "complete"
        assert checkpoint["jobs"]["job-3"]["status"] == "pending"
        assert sorted(observed) == ["original-1", "original-2"]
        assert len(list((reports / "final-human").glob("*.json"))) == 1
    elif failure:
        assert first["status"] == "needs_inspection"
        assert checkpoint["failed"] == 0
        assert checkpoint["needs_inspection"] == 1
        assert checkpoint["jobs"]["job-2"]["status"] == "pending"
        assert observed == ([] if failure == "submission" else ["original-1"])
        assert not list((reports / "final-human").glob("*.json"))
    else:
        assert checkpoint["complete"] == 2
        assert observed == ["original-1", "original-2"]
        for row in checkpoint["jobs"].values():
            data = (tmp_path / row["report_path"]).read_bytes()
            assert hashlib.sha256(data).hexdigest() == row["report_sha256"]
        assert len(downloads) == 2
