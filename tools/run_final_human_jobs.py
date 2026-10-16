"""Resume the locked final human matrix with at most two active job attachments."""

from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import fcntl
import hashlib
import json
import threading

ROOT = Path(__file__).resolve().parents[1]


def validate_result(job, report, source_digest, freeze_digest):
    if (
        report.get("status") != "complete"
        or report.get("source_digest") != source_digest
        or report.get("analysis_freeze_digest") != freeze_digest
        or report.get("seed") != job["seed"]
        or report.get("protocol") != job["protocol"]
    ):
        raise ValueError("Final result identities do not match the locked job")
    if job["phase"] == "one_step" and (
        report.get("split") != "test"
        or report.get("feedback_embargo_hours") != job["embargo_hours"]
    ):
        raise ValueError("Final result has wrong test split or availability cohort")
    if job["phase"].startswith("response_") and report.get("phase") != job["phase"]:
        raise ValueError("Final result has wrong response design")


@contextmanager
def launcher_lock(path):
    """Local lifetime lock. Never unlink a lock file while processes can use it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another launcher holds the local lock") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def observation_error(error):
    # A transport or validation error cannot establish remote job termination.
    return dict(status="needs_inspection", error_type=type(error).__name__, error=str(error)[:1000])


def claim_submission(path):
    """Durable uncertainty marker: a missing call handle never authorizes retry."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        handle.write("Submission attempted. Inspect provider history before any recovery.\n")


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=2, choices=[1, 2])
    args = parser.parse_args()
    with launcher_lock(ROOT / ".campaign/final-human-launcher.lock"):
        run_matrix(args.workers)


def run_matrix(workers):
    import subprocess
    import sys
    import time
    from datetime import datetime, timezone
    import modal

    plan = json.loads((ROOT / "reports/final-human-job-plan.json").read_text())
    raw = (ROOT / "reports/final-analysis-freeze.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != plan["freeze_sha256"]:
        raise ValueError("Job matrix and analysis lock disagree")
    frozen = json.loads(raw)
    freeze_digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    v = modal.Volume.from_name("behavioral-equivalence-research")
    directory = ROOT / "reports/final-human"
    directory.mkdir(exist_ok=True)
    logs = ROOT / "reports/runlogs"
    logs.mkdir(exist_ok=True)
    state_path = ROOT / "reports/preflight/final-human-job-progress.json"
    lock = threading.Lock()
    stop_submissions = threading.Event()
    states = {
        row["reservation_id"]: dict(
            phase=row["phase"],
            protocol=row["protocol"],
            seed=row["seed"],
            embargo_hours=row["embargo_hours"],
            status="pending",
        )
        for row in plan["jobs"]
    }

    def update(identity, **values):
        with lock:
            states[identity].update(values)
            checkpoint = dict(
                observed_at=datetime.now(timezone.utc).isoformat(),
                freeze_sha256=plan["freeze_sha256"],
                expected=len(states),
                complete=sum(s["status"] == "complete" for s in states.values()),
                failed=sum(s["status"] == "failed" for s in states.values()),
                needs_inspection=sum(s["status"] == "needs_inspection" for s in states.values()),
                jobs=states,
            )
            temporary = state_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(checkpoint, indent=2) + "\n")
            temporary.replace(state_path)

    def run(job):
        if stop_submissions.is_set():
            return
        identity = job["reservation_id"]
        local_state = ROOT / ".campaign" / job["state_file"]
        update(identity, status="attaching" if local_state.exists() else "launching")
        try:
            # The source-bound entrypoint attaches to its existing call when a
            # state file exists. It never submits a replacement for that handle.
            if not local_state.exists():
                claim_submission(
                    local_state.with_suffix(local_state.suffix + ".submission-attempt")
                )
                command = [
                    str(Path(sys.executable).parent / "modal"),
                    "run",
                    "--detach",
                    job["controller"],
                    *job["arguments"],
                ]
                with (logs / (identity + ".log")).open("w") as output:
                    process = subprocess.Popen(
                        command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT
                    )
                    while True:
                        if local_state.exists():
                            try:
                                json.loads(local_state.read_text())
                                break
                            except json.JSONDecodeError:
                                pass
                        if process.poll() is not None:
                            raise RuntimeError("Submission failed; inspect retained job log")
                        time.sleep(0.25)
            saved = json.loads(local_state.read_text())
            update(identity, status="submitted", function_call_id=saved["function_call_id"])
            result = modal.FunctionCall.from_id(saved["function_call_id"]).get()
            root = result["root"].removeprefix("/artifacts/")
            content = b"".join(v.read_file(root + "/report.json"))
            report = json.loads(content)
            validate_result(
                job, report, frozen["controllers"][job["phase"]]["source_digest"], freeze_digest
            )
            sha = hashlib.sha256(content).hexdigest()
            name = (
                job["phase"]
                + "-"
                + job["protocol"]
                + "-"
                + str(job["seed"])
                + "-"
                + str(job["embargo_hours"])
                + ".json"
            )
            (directory / name).write_bytes(content)
            update(
                identity,
                status="complete",
                report_sha256=sha,
                report_path=str((directory / name).relative_to(ROOT)),
                remote_root=root,
            )
            print(
                json.dumps(
                    dict(
                        phase=job["phase"],
                        protocol=job["protocol"],
                        seed=job["seed"],
                        embargo_hours=job["embargo_hours"],
                        status="complete",
                        report_sha256=sha,
                    )
                ),
                flush=True,
            )
        except Exception as error:
            stop_submissions.set()
            update(identity, **observation_error(error))
            print(
                json.dumps(
                    dict(
                        identity=identity,
                        status="needs_inspection",
                        error_type=type(error).__name__,
                        error=str(error)[:250],
                    )
                ),
                flush=True,
            )

    # Independent response jobs start early so every scientific track advances.
    ordered = sorted(
        plan["jobs"],
        key=lambda j: (
            j["seed"],
            0 if j["phase"].startswith("response_") else 1,
            j["phase"],
            j["protocol"],
            0 if j["embargo_hours"] == 24 else 1,
        ),
    )
    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(run, ordered))
    print(state_path.read_text(), flush=True)


if __name__ == "__main__":
    main()
