"""Resume the fixed30 supplementary fit matrix without replacing retained handles."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import threading

from run_final_human_jobs import launcher_lock

ROOT = Path(__file__).resolve().parents[1]
DIGEST = "5d9fae59fec2dbbfeafa086f78d98e16347ff7c3a020fb22cc43c89b5f5a3e5a"


def validate_report(report, protocol, architecture, seed):
    if (
        report.get("status") != "complete"
        or report.get("source_digest") != DIGEST
        or report.get("protocol") != protocol
        or report.get("architecture") != architecture
        or report.get("seed") != seed
    ):
        raise ValueError("Exact frozen supplementary fit report required")
    names = ["is_like", "is_follow", "is_comment", "is_forward", "is_hate", "watch_category"]
    if [r["endpoint"] for r in report["records"]] != names:
        raise ValueError("Every fixed endpoint required")
    for row in report["records"]:
        if row["status"] == "unavailable_no_observed_training_labels":
            continue
        if row["status"] != "complete" or [r["C"] for r in row["candidates"]] != [0.1, 1.0, 10.0]:
            raise ValueError("Complete fixed endpoint grid required")
        best = min(
            row["candidates"],
            key=lambda r: (
                r["validation"]["nll"] if r["validation"]["nll"] is not None else float("inf"),
                r["C"],
            ),
        )
        if row["selected_C"] != best["C"]:
            raise ValueError("Head selection differs from frozen validation rule")


def main():
    import subprocess
    import sys
    import modal
    from datetime import datetime, timezone

    snapshot = json.loads(
        (
            ROOT / "reports/source-snapshots" / ("supplementary-heads-" + DIGEST[:16] + ".json")
        ).read_text()
    )
    if hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest() != DIGEST:
        raise ValueError("Frozen source snapshot changed")
    for name, content in snapshot.items():
        local = ROOT / (
            "cloud/train_supplementary_heads.py"
            if name == "cloud.py"
            else "configs/human-supplementary-heads-v1.json"
            if name == "config.json"
            else name
        )
        if local.read_text() != content:
            raise ValueError("Current source differs from reserved fit matrix: " + name)
    cfg = json.loads(snapshot["config.json"])
    jobs = [(p, a, s) for p in cfg["protocols"] for s in cfg["seeds"] for a in cfg["architectures"]]
    lock = threading.Lock()
    stop = threading.Event()
    progress = {f"{p}-{a}-{s}": dict(status="pending") for p, a, s in jobs}
    progress_path = ROOT / "reports/preflight/supplementary-fit-progress.json"
    volume = modal.Volume.from_name("behavioral-equivalence-research")

    def update(identity, **values):
        with lock:
            progress[identity].update(values)
            raw = dict(
                observed_at=datetime.now(timezone.utc).isoformat(),
                source_digest=DIGEST,
                expected=30,
                complete=sum(r["status"] == "complete" for r in progress.values()),
                needs_inspection=sum(r["status"] == "needs_inspection" for r in progress.values()),
                jobs=progress,
            )
            temporary = progress_path.with_suffix(".tmp")
            temporary.write_text(json.dumps(raw, indent=2) + "\n")
            temporary.replace(progress_path)

    def run(job):
        if stop.is_set():
            return
        p, a, s = job
        identity = f"{p}-{a}-{s}"
        path = ROOT / "reports/supplementary-heads" / f"{identity}.json"
        state = ROOT / ".campaign" / f"supplementary-heads-{DIGEST[:16]}-{identity}.json"
        try:
            if not path.exists():
                if state.exists():
                    saved = json.loads(state.read_text())
                    update(identity, status="attaching", function_call_id=saved["function_call_id"])
                    result = modal.FunctionCall.from_id(saved["function_call_id"]).get()
                    content = b"".join(
                        volume.read_file(
                            result["root"].removeprefix("/artifacts/") + "/report.json"
                        )
                    )
                    validate_report(json.loads(content), p, a, s)
                    path.write_bytes(content)
                else:
                    update(identity, status="launching")
                    log = ROOT / "reports/runlogs" / ("supplementary-fit-" + identity + ".log")
                    with log.open("w") as output:
                        subprocess.run(
                            [
                                str(Path(sys.executable).parent / "modal"),
                                "run",
                                "--detach",
                                "cloud/train_supplementary_heads.py",
                                "--protocol",
                                p,
                                "--architecture",
                                a,
                                "--seed",
                                str(s),
                            ],
                            cwd=ROOT,
                            stdout=output,
                            stderr=subprocess.STDOUT,
                            check=True,
                        )
            report = json.loads(path.read_text())
            validate_report(report, p, a, s)
            for name, expected in report["output_sha256"].items():
                content = b"".join(
                    volume.read_file(f"supplementary-heads-{DIGEST[:16]}/{identity}/{name}")
                )
                if hashlib.sha256(content).hexdigest() != expected:
                    raise ValueError("Stored fitted-head artifact changed")
            update(
                identity,
                status="complete",
                report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
            print(json.dumps(dict(identity=identity, status="complete")), flush=True)
        except Exception as error:
            stop.set()
            update(
                identity,
                status="needs_inspection",
                error_type=type(error).__name__,
                error=str(error)[:500],
            )
            print(
                json.dumps(
                    dict(identity=identity, status="needs_inspection", error=str(error)[:250])
                ),
                flush=True,
            )

    with launcher_lock(ROOT / ".campaign/supplementary-fit-launcher.lock"):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(run, jobs))


if __name__ == "__main__":
    main()
