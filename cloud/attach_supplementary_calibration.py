"""Attach supplementary targets to authenticated adjacent-calibration rows on private Modal storage."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-supplementary-adjacent-targets-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "pyarrow==20.0.0", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/tools", "OPENBLAS_NUM_THREADS": "2"})
    .add_local_file(
        ROOT / "tools/supplementary_outcomes.py",
        remote_path="/research/tools/supplementary_outcomes.py",
    )
)
PREPARED = "datasets/kuairand-27k/prepared-candidate-v2"
CASE_ROOT = PREPARED + "/adjacent-calibration-314b2e408cb56897"
CASE_REPORT_SHA = "ded969e03e56d879bb7a2dbff3aa637c69eee821e45024125103a7a21ea09b19"
PREPARED_REPORT_SHA = "f94f6807749dcfa8b98df4329f3102081d057f90ec96aee6129cc03f030492fd"


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def source_identity():
    import hashlib
    import json

    base = (
        Path("/research/tools")
        if Path("/research/tools/supplementary_outcomes.py").exists()
        else ROOT / "tools"
    )
    source = {
        "cloud.py": Path(__file__).read_text(),
        "supplementary_outcomes.py": (base / "supplementary_outcomes.py").read_text(),
    }
    return source, hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def attach():
    import json
    import time
    import numpy as np
    import pyarrow.parquet as pq
    from supplementary_outcomes import FEEDBACK_FIELDS, source_bound_targets

    volume.reload()
    started = time.monotonic()
    source, digest = source_identity()
    data = Path("/artifacts") / PREPARED
    cases_root = Path("/artifacts") / CASE_ROOT
    if (
        checksum(data / "report.json") != PREPARED_REPORT_SHA
        or checksum(cases_root / "report.json") != CASE_REPORT_SHA
    ):
        raise ValueError("Audited prepared/case manifest changed")
    report = json.loads((cases_root / "report.json").read_text())
    if report["status"] != "complete" or len(report["buckets"]) != 32:
        raise ValueError("All 32 complete source buckets required")
    root = Path("/artifacts") / ("supplementary-adjacent-targets-" + digest[:16])
    root.mkdir(exist_ok=True)
    records = []
    fields = ["user_id", "time_ms", "video_id", "tab", "is_click", *FEEDBACK_FIELDS, "play_time_ms"]
    for bucket, record in enumerate(report["buckets"]):
        prepared = data / "data" / f"bucket-{bucket:02d}" / "interactions.parquet"
        packed = cases_root / f"bucket-{bucket:02d}" / "calibration.npz"
        expected_source = record["input_sha256"]["interactions.parquet"]
        expected_cases = record["output_sha256"]["calibration.npz"]
        if checksum(prepared) != expected_source or checksum(packed) != expected_cases:
            raise ValueError("Physical source or case file changed")
        table = pq.read_table(prepared, columns=fields)
        columns = {
            name: table[name].combine_chunks().to_numpy(zero_copy_only=False) for name in fields
        }
        with np.load(packed, allow_pickle=False) as archive:
            cases = {
                name: archive[name]
                for name in [
                    "case_id",
                    "user_id",
                    "time_ms",
                    "video_id",
                    "tab",
                    "target",
                    "role_code",
                    "user_role_code",
                ]
            }
        targets = source_bound_targets(columns, cases, bucket)
        # Independent direct Arrow take: avoid using the production decoded-row arrays.
        ordinal = [int(identity) % (1 << 32) for identity in cases["case_id"]]
        selected = table.take(ordinal)
        for name in FEEDBACK_FIELDS:
            values = selected[name].combine_chunks().to_numpy(zero_copy_only=False)
            reference = np.full(len(values), -1, dtype=np.int8)
            reference[values == 0] = 0
            reference[values == 1] = 1
            if not np.array_equal(reference, targets[name]):
                raise ValueError("Independent feedback target replay disagrees")
        watch = selected["play_time_ms"].combine_chunks().to_numpy(zero_copy_only=False)
        valid = np.isfinite(watch) & (watch >= 0)
        bounds = [(0, 1000), (1000, 5000), (5000, 30000), (30000, np.inf)]
        for k, (low, high) in enumerate(bounds):
            expected = np.where(valid, (watch >= low) & (watch < high), -1).astype(np.int8)
            if not np.array_equal(expected, targets[f"watch_category_{k}"]):
                raise ValueError("Independent watch target replay disagrees")
        path = root / f"bucket-{bucket:02d}.npz"
        np.savez_compressed(path, **cases, **targets)
        item = dict(
            bucket=bucket,
            cases=len(cases["case_id"]),
            source_sha256=expected_source,
            case_sha256=expected_cases,
            output_sha256=checksum(path),
            role_counts={str(role): int((cases["role_code"] == role).sum()) for role in range(4)},
            replay="all cases and all nine binary indicators agree",
        )
        records.append(item)
        del table, columns, selected, cases, targets
    result = dict(
        status="complete",
        source_digest=digest,
        root=str(root),
        prepared_report_sha256=PREPARED_REPORT_SHA,
        case_report_sha256=CASE_REPORT_SHA,
        buckets=records,
        wall_seconds=time.monotonic() - started,
        scope="Calibration query target attachment and arithmetic only. Origin caps apply only after selecting the stored parent pair_current indices. No predictive fitting, final supplementary scores, causal claim or primary-analysis alteration. Individual arrays stay private.",
    )
    (root / "source.json").write_text(json.dumps(source, indent=2))
    (root / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    _, digest = source_identity()
    state = ROOT / ".campaign" / ("supplementary-adjacent-targets-" + digest[:16] + ".json")
    if state.exists():
        saved = json.loads(state.read_text())
        call = modal.FunctionCall.from_id(saved["function_call_id"])
    else:
        amount = 0.30
        reservation = BudgetLedger(ROOT / ".campaign/human-case-audit-budget.sqlite", 4).reserve(
            "supplementary-adjacent-targets-" + digest, amount
        )
        print(json.dumps(reservation), flush=True)
        # A crash after claiming but before retaining a call ID needs inspection.
        marker = state.with_suffix(".submission-attempt")
        with marker.open("x") as handle:
            handle.write(digest + "\n")
        call = attach.spawn()
        saved = dict(source_digest=digest, function_call_id=call.object_id, reservation_usd=amount)
        state.write_text(json.dumps(saved, indent=2))
    print(json.dumps(saved), flush=True)
    result = call.get()
    output = ROOT / "reports/preflight" / ("supplementary-adjacent-targets-" + digest[:16] + ".json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status=result["status"],
                buckets=len(result["buckets"]),
                wall_seconds=result["wall_seconds"],
            )
        ),
        flush=True,
    )
