"""Cloud-only independent replay of supplementary categorical forecast artifacts."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
SCORE_SOURCE = "9a4cb36c39c2419e034a088daa1a6622c6bec84f45f443a5ecb69e2a35b8f4d1"
app = modal.App("behavioral-equivalence-supplementary-output-audit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .env({"PYTHONPATH": "/research/tools", "OPENBLAS_NUM_THREADS": "1"})
)
for name in ["supplementary_result_audit.py", "final_result_audit.py"]:
    image = image.add_local_file(ROOT / "tools" / name, remote_path="/research/tools/" + name)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while b := stream.read(8 * 1024**2):
            h.update(b)
    return h.hexdigest()


def snapshot():
    import hashlib
    import json

    base = (
        Path("/research/tools")
        if Path("/research/tools/supplementary_result_audit.py").exists()
        else ROOT / "tools"
    )
    saved = {"cloud.py": Path(__file__).read_text()}
    for name in ["supplementary_result_audit.py", "final_result_audit.py"]:
        saved["tools/" + name] = (base / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=1800,
    retries=0,
    max_containers=2,
    volumes={"/artifacts": volume},
)
def audit(protocol, seed, report_sha256):
    import hashlib
    import json
    import time
    import numpy as np
    from supplementary_result_audit import (
        categorical_audit_block,
        verify_calibration,
        bootstrap_many,
    )
    from final_result_audit import combine, check_metrics

    volume.reload()
    started = time.monotonic()
    saved, digest = snapshot()
    root = Path("/artifacts") / ("supplementary-final-" + SCORE_SOURCE[:16]) / f"{protocol}-{seed}"
    if checksum(root / "report.json") != report_sha256:
        raise ValueError("Frozen supplementary result report changed")
    report = json.loads((root / "report.json").read_text())
    source = json.loads((root / "source.json").read_text())
    if (
        report["status"] != "complete"
        or report["source_digest"] != SCORE_SOURCE
        or report["protocol"] != protocol
        or report["seed"] != seed
        or hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest() != SCORE_SOURCE
    ):
        raise ValueError("Scored source/result identity differs")
    cfg = json.loads(source["config.json"])
    spec = next(p for p in cfg["probes"] if p["protocol"] == protocol and p["seed"] == seed)
    cal = Path("/artifacts") / spec["calibration"]["root"]
    outputs = 0
    for name, expected in report["output_sha256"].items():
        if checksum(root / name) != expected:
            raise ValueError("Saved forecast/statistic output changed")
        outputs += 1
    for name, expected in spec["calibration"]["output_sha256"].items():
        if checksum(cal / name) != expected:
            raise ValueError("Shared calibration tables changed")
    with np.load(cal / "models.npz", allow_pickle=False) as archive:
        tables = {name: archive[name] for name in archive.files}
    abstractions = json.loads((cal / "abstractions.json").read_text())
    with np.load(root / "baseline-tables.npz", allow_pickle=False) as archive:
        baselines = {name: archive[name] for name in archive.files}
    blocks = {}
    case_sha = hashlib.sha256()
    case_count = 0
    for bucket in range(32):
        with np.load(root / f"bucket-{bucket:02d}-factorized.npz", allow_pickle=False) as archive:
            ids = archive["case_id"]
            users = archive["user_id"]
            targets = archive["targets"]
            case_sha.update(ids.astype("<u8").tobytes())
            case_count += len(ids)
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Repeated original forecast cases")
            for j, endpoint in enumerate(cfg["endpoints"]):
                name = endpoint["name"]
                y = targets[:, j]
                for method in ["context", "gru", "transformer"]:
                    blocks.setdefault((name, method), []).append(
                        categorical_audit_block(users, y, archive[f"{method}_endpoint{j}"])
                    )
                for method in ["constant", "markov"]:
                    table = baselines[f"endpoint{j}_{method}"]
                    p = (
                        np.broadcast_to(table, (len(ids), endpoint["classes"]))
                        if method == "constant"
                        else table[archive["markov_state"]]
                    )
                    blocks.setdefault((name, method), []).append(
                        categorical_audit_block(users, y, p)
                    )
                for k in cfg["state_counts"]:
                    p = tables[f"k{k}_endpoint{j}_emissions"][archive[f"k{k}_state"]]
                    blocks.setdefault((name, f"parent_k{k}"), []).append(
                        categorical_audit_block(users, y, p)
                    )
                for record in abstractions:
                    if record["endpoint"] != name:
                        continue
                    p = tables[record["name"] + "_probabilities"][archive["k256_state"]]
                    blocks.setdefault((name, record["name"]), []).append(
                        categorical_audit_block(users, y, p)
                    )
    if case_sha.hexdigest() != report["case_id_sha256"] or case_count != report["cases"]:
        raise ValueError("Ordered complete-case identity differs")
    merged = {key: combine(value) for key, value in blocks.items()}
    maximum_loss = 0.0
    maximum_calibration = 0.0
    maximum_user_loss = 0.0
    maximum_ci = 0.0
    proper_cells = 0
    bootstrap_cells = 0
    if {f"{e}/{m}" for e, m in merged} != set(report["metrics"]):
        raise ValueError("Reported comparator inventory differs from complete archive")
    with np.load(root / "private-user-statistics.npz", allow_pickle=False) as private:
        for (endpoint, method), value in merged.items():
            expected = report["metrics"][endpoint + "/" + method]
            differences = check_metrics(value["metrics"], expected)
            maximum_loss = max(maximum_loss, *differences.values())
            proper_cells += len(differences)
            if max(differences.values()) > 1e-10:
                raise ValueError("Categorical loss replay exceeds declared1e-10 tolerance")
            maximum_calibration = max(
                maximum_calibration,
                verify_calibration(expected, blocks[(endpoint, method)], len(value["users"])),
            )
            if not np.array_equal(
                value["users"], private[endpoint + "__users"]
            ) or not np.array_equal(value["counts"], private[endpoint + "__cases"]):
                raise ValueError("Private original-user support differs")
            for key in ["nll", "brier"]:
                observed = private[f"{endpoint}__{method}__{key}"]
                difference = float(np.max(np.abs(value[key] - observed), initial=0))
                maximum_user_loss = max(maximum_user_loss, difference)
                if difference > 1e-10:
                    raise ValueError("Private user loss replay differs")
    for endpoint, expected in report["comparisons"].items():
        if expected["status"] != "evaluated_descriptive":
            continue
        rows = expected["contrasts"]
        labels = list(rows)
        differences = np.column_stack(
            [
                merged[(endpoint, rows[label]["candidate"])]["nll"]
                - merged[(endpoint, rows[label]["reference"])]["nll"]
                for label in labels
            ]
        )
        actual = bootstrap_many(differences, replications=cfg["bootstrap_replications"], seed=seed)
        for j, label in enumerate(labels):
            row = rows[label]
            error = max(
                abs(float(differences[:, j].mean()) - row["candidate_minus_reference_nats"]),
                float(np.max(np.abs(actual[:, j] - row["ci95"]))),
            )
            maximum_ci = max(maximum_ci, error)
            bootstrap_cells += 1
            if (
                error > 1e-10
                or row["improved"] != bool(row["ci95"][1] < 0)
                or row["noninferior_at_margin"] != bool(row["ci95"][1] < row["margin_nats"])
            ):
                raise ValueError("Paired original-user bootstrap replay differs")
    out = Path("/artifacts") / ("supplementary-output-audit-" + digest[:16]) / f"{protocol}-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    result = dict(
        status="complete",
        source_digest=digest,
        scored_source_digest=SCORE_SOURCE,
        scored_report_sha256=report_sha256,
        protocol=protocol,
        seed=seed,
        case_count=case_count,
        output_hashes_checked=outputs,
        proper_loss_cells_checked=proper_cells,
        model_endpoint_blocks=len(merged),
        all_category_and_top_calibration_bins_checked=True,
        bootstrap_contrasts_checked=bootstrap_cells,
        maximum_proper_loss_difference=maximum_loss,
        maximum_calibration_difference=maximum_calibration,
        maximum_private_user_loss_difference=maximum_user_loss,
        maximum_bootstrap_difference=maximum_ci,
        tolerance=1e-10,
        wall_seconds=time.monotonic() - started,
        scope="Independent saved-forecast reconstruction and sorted-user categorical proper losses, all reported class/top calibration cells, private user statistics and all paired NLL bootstrap contrasts. No neural-forward/state-assignment recomputation. Interval compatibility diagnostics and uniform-envelope arithmetic are not independently replayed by this audit.",
    )
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(protocol: str = "known_users", seed: int = 20261015, matrix: bool = False):
    import hashlib
    import json
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools"))
    from behavioral_equivalence.budget import BudgetLedger
    from run_final_human_jobs import launcher_lock

    source, digest = snapshot()
    cfg = json.loads((ROOT / "configs/human-supplementary-final-v1.json").read_text())
    all_jobs = [(p["protocol"], p["seed"]) for p in cfg["probes"]]
    jobs = all_jobs if matrix else [(protocol, seed)]
    if any(job not in all_jobs for job in jobs):
        raise ValueError("Fixed supplementary fit identities required")
    reports = {}
    for p, s in jobs:
        path = ROOT / "reports/supplementary-final" / f"{p}-{s}.json"
        raw = path.read_bytes()
        report = json.loads(raw)
        if (
            report["status"] != "complete"
            or report["source_digest"] != SCORE_SOURCE
            or report["protocol"] != p
            or report["seed"] != s
        ):
            raise ValueError("Complete frozen scoring reports required before output audit")
        reports[(p, s)] = hashlib.sha256(raw).hexdigest()
    ledger = BudgetLedger(ROOT / ".campaign/calendar-quantization-budget.sqlite", 40)
    for p, s in all_jobs:
        reservation = ledger.reserve(f"supplementary-output-audit-{digest}-{p}-{s}", 0.10)
    print(
        json.dumps(
            dict(
                matrix_reserved_usd=1.0,
                total_reserved_usd=reservation["total_reserved_usd"],
                ceiling_usd=reservation["ceiling_usd"],
            )
        ),
        flush=True,
    )
    (
        ROOT / "reports/source-snapshots" / ("supplementary-output-audit-" + digest[:16] + ".json")
    ).write_text(json.dumps(source, indent=2) + "\n")
    stop = threading.Event()

    def run(job):
        if stop.is_set():
            return
        p, s = job
        state = ROOT / ".campaign" / f"supplementary-output-audit-{digest[:16]}-{p}-{s}.json"
        try:
            if state.exists():
                saved = json.loads(state.read_text())
                if saved["report_sha256"] != reports[job]:
                    raise ValueError("Never reattach audit handle to changed scored data")
                call = modal.FunctionCall.from_id(saved["function_call_id"])
            else:
                with state.with_suffix(".submission-attempt").open("x") as marker:
                    marker.write(digest + "\n")
                call = audit.spawn(p, s, reports[job])
                saved = dict(
                    source_digest=digest,
                    function_call_id=call.object_id,
                    protocol=p,
                    seed=s,
                    report_sha256=reports[job],
                )
                state.write_text(json.dumps(saved, indent=2))
            print(json.dumps(saved), flush=True)
            report = call.get()
            if (
                report["status"] != "complete"
                or report["source_digest"] != digest
                or report["protocol"] != p
                or report["seed"] != s
                or report["scored_report_sha256"] != reports[job]
            ):
                raise ValueError("Returned audit does not match its exact requested scored result")
            directory = ROOT / "reports/supplementary-output-audit"
            directory.mkdir(exist_ok=True)
            (directory / f"{p}-{s}.json").write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(report), flush=True)
        except Exception:
            stop.set()
            raise

    with launcher_lock(ROOT / ".campaign/supplementary-output-audit-launcher.lock"):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(run, jobs))
