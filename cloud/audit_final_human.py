"""Cloud-only hash and independent proper-loss replay for final human outputs."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-final-human-output-audit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3")
    .env({"PYTHONPATH": "/research:/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tools/final_result_audit.py", remote_path="/research/final_result_audit.py"
    )
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def snapshot():
    import hashlib
    import json

    base = Path("/research") if Path("/research/final_result_audit.py").exists() else ROOT
    helper = (
        base / "final_result_audit.py"
        if base == Path("/research")
        else base / "tools/final_result_audit.py"
    )
    saved = {"cloud.py": Path(__file__).read_text(), "helper.py": helper.read_text()}
    for name in [
        "horizon_archive.py",
        "horizon_targets.py",
        "nested_design.py",
        "property_bounds.py",
        "core.py",
        "integer_bounds.py",
        "uncertainty.py",
        "clustering.py",
    ]:
        saved[name] = (base / "src/behavioral_equivalence" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def audit(records):
    import hashlib
    import json
    import time
    import numpy as np
    from final_result_audit import loss_block, combine, check_metrics, bootstrap_primary
    from behavioral_equivalence.horizon_archive import restore_horizon_forecasts

    volume.reload()
    saved, digest = snapshot()
    started = time.monotonic()
    batch_digest = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    out = Path("/artifacts") / ("final-human-audit-" + digest[:16]) / batch_digest[:16]
    out.mkdir(parents=True, exist_ok=True)
    if (out / "report.json").exists():
        return json.loads((out / "report.json").read_text())
    results = []
    for record in records:
        root = Path("/artifacts") / record["remote_root"]
        if checksum(root / "report.json") != record["report_sha256"]:
            raise ValueError("Final report identity changed")
        report = json.loads((root / "report.json").read_text())
        source = json.loads((root / "source.json").read_text())
        if (
            hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()
            != report["source_digest"]
        ):
            raise ValueError("Final executable source identity changed")
        cfg = json.loads(source["config.json"])
        spec = next(
            s
            for s in cfg["probes"]
            if s["seed"] == report["seed"] and s["protocol"] == report["protocol"]
        )
        for name, sha in report["output_sha256"].items():
            if checksum(root / name) != sha:
                raise ValueError("Final saved output hash differs: " + name)
        key = record["phase"]
        blocks = {}
        ids_sha = hashlib.sha256()
        n = 0
        finite = None
        parameters = None
        frontier = None
        if key == "direct_horizon":
            for name in ["horizon_archive.py", "horizon_targets.py", "nested_design.py"]:
                if source["src/behavioral_equivalence/" + name] != saved[name]:
                    raise ValueError("Archive reconstruction differs from locked inference")
            with np.load(root / "finite-model-event-tables.npz", allow_pickle=False) as a:
                finite = {name: a[name] for name in a.files}
        if key.startswith("response_"):
            path = Path("/artifacts") / spec["frontier_root"]
            if checksum(path / "report.json") != spec["frontier_report_sha256"]:
                raise ValueError("Response frontier changed")
            frontier = json.loads((path / "report.json").read_text())
            if checksum(path / "models.npz") != frontier["output_sha256"]["models.npz"]:
                raise ValueError("Response probability table changed")
            with np.load(path / "models.npz", allow_pickle=False) as a:
                parameters = {name: a[name] for name in a.files}
        for name in sorted(report["output_sha256"]):
            if not name.startswith("bucket-"):
                continue
            with np.load(root / name, allow_pickle=False) as arrays:
                a = {k: arrays[k] for k in arrays.files}
            ids, users = a["case_id"], a["user_id"]
            n += len(ids)
            ids_sha.update(ids.astype("<u8").tobytes())
            if len(np.unique(ids)) != len(ids):
                raise ValueError("Duplicate original prediction cases")
            if key == "one_step":
                for model in a:
                    if model in ["case_id", "user_id", "target"]:
                        continue
                    blocks.setdefault("all/" + model, []).append(
                        loss_block(users, a["target"], a[model])
                    )
            elif key == "direct_horizon":
                predictions = restore_horizon_forecasts(a, finite, cfg["decision_horizons"])
                for j in range(len(report["events"])):
                    keep = a["targets"][:, j] >= 0
                    for model, p in predictions.items():
                        blocks.setdefault(str(j) + "/" + model, []).append(
                            loss_block(users[keep], a["targets"][keep, j], p[keep, j])
                        )
            else:
                heldout = key == "response_heldout_groups"
                coarse = parameters["fine_to_coarse"][a["fine_state"]]
                env = a["environment"]
                models = list(dict.fromkeys(row["name"] for row in frontier["records"]))
                for model in models:
                    if heldout:
                        p = np.empty((len(ids), 3))
                        for group in range(4):
                            keep = a["duration_group"] == group
                            eligible = [
                                fold for fold in cfg["folds"] if group in fold["heldout_groups"]
                            ]
                            if len(eligible) != 3:
                                raise ValueError("Wrong held-out copy count")
                            for col, fold in enumerate(eligible):
                                prefix = fold["name"] + "_" + model
                                p[keep, col] = parameters[prefix + "_abstract"][
                                    parameters[prefix + "_mapping"][coarse[keep]], env[keep]
                                ]
                    else:
                        p = parameters[model + "_abstract"][
                            parameters[model + "_mapping"][coarse], env
                        ]
                    blocks.setdefault("aggregate/" + model, []).append(
                        loss_block(users, a["target"], p)
                    )
                for model in ["markov", "context", "gru", "transformer"]:
                    p = (
                        np.repeat(a[model][:, None].astype(float), 3, axis=1)
                        if heldout
                        else a[model]
                    )
                    blocks.setdefault("aggregate/" + model, []).append(
                        loss_block(users, a["target"], p)
                    )
        if ids_sha.hexdigest() != report["ordered_case_id_sha256"] or n != report.get(
            "cases", report.get("origins")
        ):
            raise ValueError("Final original case sequence differs")
        merged = {name: combine(values) for name, values in blocks.items()}
        differences = {}
        for name, actual in merged.items():
            differences[name] = check_metrics(actual["metrics"], report["metrics"][name])
        if key == "one_step":
            primaries = [
                ("all", report["primary"], "parent_k256", report["primary"] + "-vs-parent")
            ]
        elif key.startswith("response_"):
            primaries = [
                (
                    "aggregate",
                    cfg["primary_candidate"],
                    cfg["primary_reference"],
                    cfg["primary_candidate"] + "-vs-reference",
                )
            ]
        else:
            nested = json.loads(
                (Path("/artifacts") / spec["nested_root"] / "report.json").read_text()
            )
            primary = "nested." + nested["primary"]
            primaries = [
                (str(j), primary, "parent_k256", primary + "-vs-parent")
                for j in range(len(report["events"]))
            ]
        replayed = []
        for cohort, candidate, reference, contrast in primaries:
            actual, baseline = merged[cohort + "/" + candidate], merged[cohort + "/" + reference]
            if not np.array_equal(actual["users"], baseline["users"]) or not np.array_equal(
                actual["counts"], baseline["counts"]
            ):
                raise ValueError("Replayed user pairing differs")
            expected = report["comparisons"][cohort]
            if expected.get("status") == "unsupported":
                continue
            expected = expected[contrast]
            result = bootstrap_primary(
                actual["nll"] - baseline["nll"],
                replications=expected["replications"],
                seed=expected["seed"],
            )
            if (
                abs(result["mean"] - expected["candidate_minus_reference_nats"]) > 1e-10
                or np.max(np.abs(np.asarray(result["ci95"]) - expected["ci95"])) > 1e-10
            ):
                raise ValueError("Explicit original-user bootstrap replay differs")
            replayed.append(
                dict(cohort=cohort, contrast=contrast, mean=result["mean"], ci95=result["ci95"])
            )
        results.append(
            dict(
                phase=key,
                protocol=report["protocol"],
                seed=report["seed"],
                embargo_hours=record["embargo_hours"],
                report_sha256=record["report_sha256"],
                files_hash_verified=len(report["output_sha256"]),
                proper_loss_cells_replayed=len(merged),
                primary_bootstraps_replayed=replayed,
                maximum_nll_metric_discrepancy=max(
                    (
                        value[k]
                        for value in differences.values()
                        for k in value
                        if k.startswith("nll")
                    ),
                    default=0,
                ),
                maximum_brier_metric_discrepancy=max(
                    (
                        value[k]
                        for value in differences.values()
                        for k in value
                        if k.startswith("brier")
                    ),
                    default=0,
                ),
            )
        )
        print(json.dumps(dict(phase=key, seed=report["seed"], audited=len(results))), flush=True)
    result = dict(
        status="complete",
        source_digest=digest,
        batch_digest=batch_digest,
        jobs=len(results),
        expected_full_program_jobs=50,
        results=results,
        wall_seconds=time.monotonic() - started,
        root=str(out),
        scope="All stored output hashes and all-case/aggregate proper losses replayed using separate sorted-user arithmetic; explicit original-user primary bootstrap replay. Secondary subgroup/per-fold calibration and neural forward recomputation are outside this audit. NLL tolerance1e-10, Brier1e-6 accounts for saved mixed floating precision. Not a human-generating-process guarantee or independent general theorem proof.",
    )
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    import json
    import hashlib
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    progress = json.loads((ROOT / "reports/preflight/final-human-job-progress.json").read_text())
    records = [r for r in progress["jobs"].values() if r["status"] == "complete"]
    if not records:
        raise ValueError("Completed final jobs required")
    digest = snapshot()[1]
    batch = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    reservation = BudgetLedger(ROOT / ".campaign/nested-human-budget.sqlite", 40).reserve(
        "final-audit-" + digest + "-" + batch, 0.18
    )
    state = ROOT / ".campaign" / ("final-audit-" + digest[:12] + "-" + batch[:12] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = audit.spawn(records)
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id,
                    source_digest=digest,
                    batch_digest=batch,
                    reservation=reservation,
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    result = call.get()
    (
        ROOT
        / "reports/preflight"
        / ("final-human-output-audit-" + digest[:12] + "-" + batch[:12] + ".json")
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)
