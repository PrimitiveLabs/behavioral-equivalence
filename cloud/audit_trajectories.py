"""Backward numerical replay of all sampled joint trajectory comparisons."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-trajectory-audit-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .env({"PYTHONPATH": "/research/tools", "OPENBLAS_NUM_THREADS": "4"})
)
for name in [
    "trajectory_backward_audit.py",
    "supplementary_result_audit.py",
    "final_result_audit.py",
]:
    image = image.add_local_file(ROOT / "tools" / name, remote_path="/research/tools/" + name)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def identity():
    import hashlib
    import json

    base = (
        Path("/research/tools")
        if Path("/research/tools/trajectory_backward_audit.py").exists()
        else ROOT / "tools"
    )
    source = {"cloud.py": Path(__file__).read_text()}
    for name in [
        "trajectory_backward_audit.py",
        "supplementary_result_audit.py",
        "final_result_audit.py",
    ]:
        source[name] = (base / name).read_text()
    return source, hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(8192, 8192),
    timeout=1800,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def audit(inputs):
    import json
    import time
    import numpy as np
    from trajectory_backward_audit import backward_nll
    from supplementary_result_audit import bootstrap_many

    volume.reload()
    started = time.monotonic()
    source, digest = identity()
    expected = {(p, s) for p in ["known_users", "unseen_users"] for s in range(20261015, 20261020)}
    if len(inputs) != 10 or {(r["protocol"], r["seed"]) for r in inputs} != expected:
        raise ValueError("Complete ten-fit trajectory matrix required")
    max_loss = max_metric = max_bootstrap = 0.0
    cells = hashes = contrasts = 0
    shared_ids = {}
    records = []
    for entry in inputs:
        root = Path(entry["artifact_root"])
        if checksum(root / "report.json") != entry["report_sha256"]:
            raise ValueError("Trajectory report changed")
        report = json.loads((root / "report.json").read_text())
        if (report["protocol"], report["seed"]) != (entry["protocol"], entry["seed"]):
            raise ValueError("Audit case mismatch")
        for name, sha in report["output_sha256"].items():
            if checksum(root / name) != sha:
                raise ValueError("Saved trajectory output changed")
            hashes += 1
        cfg = json.loads(json.loads((root / "source.json").read_text())["config.json"])
        probe = next(
            p
            for p in cfg["probes"]
            if (p["protocol"], p["seed"]) == (entry["protocol"], entry["seed"])
        )
        with np.load(root / "sampled-trajectories.npz", allow_pickle=False) as a:
            arrays = {k: a[k] for k in a.files}
        with np.load(root / "joint-loss-prefixes.npz", allow_pickle=False) as a:
            saved = {k: a[k] for k in a.files}
        ids = np.sort(arrays["case_id"])
        if entry["protocol"] in shared_ids and not np.array_equal(
            ids, shared_ids[entry["protocol"]]
        ):
            raise ValueError("Fits used different sampled cases")
        shared_ids[entry["protocol"]] = ids
        nested = Path("/artifacts") / probe["nested_root"]
        if checksum(nested / "report.json") != probe["nested_report_sha256"]:
            raise ValueError("Nested parent changed")
        nr = json.loads((nested / "report.json").read_text())
        parent = Path("/artifacts") / nr["parent_root"]
        if checksum(parent / "report.json") != nr["parent_report_sha256"]:
            raise ValueError("Parent report changed")
        pr = json.loads((parent / "report.json").read_text())
        if checksum(parent / "k256-parent.npz") != pr["output_sha256"]["k256-parent.npz"]:
            raise ValueError("Parent arrays changed")
        with np.load(parent / "k256-parent.npz", allow_pickle=False) as a:
            models = {"parent_k256": (a["transitions"], a["emissions"], np.arange(256))}
        for name in saved:
            if name == "parent_k256" or name.endswith(".frozen_origin_iid"):
                continue
            row = next(r for r in nr["records"] if r["name"] == name)
            filename = name + "-model.npz"
            if checksum(nested / filename) != row["output_sha256"][filename]:
                raise ValueError("Quotient arrays changed")
            with np.load(nested / filename, allow_pickle=False) as a:
                p = a["transitions"]
                k = len(p) // 3
                e = np.array([[p[3 * s, 3 * s + 1], p[3 * s, 3 * s + 2]] for s in range(k)])
                t = np.array([[p[3 * s + 1, ::3], p[3 * s + 2, ::3]] for s in range(k)])
                models[name] = (t, e, a["latent_mapping"])
        for column, h in enumerate(cfg["decision_horizons"]):
            keep = arrays["observed"][:, column]
            y = arrays["outcomes"][keep, :h]
            users = arrays["user_id"][keep]
            order = np.argsort(users, kind="stable")
            ordered = users[order]
            starts = (
                np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
                if len(users)
                else np.empty(0, dtype=int)
            )
            counts = np.diff(np.r_[starts, len(users)]) if len(users) else np.empty(0, dtype=int)
            means = {}
            for name, stored in saved.items():
                if name in models:
                    t, e, mapping = models[name]
                    initial = mapping[arrays["state"][keep]]
                    values = (
                        np.concatenate(
                            [
                                backward_nll(t, e, initial[s : s + 256], y[s : s + 256])
                                for s in range(0, len(y), 256)
                            ]
                        )
                        if len(y)
                        else np.empty(0)
                    )
                else:
                    arch = name.split(".")[0]
                    p = arrays[arch][keep]
                    ones = y.sum(axis=1)
                    zeros = h - ones
                    values = np.zeros(len(y))
                    with np.errstate(divide="ignore"):
                        a = ones > 0
                        b = zeros > 0
                        values[a] -= ones[a] * np.log(p[a])
                        values[b] -= zeros[b] * np.log1p(-p[b])
                original = stored[keep, h - 1]
                if not np.array_equal(np.isinf(values), np.isinf(original)):
                    raise ValueError("Zero-support likelihood differs")
                finite = np.isfinite(values)
                difference = float(np.max(np.abs(values[finite] - original[finite]), initial=0))
                if difference > 1e-10:
                    raise ValueError("Backward joint likelihood differs")
                max_loss = max(max_loss, difference)
                per_user = (
                    np.add.reduceat(values[order] / h, starts) / counts
                    if len(users)
                    else np.empty(0)
                )
                target = report["metrics"][f"{h}/{name}"]
                failures = int(np.isinf(values).sum())
                if (
                    target["cases"] != len(y)
                    or target["users"] != len(starts)
                    or target["zero_probability_sequences"] != failures
                ):
                    raise ValueError("Metric support differs")
                checks = {
                    "joint_nll_equal_user_mean": float(per_user.mean() * h)
                    if len(starts) and not failures
                    else None,
                    "nll_per_decision_equal_user_mean": float(per_user.mean())
                    if len(starts) and not failures
                    else None,
                    "joint_nll_case_mean": float(values.mean())
                    if len(y) and not failures
                    else None,
                }
                for key, value in checks.items():
                    if value is None:
                        if target[key] is not None:
                            raise ValueError("Infinite or empty metric differs")
                    else:
                        delta = abs(value - target[key])
                        max_metric = max(max_metric, delta)
                        if delta > 1e-10:
                            raise ValueError("Independent user loss differs")
                if np.isfinite(per_user).all():
                    means[name] = per_user
                cells += 1
            comparisons = report["comparisons"][str(h)]
            if "status" not in comparisons:
                labels = list(comparisons)
                differences = np.column_stack(
                    [
                        means[comparisons[n]["candidate"]] - means[comparisons[n]["reference"]]
                        for n in labels
                    ]
                )
                ci = bootstrap_many(differences, replications=2000, seed=cfg["sample_seed"])
                for j, label in enumerate(labels):
                    target = comparisons[label]
                    delta = max(
                        abs(
                            float(differences[:, j].mean())
                            - target["candidate_minus_reference_nats"]
                        ),
                        float(np.max(np.abs(ci[:, j] - target["ci95"]))),
                    )
                    max_bootstrap = max(max_bootstrap, delta)
                    if delta > 1e-10:
                        raise ValueError("Independent user bootstrap differs")
                    contrasts += 1
        records.append(dict(protocol=entry["protocol"], seed=entry["seed"], status="passed"))
        print(json.dumps(records[-1]), flush=True)
    out = Path("/artifacts") / ("trajectory-output-audit-" + digest[:16])
    out.mkdir(exist_ok=True)
    report = dict(
        status="complete",
        source_digest=digest,
        inputs=inputs,
        records=records,
        hashes_verified=hashes,
        metric_cells=cells,
        bootstrap_contrasts=contrasts,
        maximum_likelihood_difference=max_loss,
        maximum_metric_difference=max_metric,
        maximum_bootstrap_difference=max_bootstrap,
        wall_seconds=time.monotonic() - started,
        scope="All fully observed sampled sequences, all saved models/horizons, backward versus forward likelihood, sorted-user losses and explicit-index user bootstrap. Same sampled IDs across fits verified; outcome-blind selection rule and raw representative construction not independently re-executed here.",
    )
    (out / "source.json").write_text(json.dumps(source, sort_keys=True))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    import json
    import hashlib
    import sys
    import fcntl

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    inputs = []
    for protocol in ["known_users", "unseen_users"]:
        for seed in range(20261015, 20261020):
            raw = (ROOT / "reports/trajectory-final" / f"{protocol}-{seed}.json").read_bytes()
            report = json.loads(raw)
            if (
                report["status"] != "complete"
                or report["protocol"] != protocol
                or report["seed"] != seed
            ):
                raise ValueError("All ten trajectory scoring reports required before audit")
            inputs.append(
                dict(
                    protocol=protocol,
                    seed=seed,
                    artifact_root=report["artifact_root"],
                    report_sha256=hashlib.sha256(raw).hexdigest(),
                )
            )
    source, digest = identity()
    batch = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    lock = (ROOT / ".campaign/trajectory-audit.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reservation = BudgetLedger(ROOT / ".campaign/calibrated-parent-budget.sqlite", 25).reserve(
        "trajectory-audit-" + digest + "-" + batch, 0.30
    )
    path = ROOT / ".campaign" / ("trajectory-audit-" + digest[:16] + "-" + batch[:16] + ".json")
    if path.exists():
        call = modal.FunctionCall.from_id(json.loads(path.read_text())["function_call_id"])
    else:
        with path.with_suffix(".submission-attempt").open("x") as marker:
            marker.write(digest + "\n")
        call = audit.spawn(inputs)
        path.write_text(
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
    print(path.read_text(), flush=True)
    result = call.get()
    (ROOT / "reports/preflight" / ("trajectory-output-audit-" + digest[:16] + ".json")).write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result), flush=True)
