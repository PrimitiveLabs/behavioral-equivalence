"""Audited per-user cap and pooled count-model intervals, without human test access."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-human-cluster-uncertainty-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "configs/human-cluster-uncertainty-v1.json", remote_path="/research/config.json"
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
    import json
    import hashlib

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    config = (
        base / "config.json"
        if base == Path("/research")
        else base / "configs/human-cluster-uncertainty-v1.json"
    )
    saved = {"cloud.py": Path(__file__).read_text(), "config.json": config.read_text()}
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
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
def run():
    import json
    import time
    from fractions import Fraction
    import numpy as np
    from behavioral_equivalence.cluster_ratio import pooled_decision_intervals
    from behavioral_equivalence.latent_estimation import adjacent_transition_counts
    from behavioral_equivalence.integer_bounds import robust_enclosures
    from behavioral_equivalence.decision_model import decision_horizon

    volume.reload()
    saved, digest = snapshot()
    cfg = json.loads(saved["config.json"])
    root = Path("/artifacts") / ("human-cluster-uncertainty-" + digest[:16])
    root.mkdir(exist_ok=True)
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    data = Path("/artifacts") / cfg["calibration_root"]
    if checksum(data / "report.json") != cfg["calibration_report_sha256"]:
        raise ValueError("Calibration identity changed")
    calibration = json.loads((data / "report.json").read_text())
    records = []
    started = time.monotonic()
    for parent in cfg["parents"]:
        src = Path("/artifacts") / parent["root"]
        if checksum(src / "report.json") != parent["report_sha256"]:
            raise ValueError("Parent identity changed")
        pr = json.loads((src / "report.json").read_text())
        out = root / f"{parent['protocol']}-{parent['seed']}"
        out.mkdir(exist_ok=True)
        if (out / "report.json").exists():
            cached = json.loads((out / "report.json").read_text())
            if cached["source_digest"] != digest or any(
                checksum(out / n) != h for n, h in cached["output_sha256"].items()
            ):
                raise ValueError("Cached uncertainty artifact changed")
            records.append(cached)
            continue
        counts = {k: np.zeros((k, 2, k), dtype=np.int64) for k in cfg["state_counts"]}
        users_seen = set()
        maximum = 0
        for bucket in calibration["buckets"]:
            name = f"bucket-{bucket['bucket']:02d}"
            raw = data / name / "calibration.npz"
            states = src / (name + "-states.npz")
            if (
                checksum(raw) != bucket["output_sha256"]["calibration.npz"]
                or checksum(states) != pr["output_sha256"][states.name]
            ):
                raise ValueError("Calibration rows or fitted states changed")
            with np.load(raw, allow_pickle=False) as z:
                a = {n: z[n] for n in ["user_id", "target", "bundle_index", "case_id", "role_code"]}
            with np.load(states, allow_pickle=False) as z:
                chosen, current, following = (
                    z["source_query_indices"],
                    z["pair_current"],
                    z["pair_next"],
                )
                if not np.array_equal(z["case_id"], a["case_id"][chosen]) or not np.all(
                    a["role_code"][chosen] == 2
                ):
                    raise ValueError("Query identity or calibration role changed")
                users = a["user_id"][chosen]
                unique, frequency = np.unique(users[current], return_counts=True)
                if set(map(int, unique)) & users_seen or np.any(
                    frequency > cfg["maximum_per_user"]
                ):
                    raise ValueError("Duplicate user buckets or cap violation")
                users_seen.update(map(int, unique))
                maximum = max(maximum, int(frequency.max(initial=0)))
                for k in counts:
                    counts[k] += adjacent_transition_counts(
                        z[f"k{k}_state"],
                        a["target"][chosen],
                        users,
                        a["bundle_index"][chosen],
                        current,
                        following,
                        k,
                    )
        if len(users_seen) != pr["users"]:
            raise ValueError("User denominator changed")
        summaries, outputs = [], {}
        for k, table in counts.items():
            with np.load(src / f"k{k}-parent.npz", allow_pickle=False) as z:
                if checksum(src / f"k{k}-parent.npz") != pr["output_sha256"][
                    f"k{k}-parent.npz"
                ] or not np.array_equal(table, z["counts"]):
                    raise ValueError("Independent count reconstruction differs")
            model, cert = pooled_decision_intervals(
                table,
                users=len(users_seen),
                maximum_per_user=cfg["maximum_per_user"],
                alpha=Fraction(*cfg["alpha"]),
            )
            name = f"k{k}-intervals.npz"
            np.savez_compressed(out / name, lower=model.lower, upper=model.upper, counts=table)
            outputs[name] = checksum(out / name)
            cert["positive_reachability"] = robust_enclosures(
                model, "positive", [decision_horizon(h) for h in cfg["horizons"]]
            )
            name = f"k{k}-certificate.json"
            (out / name).write_text(json.dumps(cert, indent=2))
            outputs[name] = checksum(out / name)
            widths = model.upper[::3, 2::3].diagonal() - model.lower[::3, 2::3].diagonal()
            summaries.append(
                dict(
                    states=k,
                    mean_emission_width=float(widths.mean()),
                    vacuous_emission_rows=int(np.count_nonzero(widths == 1)),
                )
            )
        report = dict(
            status="complete",
            source_digest=digest,
            parent=parent,
            users=len(users_seen),
            maximum_observed_origins_per_user=maximum,
            models=summaries,
            output_sha256=outputs,
            root=str(out),
            human_multistep_coverage_established=False,
            final_test_access=False,
            scope=cfg["scope"],
        )
        (out / "report.json").write_text(json.dumps(report, indent=2))
        records.append(report)
        (root / "progress.json").write_text(
            json.dumps(dict(complete=len(records), expected=len(cfg["parents"])))
        )
        volume.commit()
    result = dict(
        status="complete",
        source_digest=digest,
        records=records,
        root=str(root),
        wall_seconds=time.monotonic() - started,
    )
    (root / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    saved, digest = snapshot()
    reservation = BudgetLedger(ROOT / ".campaign/uncertainty-budget.sqlite", 8).reserve(
        "human-cluster-" + digest, 0.5
    )
    state = ROOT / ".campaign" / ("human-cluster-" + digest[:16] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = run.spawn()
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id, source_digest=digest, reservation=reservation
                ),
                indent=2,
            )
        )
        (ROOT / "reports/source-snapshots" / ("human-cluster-" + digest[:16] + ".json")).write_text(
            json.dumps(saved, indent=2)
        )
    print(state.read_text(), flush=True)
    print(json.dumps(call.get(), indent=2), flush=True)
