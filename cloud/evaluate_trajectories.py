"""Exploratory joint trajectories using existing origin-only state assignments."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-joint-trajectories-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "pydantic==2.11.7")
    .env({"PYTHONPATH": "/research/tools:/research/src", "OPENBLAS_NUM_THREADS": "4"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "configs/human-trajectory-final-v1.json", remote_path="/research/config.json"
    )
)
for name in ["trajectory_likelihood.py", "trajectory_design.py"]:
    image = image.add_local_file(ROOT / "tools" / name, remote_path="/research/tools/" + name)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def identity():
    import json
    import hashlib

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    source = {
        "cloud.py": Path(__file__).read_text(),
        "config.json": (
            base / ("config.json" if base != ROOT else "configs/human-trajectory-final-v1.json")
        ).read_text(),
    }
    for name in ["trajectory_likelihood.py", "trajectory_design.py"]:
        source[name] = (base / "tools" / name).read_text()
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        source["src/" + p.name] = p.read_text()
    return source, hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(4, 4),
    memory=(8192, 8192),
    timeout=1800,
    max_containers=2,
    retries=0,
    volumes={"/artifacts": volume},
)
def evaluate(protocol, seed):
    import json
    import time
    import numpy as np
    from trajectory_design import sampled_case_ids, decision_arrays, user_sequence_losses
    from trajectory_likelihood import sequence_nll
    from behavioral_equivalence.streaming_scores import shared_user_bootstrap

    started = time.monotonic()
    volume.reload()
    source, digest = identity()
    cfg = json.loads(source["config.json"])
    if cfg["status"] != "execution_frozen":
        raise ValueError("Frozen supplementary design required")
    probe = next(p for p in cfg["probes"] if (p["protocol"], p["seed"]) == (protocol, seed))
    out = Path("/artifacts") / ("joint-trajectories-" + digest[:16]) / f"{protocol}-{seed}"
    if (out / "report.json").exists():
        report = json.loads((out / "report.json").read_text())
        if report["source_digest"] != digest or any(
            checksum(out / n) != h for n, h in report["output_sha256"].items()
        ):
            raise ValueError("Cached trajectory artifacts changed")
        return report
    data = Path(cfg["sequence_root"])
    if checksum(data / "report.json") != cfg["sequence_report_sha256"]:
        raise ValueError("Sequence manifest changed")
    sr = json.loads((data / "report.json").read_text())
    candidates = []
    for row in sr["buckets"]:
        path = data / row["file"]
        if checksum(path) != row["sha256"]:
            raise ValueError("Reconstructed sequence changed")
        with np.load(path, allow_pickle=False) as a:
            keep = (
                np.ones(len(a["case_id"]), dtype=bool)
                if protocol == "known_users"
                else a["user_role_code"] == 3
            )
            candidates.append(a["case_id"][keep])
    sampled = sampled_case_ids(
        np.concatenate(candidates), cfg["sample_limit_per_protocol"], cfg["sample_seed"]
    )
    nested = Path("/artifacts") / probe["nested_root"]
    if checksum(nested / "report.json") != probe["nested_report_sha256"]:
        raise ValueError("Nested model identity changed")
    nr = json.loads((nested / "report.json").read_text())
    parent = Path("/artifacts") / nr["parent_root"]
    if checksum(parent / "report.json") != nr["parent_report_sha256"]:
        raise ValueError("Parent identity changed")
    pr = json.loads((parent / "report.json").read_text())
    if checksum(parent / "k256-parent.npz") != pr["output_sha256"]["k256-parent.npz"]:
        raise ValueError("Parent arrays changed")
    with np.load(parent / "k256-parent.npz", allow_pickle=False) as a:
        models = {"parent_k256": (a["transitions"], a["emissions"], np.arange(256))}
    records = {r["name"]: r for r in nr["records"]}

    def load_model(name):
        row = records[name]
        filename = name + "-model.npz"
        if checksum(nested / filename) != row["output_sha256"][filename]:
            raise ValueError("Quotient arrays changed")
        with np.load(nested / filename, allow_pickle=False) as a:
            t, e = decision_arrays(a["transitions"])
            return t, e, a["latent_mapping"]

    finite_name = "finite-k64-b0p01"
    models[finite_name] = load_model(finite_name)
    models["strict-k64-b0p01"] = load_model("strict-k64-b0p01")
    capacity = len(models[finite_name][1])
    for family in ["random", "embedding", "predictive"]:
        name = f"control-{family}-k{capacity}"
        models[name] = load_model(name)
    forecasts = probe["forecast_report"]
    root = Path(forecasts["root"])
    if checksum(root / "report.json") != probe["forecast_report_sha256"]:
        raise ValueError("Final forecast report changed")
    collected = {key: [] for key in ["case_id", "user_id", "outcomes", "observed", "state"]}
    for arch in ["context", "gru", "transformer"]:
        collected[arch] = []
    for row in sr["buckets"]:
        with np.load(data / row["file"], allow_pickle=False) as a:
            keep = np.isin(a["case_id"], sampled)
            if not keep.any():
                continue
            ids = a["case_id"][keep]
            for key in ["case_id", "user_id", "outcomes", "observed"]:
                collected[key].append(a[key][keep])
        filename = f"bucket-{row['bucket']:02d}-test-forecasts.npz"
        if checksum(root / filename) != forecasts["output_sha256"][filename]:
            raise ValueError("Stored origin predictions changed")
        with np.load(root / filename, allow_pickle=False) as a:
            raw_ids = a["case_id"]
            order = np.argsort(raw_ids)
            pos = np.searchsorted(raw_ids[order], ids)
            if np.any(pos >= len(order)) or not np.array_equal(raw_ids[order[pos]], ids):
                raise ValueError("Sampled origins not in frozen forecast cohort")
            ix = order[pos]
            if not np.array_equal(a["user_id"][ix], collected["user_id"][-1]):
                raise ValueError("Forecast user identity differs")
            collected["state"].append(a["state_k256"][ix])
            for arch in ["context", "gru", "transformer"]:
                collected[arch].append(a["iid_origin." + arch + ".iid"][ix])
    arrays = {k: np.concatenate(v) for k, v in collected.items()}
    if not np.array_equal(np.sort(arrays["case_id"]), sampled):
        raise ValueError("Complete sampled origin identity required")
    outcomes = np.maximum(arrays["outcomes"], 0)
    losses = {}
    for name, (t, e, mapping) in models.items():
        blocks = []
        for start in range(0, len(sampled), 256):
            sl = slice(start, start + 256)
            blocks.append(
                sequence_nll(t, e, mapping[arrays["state"][sl]], outcomes[sl], return_prefix=True)
            )
        losses[name] = np.concatenate(blocks)
        print(
            json.dumps(dict(protocol=protocol, seed=seed, model=name, sequences=len(sampled))),
            flush=True,
        )
    for arch in ["context", "gru", "transformer"]:
        p = arrays[arch]
        with np.errstate(divide="ignore"):
            terms = np.where(outcomes == 1, -np.log(p[:, None]), -np.log1p(-p[:, None]))
        losses[arch + ".frozen_origin_iid"] = np.cumsum(terms, axis=1)
    metrics, comparisons = {}, {}
    for column, h in enumerate(cfg["decision_horizons"]):
        keep = arrays["observed"][:, column]
        users = arrays["user_id"][keep]
        per_user = {}
        counts = None
        ids = None
        for name, values in losses.items():
            v = values[keep, h - 1]
            ids, counts, means = user_sequence_losses(users, v / h)
            failures = int(np.isinf(v).sum())
            metrics[f"{h}/{name}"] = dict(
                cases=len(v),
                users=len(ids),
                zero_probability_sequences=failures,
                joint_nll_equal_user_mean=float(means.mean() * h)
                if len(ids) and not failures
                else None,
                nll_per_decision_equal_user_mean=float(means.mean())
                if len(ids) and not failures
                else None,
                joint_nll_case_mean=float(v.mean()) if len(v) and not failures else None,
                status="infinite_nll" if failures else "evaluated" if len(ids) else "empty",
            )
            if np.isfinite(means).all():
                per_user[name] = means
        pairs = {
            name + "-vs-parent": (name, "parent_k256")
            for name in per_user
            if name != "parent_k256" and "parent_k256" in per_user
        }
        comparisons[str(h)] = (
            shared_user_bootstrap(
                ids,
                counts,
                per_user,
                pairs,
                replications=2000,
                seed=cfg["sample_seed"],
                margin=0.01,
            )
            if len(ids) >= 50 and int(keep.sum()) >= 2000 and pairs
            else {"status": "insufficient_finite_support"}
        )
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.json").write_text(json.dumps(source, sort_keys=True))
    np.savez_compressed(out / "sampled-trajectories.npz", **arrays)
    np.savez_compressed(out / "joint-loss-prefixes.npz", **losses)
    report = dict(
        status="complete",
        source_digest=digest,
        protocol=protocol,
        seed=seed,
        artifact_root=str(out),
        sampled_origins=len(sampled),
        sampled_users=len(np.unique(arrays["user_id"])),
        capacity_matched_states=capacity,
        metrics=metrics,
        comparisons=comparisons,
        censoring={
            str(h): int((~arrays["observed"][:, j]).sum())
            for j, h in enumerate(cfg["decision_horizons"])
        },
        output_sha256={p.name: checksum(p) for p in out.iterdir() if p.is_file()},
        wall_seconds=time.monotonic() - started,
        scope=cfg["scope"],
        numeric_policy="Unclipped sequence probabilities; infinite loss has null mean plus explicit failure count; comparisons only on fully finite models and shared supported original users.",
    )
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    volume.commit()
    return report


@app.local_entrypoint()
def main(protocol: str = "known_users", seed: int = 20261015, matrix: bool = False):
    import json
    import sys
    import threading
    from concurrent.futures import ThreadPoolExecutor

    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(ROOT / "tools"))
    from behavioral_equivalence.budget import BudgetLedger
    from run_final_human_jobs import launcher_lock

    source, digest = identity()
    cfg = json.loads(source["config.json"])
    jobs = [(p["protocol"], p["seed"]) for p in cfg["probes"]]
    if len(jobs) != 10 or len(set(jobs)) != 10 or (protocol, seed) not in jobs:
        raise ValueError("All10 fixed fitted-model cases required")
    ledger = BudgetLedger(ROOT / ".campaign/calibrated-parent-budget.sqlite", 25)
    for p, s in jobs:
        reservation = ledger.reserve(f"trajectory-final-{digest}-{p}-{s}", 0.30)
    print(
        json.dumps(
            dict(
                matrix_reserved_usd=3.0,
                total_reserved_usd=reservation["total_reserved_usd"],
                ceiling_usd=reservation["ceiling_usd"],
            )
        ),
        flush=True,
    )
    (ROOT / "reports/source-snapshots" / ("trajectory-final-" + digest[:16] + ".json")).write_text(
        json.dumps(source, indent=2) + "\n"
    )
    stop = threading.Event()

    def run(job):
        if stop.is_set():
            return
        p, s = job
        state = ROOT / ".campaign" / f"trajectory-final-{digest[:16]}-{p}-{s}.json"
        try:
            if state.exists():
                saved = json.loads(state.read_text())
                call = modal.FunctionCall.from_id(saved["function_call_id"])
            else:
                with state.with_suffix(".submission-attempt").open("x") as marker:
                    marker.write(digest + "\n")
                call = evaluate.spawn(p, s)
                saved = dict(
                    source_digest=digest, function_call_id=call.object_id, protocol=p, seed=s
                )
                state.write_text(json.dumps(saved, indent=2))
            print(json.dumps(saved), flush=True)
            result = call.get()
            content = json.dumps(result, indent=2, allow_nan=False).encode()
            report = json.loads(content)
            if (
                report["status"] != "complete"
                or report["source_digest"] != digest
                or report["protocol"] != p
                or report["seed"] != s
            ):
                raise ValueError("Trajectory final report identity mismatch")
            directory = ROOT / "reports/trajectory-final"
            directory.mkdir(exist_ok=True)
            (directory / f"{p}-{s}.json").write_bytes(content)
            print(
                json.dumps(
                    dict(
                        status="complete",
                        protocol=p,
                        seed=s,
                        sampled_origins=report["sampled_origins"],
                        wall_seconds=report["wall_seconds"],
                    )
                ),
                flush=True,
            )
        except Exception:
            stop.set()
            raise

    with launcher_lock(ROOT / ".campaign/trajectory-final-launcher.lock"):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(run, jobs if matrix else [(protocol, seed)]))
