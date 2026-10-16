"""Calibrate supplementary endpoint emissions using existing audited GRU states."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-supplementary-calibration-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.1")
    .env({"PYTHONPATH": "/research/src:/research/tools", "OPENBLAS_NUM_THREADS": "2"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "configs/human-supplementary-calibration-v1.json",
        remote_path="/research/config.json",
    )
)
for name in [
    "supplementary_outcomes.py",
    "supplementary_counts.py",
    "supplementary_abstraction.py",
]:
    image = image.add_local_file(ROOT / "tools" / name, remote_path="/research/tools/" + name)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def snapshot():
    import hashlib
    import json

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    config = (
        base / "config.json"
        if base == Path("/research")
        else base / "configs/human-supplementary-calibration-v1.json"
    )
    saved = {"cloud.py": Path(__file__).read_text(), "config.json": config.read_text()}
    for name in [
        "supplementary_outcomes.py",
        "supplementary_counts.py",
        "supplementary_abstraction.py",
    ]:
        saved["tools/" + name] = (base / "tools" / name).read_text()
    for path in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + path.name] = path.read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=1800,
    retries=0,
    max_containers=2,
    volumes={"/artifacts": volume},
)
def calibrate(protocol, seed):
    import hashlib
    import json
    import time
    from fractions import Fraction
    import numpy as np
    from behavioral_equivalence.cluster_ratio import cluster_ratio_from_totals
    from behavioral_equivalence.nested_design import latent_partition
    from behavioral_equivalence.integer_bounds import verify_abstraction
    from supplementary_outcomes import categorical_targets
    from supplementary_counts import endpoint_counts, smoothed_emissions
    from supplementary_abstraction import certify_endpoint

    volume.reload()
    started = time.monotonic()
    saved, digest = snapshot()
    cfg = json.loads(saved["config.json"])
    parent = next(p for p in cfg["parents"] if p["protocol"] == protocol and p["seed"] == seed)
    data = Path("/artifacts") / cfg["calibration_root"]
    targets = Path("/artifacts") / cfg["target_root"]
    src = Path("/artifacts") / parent["root"]
    if (
        checksum(data / "report.json") != cfg["calibration_report_sha256"]
        or checksum(src / "report.json") != parent["report_sha256"]
    ):
        raise ValueError("Audited calibration or frozen state parent changed")
    cr = json.loads((data / "report.json").read_text())
    tr = json.loads((targets / "report.json").read_text())
    pr = json.loads((src / "report.json").read_text())
    if (
        hashlib.sha256(json.dumps(tr, sort_keys=True).encode()).hexdigest()
        != cfg["target_report_canonical_digest"]
    ):
        raise ValueError("Supplementary calibration target manifest changed")
    if (
        any(r["status"] != "complete" for r in [cr, tr, pr])
        or len(cr["buckets"]) != 32
        or len(tr["buckets"]) != 32
    ):
        raise ValueError("Complete calibration inputs required")
    quant = src.parent
    if checksum(quant / "report.json") != pr["quantization_report_sha256"]:
        raise ValueError("Training quantizer report changed")
    qr = json.loads((quant / "report.json").read_text())
    center_path = quant / "k256-model.npz"
    if checksum(center_path) != qr["output_sha256"]["k256-model.npz"]:
        raise ValueError("Training parent features changed")
    with np.load(center_path, allow_pickle=False) as a:
        features = a["centers"]
    names = list(cfg["endpoint_classes"])
    classes = list(cfg["endpoint_classes"].values())
    counts = {
        (k, j): np.zeros((k, c), dtype=np.int64)
        for k in cfg["state_counts"]
        for j, c in enumerate(classes)
    }
    users_seen = set()
    maximum = 0
    origin_sha = hashlib.sha256()
    origins = 0
    for bucket in range(32):
        raw = data / f"bucket-{bucket:02d}" / "calibration.npz"
        state = src / f"bucket-{bucket:02d}-states.npz"
        target = targets / f"bucket-{bucket:02d}.npz"
        if (
            checksum(raw) != cr["buckets"][bucket]["output_sha256"]["calibration.npz"]
            or checksum(state) != pr["output_sha256"][state.name]
            or checksum(target) != tr["buckets"][bucket]["output_sha256"]
        ):
            raise ValueError("Calibration rows, frozen states or new target arrays changed")
        with (
            np.load(raw, allow_pickle=False) as a,
            np.load(state, allow_pickle=False) as z,
            np.load(target, allow_pickle=False) as t,
        ):
            chosen = z["source_query_indices"]
            current = z["pair_current"]
            rows = chosen[current]
            if (
                not np.array_equal(a["case_id"], t["case_id"])
                or not np.array_equal(z["case_id"], a["case_id"][chosen])
                or not np.all(a["role_code"][rows] == 2)
                or (protocol == "unseen_users" and not np.all(a["user_role_code"][rows] == 2))
            ):
                raise ValueError("Origin/label/role identity changed")
            indicators = {
                name: t[name][rows]
                for name in [
                    "is_like",
                    "is_follow",
                    "is_comment",
                    "is_forward",
                    "is_hate",
                    *[f"watch_category_{k}" for k in range(4)],
                ]
            }
            y = categorical_targets(indicators)
            addition, users, cap = endpoint_counts(
                {k: z[f"k{k}_state"][current] for k in cfg["state_counts"]},
                y,
                a["user_id"][rows],
                a["case_id"][rows],
                cfg["maximum_per_user"],
            )
            if set(map(int, users)) & users_seen:
                raise ValueError("User appears in multiple prepared buckets")
            users_seen.update(map(int, users))
            maximum = max(maximum, cap)
            origins += len(rows)
            origin_sha.update(a["case_id"][rows].astype("<u8").tobytes())
            for key, value in addition.items():
                counts[key] += value
    if len(users_seen) != pr["users"]:
        raise ValueError("Independent-user denominator differs from audited parent")
    arrays = {}
    intervals = []
    for (k, j), table in counts.items():
        lower, upper, certificate = cluster_ratio_from_totals(
            table,
            users=len(users_seen),
            maximum_per_user=cfg["maximum_per_user"],
            alpha=Fraction(*cfg["interval_alpha"]),
        )
        prefix = f"k{k}_endpoint{j}_"
        arrays.update(
            {
                prefix + "counts": table,
                prefix + "emissions": smoothed_emissions(
                    table, cfg["point_emission_smoothing_total"]
                ),
                prefix + "lower": lower,
                prefix + "upper": upper,
            }
        )
        intervals.append(
            dict(
                states=k,
                endpoint=names[j],
                certificate=certificate,
                observed_labels=int(table.sum()),
                mean_width=float((upper - lower).mean()),
                vacuous_coordinates=int(np.sum((lower == 0) & (upper == 1))),
            )
        )
    out = Path("/artifacts") / ("supplementary-calibration-" + digest[:16]) / f"{protocol}-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    records = []
    abstractions = []
    starts = {k: latent_partition(features, k, seed) for k in cfg["parent_start_counts"]}
    for j, endpoint in enumerate(names):
        emissions = arrays[f"k256_endpoint{j}_emissions"]
        for start, blocks in starts.items():
            baseline = certify_endpoint(emissions, blocks, 1.0)
            designs = [("feature_unconstrained", 1.0, baseline)]
            for epsilon in cfg["epsilons"]:
                fitted = certify_endpoint(emissions, blocks, epsilon)
                designs.append(("refined", epsilon, fitted))
                random_blocks = latent_partition(features, len(fitted["blocks"]), seed, random=True)
                designs.append(
                    ("random_matched", epsilon, certify_endpoint(emissions, random_blocks, 1.0))
                )
            for method, epsilon, result in designs:
                verify_abstraction(result["parent"], result["abstract"], result["certificate"])
                name = f"{endpoint}-start{start}-{method}-epsilon{epsilon}"
                arrays[name + "_probabilities"] = result["probabilities"]
                abstractions.append(
                    dict(
                        name=name,
                        endpoint=endpoint,
                        method=method,
                        requested_epsilon=epsilon,
                        blocks=result["blocks"],
                        certificate=result["certificate"],
                        trace=result["trace"],
                        scope=result["scope"],
                    )
                )
                records.append(
                    dict(
                        name=name,
                        endpoint=endpoint,
                        method=method,
                        start_count=start,
                        requested_epsilon=epsilon,
                        reduced_latent_states=len(result["blocks"]),
                        actual_radius=result["trace"][-1]["radius"],
                    )
                )
    np.savez_compressed(out / "models.npz", **arrays)
    (out / "abstractions.json").write_text(json.dumps(abstractions))
    (out / "intervals.json").write_text(json.dumps(intervals))
    report = dict(
        status="complete",
        source_digest=digest,
        protocol=protocol,
        seed=seed,
        parent=parent,
        users=len(users_seen),
        origins=origins,
        maximum_observed_origins_per_user=maximum,
        origin_id_sha256=origin_sha.hexdigest(),
        models=intervals,
        abstraction_records=records,
        certificates_replayed=len(abstractions),
        output_sha256={
            name: checksum(out / name)
            for name in ["models.npz", "abstractions.json", "intervals.json"]
        },
        wall_seconds=time.monotonic() - started,
        root=str(out),
        human_coverage_established=False,
        scope=cfg["certificate_scope"],
        uncertainty_scope=cfg["alpha_scope"],
    )
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return dict(status="complete", root=str(out), wall_seconds=report["wall_seconds"])


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

    source, digest = snapshot()
    cfg = json.loads(source["config.json"])
    if cfg["status"] != "calibration_design_frozen_before_emission_estimation":
        raise ValueError("Frozen calibration design required")
    all_jobs = [(p["protocol"], p["seed"]) for p in cfg["parents"]]
    if (protocol, seed) not in all_jobs:
        raise ValueError("Fixed parent matrix required")
    ledger = BudgetLedger(ROOT / ".campaign/calibrated-parent-budget.sqlite", 25)
    for p, s in all_jobs:
        reservation = ledger.reserve(f"supplementary-calibration-{digest}-{p}-{s}", 0.15)
    print(
        json.dumps(
            dict(
                matrix_reserved_usd=1.5,
                total_reserved_usd=reservation["total_reserved_usd"],
                ceiling_usd=reservation["ceiling_usd"],
            )
        ),
        flush=True,
    )
    (
        ROOT / "reports/source-snapshots" / ("supplementary-calibration-" + digest[:16] + ".json")
    ).write_text(json.dumps(source, indent=2) + "\n")
    stop = threading.Event()

    def run(job):
        if stop.is_set():
            return
        p, s = job
        state = ROOT / ".campaign" / f"supplementary-calibration-{digest[:16]}-{p}-{s}.json"
        try:
            if state.exists():
                saved = json.loads(state.read_text())
                call = modal.FunctionCall.from_id(saved["function_call_id"])
            else:
                with state.with_suffix(".submission-attempt").open("x") as marker:
                    marker.write(digest + "\n")
                call = calibrate.spawn(p, s)
                saved = dict(
                    source_digest=digest, function_call_id=call.object_id, protocol=p, seed=s
                )
                state.write_text(json.dumps(saved, indent=2))
            print(json.dumps(saved), flush=True)
            result = call.get()
            content = b"".join(
                volume.read_file(result["root"].removeprefix("/artifacts/") + "/report.json")
            )
            report = json.loads(content)
            if (
                report["source_digest"] != digest
                or report["protocol"] != p
                or report["seed"] != s
                or report["status"] != "complete"
            ):
                raise ValueError("Calibration report identity mismatch")
            directory = ROOT / "reports/supplementary-calibration"
            directory.mkdir(exist_ok=True)
            (directory / f"{p}-{s}.json").write_bytes(content)
            print(
                json.dumps(
                    dict(
                        protocol=p,
                        seed=s,
                        status="complete",
                        certificates_replayed=report["certificates_replayed"],
                        wall_seconds=report["wall_seconds"],
                    )
                ),
                flush=True,
            )
        except Exception:
            stop.set()
            raise

    with launcher_lock(ROOT / ".campaign/supplementary-calibration-launcher.lock"):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(run, all_jobs if matrix else [(protocol, seed)]))
