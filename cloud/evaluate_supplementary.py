"""Complete supplementary human scoring, separated from locked click hypotheses."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-supplementary-final-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3")
    .env(
        {
            "PYTHONPATH": "/research/src:/research/tools",
            "OMP_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
        }
    )
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)
TOOLS = [
    "supplementary_heads.py",
    "supplementary_outcomes.py",
    "supplementary_metrics.py",
    "supplementary_intervals.py",
    "supplementary_counts.py",
]
for name in TOOLS:
    image = image.add_local_file(ROOT / "tools" / name, remote_path="/research/tools/" + name)
# The complete matrix freeze must exist before this image is submitted.
image = image.add_local_file(
    ROOT / "configs/human-supplementary-final-v1.json", remote_path="/research/config.json"
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    import hashlib
    import json

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    config = (
        base / "config.json"
        if base == Path("/research")
        else base / "configs/human-supplementary-final-v1.json"
    )
    saved = {"cloud.py": Path(__file__).read_text(), "config.json": config.read_text()}
    for name in TOOLS:
        saved["tools/" + name] = (base / "tools" / name).read_text()
    for path in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + path.name] = path.read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=1800,
    retries=0,
    max_containers=2,
    volumes={"/artifacts": volume},
)
def evaluate(protocol, seed):
    import hashlib
    import json
    import time
    import numpy as np
    import torch
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.content_scaling import scale_duration
    from behavioral_equivalence.item_context import encode_items
    from behavioral_equivalence.quantization import assign_centers
    from behavioral_equivalence.case_inference import forecast_indices
    from behavioral_equivalence.streaming_scores import shared_user_bootstrap
    from behavioral_equivalence.cohort_uncertainty import cohort_interval_diagnostic
    from supplementary_heads import predict_head
    from supplementary_metrics import categorical_block, merge_categorical_blocks
    from supplementary_outcomes import categorical_targets
    from supplementary_counts import smoothed_emissions
    from supplementary_intervals import uniform_endpoint_envelope

    volume.reload()
    started = time.monotonic()
    saved, digest = snapshot()
    cfg = json.loads(saved["config.json"])
    if cfg["status"] != "execution_frozen" or len(cfg["probes"]) != 10:
        raise ValueError("Complete frozen supplementary test matrix required")
    spec = next(p for p in cfg["probes"] if p["protocol"] == protocol and p["seed"] == seed)
    cases = Path("/artifacts") / cfg["case_root"]
    context_root = Path("/artifacts") / cfg["context_root"]
    targets = Path("/artifacts") / cfg["target_root"]
    if (
        checksum(cases / "report.json") != cfg["case_report_sha256"]
        or checksum(context_root / "report.json") != cfg["context_report_sha256"]
    ):
        raise ValueError("Audited case/context inputs changed")
    cr = json.loads((cases / "report.json").read_text())
    ir = json.loads((context_root / "report.json").read_text())
    tr = json.loads((targets / "report.json").read_text())
    if (
        hashlib.sha256(json.dumps(tr, sort_keys=True).encode()).hexdigest()
        != cfg["target_report_canonical_digest"]
    ):
        raise ValueError("Supplementary target identity changed")
    if any(r["status"] != "complete" or len(r["buckets"]) != 32 for r in [cr, ir, tr]):
        raise ValueError("All32 source buckets required")
    cal = Path("/artifacts") / spec["calibration"]["root"]
    if checksum(cal / "report.json") != spec["calibration"]["report_sha256"]:
        raise ValueError("Frozen calibration report changed")
    car = json.loads((cal / "report.json").read_text())
    if (
        car["source_digest"] != spec["calibration"]["source_digest"]
        or car["protocol"] != protocol
        or car["seed"] != seed
    ):
        raise ValueError("Calibration belongs to a different fitted model")
    for name, sha in spec["calibration"]["output_sha256"].items():
        if checksum(cal / name) != sha:
            raise ValueError("Frozen calibrated emission/certificate artifact changed")
    with np.load(cal / "models.npz", allow_pickle=False) as a:
        tables = {name: a[name] for name in a.files}
    abstract_records = json.loads((cal / "abstractions.json").read_text())
    names = [e["name"] for e in cfg["endpoints"]]
    primary_records = {
        r["endpoint"]: r
        for r in abstract_records
        if r["method"] == "refined"
        and r["requested_epsilon"] == cfg["primary_epsilon"]
        and "-start64-" in r["name"]
    }
    if set(primary_records) != set(names):
        raise ValueError("Fixed six-endpoint refined primary required")
    envelopes = {
        name: uniform_endpoint_envelope(
            tables[f"k256_endpoint{j}_lower"],
            tables[f"k256_endpoint{j}_upper"],
            primary_records[name]["blocks"],
        )
        for j, name in enumerate(names)
    }
    parent = Path("/artifacts") / car["parent"]["root"]
    quant = parent.parent
    pr = json.loads((parent / "report.json").read_text())
    if (
        checksum(parent / "report.json") != car["parent"]["report_sha256"]
        or checksum(quant / "report.json") != pr["quantization_report_sha256"]
    ):
        raise ValueError("Frozen parent quantizer changed")
    qr = json.loads((quant / "report.json").read_text())
    quantizers = {}
    if qr["parent_fit_report_sha256"] != spec["fit_report_sha256"]["gru"]:
        raise ValueError("Calibration/test state encoder identity mismatch")
    for k in cfg["state_counts"]:
        path = quant / f"k{k}-model.npz"
        if checksum(path) != qr["output_sha256"][path.name]:
            raise ValueError("Training state centers changed")
        with np.load(path, allow_pickle=False) as a:
            quantizers[k] = {name: a[name] for name in ["mean", "scale", "centers"]}
    training_root = Path("/artifacts") / spec["training_root"]
    original = json.loads((training_root / "source.json").read_text())
    original_digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    for name in [
        "neural.py",
        "exposure_neural.py",
        "content_neural.py",
        "item_context.py",
        "content_scaling.py",
    ]:
        if (
            original["src/behavioral_equivalence/" + name]
            != saved["src/behavioral_equivalence/" + name]
        ):
            raise ValueError("Frozen encoder code changed")
    training = next(
        json.loads(v)
        for k, v in original.items()
        if k.startswith("configs/human-training-calendar")
    )
    inputs = Path("/artifacts") / training["input_artifacts"][protocol]["path"]
    if checksum(inputs / "report.json") != training["input_artifacts"][protocol]["report_sha256"]:
        raise ValueError("Training-only preprocessing changed")
    input_report = json.loads((inputs / "report.json").read_text())
    for name in ["item-vocabulary.npy", "duration-scaling.json"]:
        if checksum(inputs / name) != input_report["output_sha256"][name]:
            raise ValueError("Preprocessing artifacts changed")
    vocab = np.load(inputs / "item-vocabulary.npy", allow_pickle=False)
    scaling = json.loads((inputs / "duration-scaling.json").read_text())
    models = {}
    heads = {}
    torch.set_num_threads(8)
    for architecture, fit_sha in spec["fit_report_sha256"].items():
        path = training_root / "fits" / f"{protocol}-{architecture}-{seed}"
        if checksum(path / "report.json") != fit_sha:
            raise ValueError("Encoder fit report changed")
        r = json.loads((path / "report.json").read_text())
        if (
            r["status"] != "complete"
            or r["source_digest"] != original_digest
            or any(checksum(path / name) != sha for name, sha in r["output_sha256"].items())
        ):
            raise ValueError("Completed encoder weights changed")
        model = build_content_model(
            architecture, input_report["vocabulary_size"], training["hidden"]
        )
        model.load_state_dict(torch.load(path / "best.pt", weights_only=True, map_location="cpu"))
        model.eval()
        models[architecture] = model
        head = Path("/artifacts") / spec["heads"][architecture]["root"]
        if (
            checksum(head / "report.json") != spec["heads"][architecture]["report_sha256"]
            or checksum(head / "model.npz") != spec["heads"][architecture]["model_sha256"]
        ):
            raise ValueError("Frozen supplementary heads changed")
        with np.load(head / "model.npz", allow_pickle=False) as a:
            heads[architecture] = [
                {
                    key: a[f"endpoint{j}_{key}"]
                    for key in ["mean", "scale", "coefficients", "intercept"]
                }
                for j in range(6)
            ]
    cohort = Path("/artifacts") / cfg["fitting_cohort_root"]
    cohort_report = json.loads((cohort / "report.json").read_text())
    if (
        hashlib.sha256(json.dumps(cohort_report, sort_keys=True).encode()).hexdigest()
        != cfg["fitting_cohort_canonical_digest"]
    ):
        raise ValueError("Baseline training cohort changed")
    record = next(
        r for r in cohort_report["records"] if r["protocol"] == protocol and r["role"] == "train"
    )
    if checksum(cohort / record["path"]) != record["sha256"]:
        raise ValueError("Baseline fitting cases changed")
    baselines = {}
    with np.load(cohort / record["path"], allow_pickle=False) as a:
        state = a["tokens"][:, -1].astype(int) * 15 + a["tab"]
        y = a["targets"]
        for j, endpoint in enumerate(cfg["endpoints"]):
            c = endpoint["classes"]
            observed = y[:, j] >= 0
            counts = np.bincount(state[observed] * c + y[observed, j], minlength=270 * c).reshape(
                270, c
            )
            baselines[(j, "markov")] = smoothed_emissions(counts)
            baselines[(j, "constant")] = smoothed_emissions(counts.sum(axis=0, keepdims=True))[0]
    out = Path("/artifacts") / ("supplementary-final-" + digest[:16]) / f"{protocol}-{seed}"
    out.mkdir(parents=True, exist_ok=True)
    summaries = {}
    outputs = {}
    diagnostic_parts = []
    case_sha = hashlib.sha256()
    total = 0
    for bucket in range(32):
        name = "embargo-24h.npz"
        case_path = cases / f"bucket-{bucket:02d}" / name
        context_path = context_root / f"bucket-{bucket:02d}" / name
        target_path = targets / f"bucket-{bucket:02d}.npz"
        if (
            checksum(case_path) != cr["buckets"][bucket]["output_sha256"][name]
            or checksum(context_path) != ir["buckets"][bucket]["output_sha256"][name]
            or checksum(target_path) != tr["buckets"][bucket]["output_sha256"]
        ):
            raise ValueError("Test source/context/target bucket changed")
        with (
            np.load(case_path, allow_pickle=False) as a,
            np.load(context_path, allow_pickle=False) as intrinsic,
            np.load(target_path, allow_pickle=False) as t,
        ):
            if not np.array_equal(a["case_id"], t["case_id"]):
                raise ValueError("Supplementary labels and cases differ")
            selected = forecast_indices(
                a["case_id"],
                a["user_id"],
                a["role_code"],
                a["user_role_code"],
                intrinsic["case_id"],
                protocol,
                "test",
            )
            part = {
                key: a[key][selected] for key in ["case_id", "user_id", "tokens", "tab", "video_id"]
            }
            context = scale_duration(intrinsic["context"][selected], scaling)
            y = categorical_targets(
                {
                    key: t[key][selected]
                    for key in [
                        "is_like",
                        "is_follow",
                        "is_comment",
                        "is_forward",
                        "is_hate",
                        *[f"watch_category_{k}" for k in range(4)],
                    ]
                }
            )
        ids, users = part["case_id"], part["user_id"]
        items = encode_items(vocab, part["video_id"])
        case_sha.update(ids.astype("<u8").tobytes())
        total += len(ids)
        archive = dict(case_id=ids, user_id=users, targets=y)
        states = {k: [] for k in cfg["state_counts"]}
        for architecture, model in models.items():
            predictions = [[] for _ in names]
            for offset in range(0, len(ids), cfg["inference_batch_size"]):
                sl = slice(offset, offset + cfg["inference_batch_size"])
                args = [
                    torch.as_tensor(part["tokens"][sl], dtype=torch.long),
                    torch.as_tensor(part["tab"][sl], dtype=torch.long),
                    torch.as_tensor(items[sl], dtype=torch.long),
                    torch.as_tensor(context[sl], dtype=torch.float32),
                ]
                with torch.inference_mode():
                    z = model.output[:-1](model.representation(*args)).numpy().astype(float)
                for j in range(6):
                    predictions[j].append(predict_head(z, heads[architecture][j]))
                if architecture == "gru":
                    for k, q in quantizers.items():
                        states[k].append(assign_centers(z, q["mean"], q["scale"], q["centers"]))
            for j, endpoint in enumerate(names):
                p = (
                    np.concatenate(predictions[j])
                    if predictions[j]
                    else np.empty((0, cfg["endpoints"][j]["classes"]))
                )
                archive[f"{architecture}_endpoint{j}"] = p
                summaries.setdefault((endpoint, architecture), []).append(
                    categorical_block(users, y[:, j], p)
                )
        states = {k: np.concatenate(v) if v else np.empty(0, dtype=int) for k, v in states.items()}
        for k, z in states.items():
            archive[f"k{k}_state"] = z.astype(np.int16)
        markov_state = part["tokens"][:, -1].astype(int) * 15 + part["tab"]
        archive["markov_state"] = markov_state.astype(np.int16)
        for j, endpoint in enumerate(names):
            for method, p in [
                ("markov", baselines[(j, "markov")][markov_state]),
                (
                    "constant",
                    np.broadcast_to(
                        baselines[(j, "constant")], (len(ids), cfg["endpoints"][j]["classes"])
                    ),
                ),
            ]:
                summaries.setdefault((endpoint, method), []).append(
                    categorical_block(users, y[:, j], p)
                )
            for k in cfg["state_counts"]:
                method = f"parent_k{k}"
                p = tables[f"k{k}_endpoint{j}_emissions"][states[k]]
                summaries.setdefault((endpoint, method), []).append(
                    categorical_block(users, y[:, j], p)
                )
            for r in [r for r in abstract_records if r["endpoint"] == endpoint]:
                p = tables[r["name"] + "_probabilities"][states[256]]
                summaries.setdefault((endpoint, r["name"]), []).append(
                    categorical_block(users, y[:, j], p)
                )
        diagnostic_parts.append(
            dict(users=users, targets=y, **{f"k{k}": z for k, z in states.items()})
        )
        path = out / f"bucket-{bucket:02d}-factorized.npz"
        np.savez_compressed(path, **archive)
        outputs[path.name] = checksum(path)
        print(
            json.dumps(
                dict(
                    protocol=protocol,
                    seed=seed,
                    bucket=bucket,
                    cases=len(ids),
                    cumulative_cases=total,
                )
            ),
            flush=True,
        )
    merged = {key: merge_categorical_blocks(blocks) for key, blocks in summaries.items()}
    metrics = {f"{e}/{m}": r["metrics"] for (e, m), r in merged.items()}
    comparisons = {}
    private = {}
    for j, endpoint in enumerate(names):
        members = {
            name: r["private_user_statistics"] for (e, name), r in merged.items() if e == endpoint
        }
        base = members["parent_k256"]
        for name, r in members.items():
            if any(not np.array_equal(r[key], base[key]) for key in ["user_ids", "case_counts"]):
                raise ValueError("Endpoint model supports differ")
            private[f"{endpoint}__{name}__nll"] = r["user_nll"]
            private[f"{endpoint}__{name}__brier"] = r["user_brier"]
        private[f"{endpoint}__users"] = base["user_ids"]
        private[f"{endpoint}__cases"] = base["case_counts"]
        contrasts = {
            f"{name}_versus_parent": (name, "parent_k256")
            for name in members
            if name != "parent_k256"
        }
        primary = primary_records[endpoint]["name"]
        random = primary.replace("-refined-", "-random_matched-")
        feature = f"{endpoint}-start64-feature_unconstrained-epsilon1.0"
        contrasts.update(
            refined_versus_random=(primary, random), refined_versus_feature=(primary, feature)
        )
        if (
            len(base["user_ids"]) >= cfg["minimum_users"]
            and int(base["case_counts"].sum()) >= cfg["minimum_cases"]
        ):
            comparisons[endpoint] = dict(
                status="evaluated_descriptive",
                contrasts=shared_user_bootstrap(
                    base["user_ids"],
                    base["case_counts"],
                    {name: r["user_nll"] for name, r in members.items()},
                    contrasts,
                    replications=cfg["bootstrap_replications"],
                    seed=seed,
                    margin=0.01,
                ),
            )
        else:
            comparisons[endpoint] = dict(status="unsupported", users=len(base["user_ids"]))
    np.savez_compressed(out / "private-user-statistics.npz", **private)
    outputs["private-user-statistics.npz"] = checksum(out / "private-user-statistics.npz")
    all_users = np.concatenate([p["users"] for p in diagnostic_parts])
    all_targets = np.concatenate([p["targets"] for p in diagnostic_parts])
    all_states = {
        k: np.concatenate([p[f"k{k}"] for p in diagnostic_parts]) for k in cfg["state_counts"]
    }
    diagnostics = {}
    uniform_metadata = {}
    for j, endpoint in enumerate(names):
        keep = all_targets[:, j] >= 0
        y = all_targets[keep, j]
        u = all_users[keep]
        for k in cfg["state_counts"]:
            state = all_states[k][keep]
            for category in range(cfg["endpoints"][j]["classes"]):
                diagnostics[f"{endpoint}/k{k}/category{category}"] = cohort_interval_diagnostic(
                    (y == category).astype(int),
                    tables[f"k{k}_endpoint{j}_lower"][state, category],
                    tables[f"k{k}_endpoint{j}_upper"][state, category],
                    u,
                    replications=cfg["bootstrap_replications"],
                    seed=seed,
                )
        envelope = envelopes[endpoint]
        state = all_states[256][keep]
        uniform_metadata[endpoint] = {
            key: envelope[key] for key in ["uniform_delta", "delta_exact", "scope"]
        }
        for category in range(cfg["endpoints"][j]["classes"]):
            diagnostics[f"{endpoint}/refined64_uniform/category{category}"] = (
                cohort_interval_diagnostic(
                    (y == category).astype(int),
                    envelope["original_lower"][state, category],
                    envelope["original_upper"][state, category],
                    u,
                    replications=cfg["bootstrap_replications"],
                    seed=seed,
                )
            )
    # Baseline tables make their forecasts reproducible without refitting.
    np.savez_compressed(
        out / "baseline-tables.npz",
        **{f"endpoint{j}_{method}": v for (j, method), v in baselines.items()},
    )
    outputs["baseline-tables.npz"] = checksum(out / "baseline-tables.npz")
    report = dict(
        status="complete",
        source_digest=digest,
        protocol=protocol,
        seed=seed,
        cases=total,
        case_id_sha256=case_sha.hexdigest(),
        calibration=spec["calibration"],
        heads=spec["heads"],
        metrics=metrics,
        comparisons=comparisons,
        interval_diagnostics=diagnostics,
        uniform_envelopes=uniform_metadata,
        output_sha256=outputs,
        wall_seconds=time.monotonic() - started,
        root=str(out),
        forecast_factorization=cfg["forecast_archive"],
        scope=cfg["scope"],
        human_confidence_interval_coverage_established=False,
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
    jobs = [(p["protocol"], p["seed"]) for p in cfg["probes"]]
    if len(jobs) != 10 or len(set(jobs)) != 10 or (protocol, seed) not in jobs:
        raise ValueError("All10 fixed fitted-model cases required")
    ledger = BudgetLedger(ROOT / ".campaign/calendar-quantization-budget.sqlite", 40)
    for p, s in jobs:
        reservation = ledger.reserve(f"supplementary-final-{digest}-{p}-{s}", 0.30)
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
    (
        ROOT / "reports/source-snapshots" / ("supplementary-final-" + digest[:16] + ".json")
    ).write_text(json.dumps(source, indent=2) + "\n")
    stop = threading.Event()

    def run(job):
        if stop.is_set():
            return
        p, s = job
        state = ROOT / ".campaign" / f"supplementary-final-{digest[:16]}-{p}-{s}.json"
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
            content = b"".join(
                volume.read_file(result["root"].removeprefix("/artifacts/") + "/report.json")
            )
            report = json.loads(content)
            if (
                report["status"] != "complete"
                or report["source_digest"] != digest
                or report["protocol"] != p
                or report["seed"] != s
            ):
                raise ValueError("Supplementary final report identity mismatch")
            directory = ROOT / "reports/supplementary-final"
            directory.mkdir(exist_ok=True)
            (directory / f"{p}-{s}.json").write_bytes(content)
            print(
                json.dumps(
                    dict(
                        status="complete",
                        protocol=p,
                        seed=s,
                        cases=report["cases"],
                        wall_seconds=report["wall_seconds"],
                    )
                ),
                flush=True,
            )
        except Exception:
            stop.set()
            raise

    with launcher_lock(ROOT / ".campaign/supplementary-final-launcher.lock"):
        with ThreadPoolExecutor(2) as pool:
            list(pool.map(run, jobs if matrix else [(protocol, seed)]))
