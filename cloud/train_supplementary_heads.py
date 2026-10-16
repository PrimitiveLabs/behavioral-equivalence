"""Fit supplementary outcome heads on authenticated frozen encoder representations."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-supplementary-heads-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.1")
    .env(
        {
            "PYTHONPATH": "/research/src:/research/tools",
            "OPENBLAS_NUM_THREADS": "8",
            "OMP_NUM_THREADS": "8",
        }
    )
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tools/supplementary_heads.py", remote_path="/research/tools/supplementary_heads.py"
    )
    .add_local_file(
        ROOT / "configs/human-supplementary-heads-v1.json", remote_path="/research/config.json"
    )
)


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
        else base / "configs/human-supplementary-heads-v1.json"
    )
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "config.json": config.read_text(),
        "tools/supplementary_heads.py": (base / "tools/supplementary_heads.py").read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=900,
    retries=0,
    max_containers=2,
    volumes={"/artifacts": volume},
)
def fit(protocol, architecture, seed):
    import hashlib
    import json
    import time
    import numpy as np
    import torch
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.content_scaling import scale_duration
    from behavioral_equivalence.item_context import encode_items
    from supplementary_heads import fit_head, predict_head, categorical_scores

    volume.reload()
    started = time.monotonic()
    saved, digest = snapshot()
    cfg = json.loads(saved["config.json"])
    if (
        cfg["status"] != "execution_frozen"
        or protocol not in cfg["protocols"]
        or architecture not in cfg["architectures"]
        or seed not in cfg["seeds"]
    ):
        raise ValueError("Frozen complete supplementary fitting matrix required")
    probe = next(p for p in cfg["parent_probes"] if p["protocol"] == protocol and p["seed"] == seed)
    root = Path("/artifacts") / probe["training_root"]
    parent = root / "fits" / f"{protocol}-{architecture}-{seed}"
    if checksum(parent / "report.json") != probe["fit_report_sha256"][architecture]:
        raise ValueError("Parent encoder report changed")
    pr = json.loads((parent / "report.json").read_text())
    original = json.loads((root / "source.json").read_text())
    original_digest = hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()
    if (
        pr["status"] != "complete"
        or pr["source_digest"] != original_digest
        or pr["protocol"] != protocol
        or pr["architecture"] != architecture
        or pr["seed"] != seed
    ):
        raise ValueError("Authenticated complete exact encoder fit required")
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
            raise ValueError("Frozen encoder implementation changed")
    if any(checksum(parent / name) != sha for name, sha in pr["output_sha256"].items()):
        raise ValueError("Parent weights/artifacts changed")
    training = next(
        json.loads(v)
        for k, v in original.items()
        if k.startswith("configs/human-training-calendar")
    )
    inputs = Path("/artifacts") / training["input_artifacts"][protocol]["path"]
    if checksum(inputs / "report.json") != pr["input_report_sha256"]:
        raise ValueError("Frozen preprocessing report changed")
    ir = json.loads((inputs / "report.json").read_text())
    for name in ["item-vocabulary.npy", "duration-scaling.json"]:
        if checksum(inputs / name) != ir["output_sha256"][name]:
            raise ValueError("Training-only preprocessing changed")
    vocab = np.load(inputs / "item-vocabulary.npy", allow_pickle=False)
    scaling = json.loads((inputs / "duration-scaling.json").read_text())
    cohort = Path("/artifacts") / cfg["fitting_cohort_root"]
    cohort_report = json.loads((cohort / "report.json").read_text())
    if (
        hashlib.sha256(json.dumps(cohort_report, sort_keys=True).encode()).hexdigest()
        != cfg["fitting_cohort_canonical_digest"]
        or cohort_report["status"] != "complete"
    ):
        raise ValueError("Audited supplementary fitting cohort changed")
    torch.set_num_threads(8)
    model = build_content_model(architecture, ir["vocabulary_size"], training["hidden"])
    model.load_state_dict(torch.load(parent / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    data = {}
    for role in ["train", "validation"]:
        record = next(
            r for r in cohort_report["records"] if r["protocol"] == protocol and r["role"] == role
        )
        path = cohort / record["path"]
        if checksum(path) != record["sha256"]:
            raise ValueError("Fitting arrays changed")
        with np.load(path, allow_pickle=False) as a:
            items = encode_items(vocab, a["video_id"])
            context = scale_duration(a["intrinsic_context"], scaling)
            parts = []
            for offset in range(0, len(items), 4096):
                sl = slice(offset, offset + 4096)
                args = [
                    torch.as_tensor(a["tokens"][sl], dtype=torch.long),
                    torch.as_tensor(a["tab"][sl], dtype=torch.long),
                    torch.as_tensor(items[sl], dtype=torch.long),
                    torch.as_tensor(context[sl], dtype=torch.float32),
                ]
                with torch.inference_mode():
                    parts.append(
                        model.output[:-1](model.representation(*args)).numpy().astype(float)
                    )
            data[role] = dict(
                features=np.concatenate(parts),
                targets=a["targets"],
                users=a["user_id"],
                case_id_sha256=record["case_id_sha256"],
            )
    out = (
        Path("/artifacts")
        / ("supplementary-heads-" + digest[:16])
        / f"{protocol}-{architecture}-{seed}"
    )
    out.mkdir(parents=True, exist_ok=True)
    parameters = {}
    records = []
    for index, endpoint in enumerate(cfg["endpoints"]):
        train = data["train"]
        validation = data["validation"]
        target = train["targets"][:, index]
        vy = validation["targets"][:, index]
        if not np.any(target >= 0):
            records.append(
                dict(endpoint=endpoint["name"], status="unavailable_no_observed_training_labels")
            )
            continue
        candidates = []
        for C in cfg["head"]["C_grid"]:
            fitted, details = fit_head(
                train["features"], target, train["users"], endpoint["classes"], C=C
            )
            metrics = categorical_scores(
                predict_head(validation["features"], fitted),
                vy,
                validation["users"],
                cfg["head"]["probability_floor"],
            )
            candidates.append((C, fitted, details, metrics))
        selected = min(
            candidates,
            key=lambda p: (p[3]["nll"] if p[3]["nll"] is not None else float("inf"), p[0]),
        )
        C, fitted, details, metrics = selected
        for name, array in fitted.items():
            parameters[f"endpoint{index}_{name}"] = array
        records.append(
            dict(
                endpoint=endpoint["name"],
                status="complete",
                classes=endpoint["classes"],
                selected_C=C,
                fit=details,
                validation=metrics,
                selection="equal-user validation NLL; tie resolved by lower C (stronger regularization)"
                if metrics["nll"] is not None
                else "no observed validation labels; lowest C fallback",
                candidates=[dict(C=c, fit=r, validation=m) for c, _, r, m in candidates],
            )
        )
    np.savez_compressed(out / "model.npz", **parameters)
    report = dict(
        status="complete",
        source_digest=digest,
        protocol=protocol,
        architecture=architecture,
        seed=seed,
        parent_fit_report_sha256=checksum(parent / "report.json"),
        fitting_cohort_canonical_digest=cfg["fitting_cohort_canonical_digest"],
        sample_id_sha256={role: d["case_id_sha256"] for role, d in data.items()},
        records=records,
        output_sha256={"model.npz": checksum(out / "model.npz")},
        wall_seconds=time.monotonic() - started,
        scope="Outcome-specific heads on frozen representations. Supplementary validation only; no supplementary test prediction, no primary click replacement or new certificate claim.",
    )
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return dict(root=str(out), status=report["status"], wall_seconds=report["wall_seconds"])


@app.local_entrypoint()
def main(protocol: str = "known_users", architecture: str = "context", seed: int = 20261015):
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    source, digest = snapshot()
    cfg = json.loads(source["config.json"])
    if (
        protocol not in cfg["protocols"]
        or architecture not in cfg["architectures"]
        or seed not in cfg["seeds"]
    ):
        raise ValueError("Fixed fitting matrix required")
    ledger = BudgetLedger(ROOT / ".campaign/human-training-calendar-v2-budget.sqlite", 30)
    for p in cfg["protocols"]:
        for a in cfg["architectures"]:
            for s in cfg["seeds"]:
                reservation = ledger.reserve(f"supplementary-heads-{digest}-{p}-{a}-{s}", 0.15)
    print(
        json.dumps(
            dict(
                matrix_reserved_usd=4.5,
                total_reserved_usd=reservation["total_reserved_usd"],
                ceiling_usd=reservation["ceiling_usd"],
            )
        ),
        flush=True,
    )
    state = (
        ROOT
        / ".campaign"
        / f"supplementary-heads-{digest[:16]}-{protocol}-{architecture}-{seed}.json"
    )
    if state.exists():
        saved = json.loads(state.read_text())
        call = modal.FunctionCall.from_id(saved["function_call_id"])
    else:
        with state.with_suffix(".submission-attempt").open("x") as marker:
            marker.write(digest + "\n")
        call = fit.spawn(protocol, architecture, seed)
        saved = dict(
            source_digest=digest,
            function_call_id=call.object_id,
            protocol=protocol,
            architecture=architecture,
            seed=seed,
        )
        state.write_text(json.dumps(saved, indent=2))
    print(json.dumps(saved), flush=True)
    (
        ROOT / "reports/source-snapshots" / ("supplementary-heads-" + digest[:16] + ".json")
    ).write_text(json.dumps(source, indent=2) + "\n")
    result = call.get()
    content = b"".join(
        volume.read_file(result["root"].removeprefix("/artifacts/") + "/report.json")
    )
    report = json.loads(content)
    if (
        report["source_digest"] != digest
        or report["protocol"] != protocol
        or report["architecture"] != architecture
        or report["seed"] != seed
    ):
        raise ValueError("Returned head report identity mismatch")
    directory = ROOT / "reports/supplementary-heads"
    directory.mkdir(exist_ok=True)
    (directory / f"{protocol}-{architecture}-{seed}.json").write_bytes(content)
    print(json.dumps(result), flush=True)
