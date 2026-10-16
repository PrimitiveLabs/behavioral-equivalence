"""Source-bound multi-seed sequence training, with no final test scoring."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-human-calendar-training-v4-replicates")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "pytest==8.4.1")
    .env(
        {
            "PYTHONPATH": "/research/src",
            "OPENBLAS_NUM_THREADS": "1",
            "OMP_NUM_THREADS": "8",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }
    )
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "configs/human-training-calendar-v4-replicates.json",
        remote_path="/research/configs/human-training-calendar-v4-replicates.json",
    )
    .add_local_file(
        ROOT / "tests/test_content_neural.py",
        remote_path="/research/tests/test_content_neural.py",
    )
)


def snapshot() -> tuple[dict, str]:
    import hashlib
    import json

    base = (
        Path("/research")
        if Path("/research/configs/human-training-calendar-v4-replicates.json").exists()
        else ROOT
    )
    saved = {"cloud/train_calendar_replicates.py": Path(__file__).read_text()}
    for name in [
        "configs/human-training-calendar-v4-replicates.json",
        "tests/test_content_neural.py",
    ]:
        saved[name] = (base / name).read_text()
    for name in [
        "neural.py",
        "exposure_neural.py",
        "content_neural.py",
        "content_scaling.py",
        "item_context.py",
        "statistics.py",
        "budget.py",
    ]:
        saved["src/behavioral_equivalence/" + name] = (
            base / "src/behavioral_equivalence" / name
        ).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


def sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            h.update(block)
    return h.hexdigest()


def context() -> tuple[Path, dict, str]:
    import json

    _, digest = snapshot()
    config = json.loads(
        Path("/research/configs/human-training-calendar-v4-replicates.json").read_text()
    )
    return Path("/artifacts/human-calendar-training-v4-replicates-" + digest[:12]), config, digest


def input_path(root: Path, config: dict, protocol: str) -> Path:
    if "input_artifacts" in config:
        return Path("/artifacts") / config["input_artifacts"][protocol]["path"]
    return root / "inputs" / protocol


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(4096, 4096),
    timeout=300,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def smoke() -> dict:
    import json
    import subprocess
    import time
    import torch
    from behavioral_equivalence.content_neural import build_content_model

    root, config, digest = context()
    root.mkdir(exist_ok=True)
    saved, _ = snapshot()
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    result = subprocess.run(
        ["python", "-m", "pytest", "-q", "-W", "error", "/research/tests/test_content_neural.py"],
        capture_output=True,
        text=True,
        timeout=100,
    )
    torch.set_num_threads(8)
    checks = []
    timing = []
    for architecture in config["architectures"]:
        model = build_content_model(architecture, 10, hidden=16).to("cpu")
        x = torch.zeros((8, 32), dtype=torch.long, device="cpu")
        x[:, -1] = 17
        tab, item = torch.zeros(8, dtype=torch.long, device="cpu"), torch.arange(8, device="cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x, tab, item, torch.zeros((8, 2), device="cpu")),
            torch.arange(8, device="cpu").float() % 2,
        )
        loss.backward()
        optimizer.step()
        if not torch.isfinite(loss):
            raise AssertionError("CPU training smoke produced nonfinite loss")
        checks.append({"architecture": architecture, "cpu_backward": "passed"})
        full = build_content_model(
            architecture, config["maximum_item_vocabulary"], hidden=config["hidden"]
        ).to("cpu")
        opt = torch.optim.AdamW(full.parameters(), lr=config["optimizer"]["lr"])
        batch = config["batch_size"]
        arguments = [
            torch.randint(1, 18, (batch, 32), device="cpu"),
            torch.randint(0, 15, (batch,), device="cpu"),
            torch.randint(0, config["maximum_item_vocabulary"] + 1, (batch,), device="cpu"),
            torch.zeros((batch, 2), device="cpu"),
        ]
        targets = torch.arange(batch, device="cpu").float() % 2
        for step in range(13):
            if step == 3:
                began = time.monotonic()
            opt.zero_grad(set_to_none=True)
            current = torch.nn.functional.binary_cross_entropy_with_logits(
                full(*arguments), targets
            )
            current.backward()
            opt.step()
        seconds = (time.monotonic() - began) / 10
        timing.append(
            {
                "architecture": architecture,
                "parameters": sum(p.numel() for p in full.parameters()),
                "training_step_seconds": seconds,
                "cases_per_second": batch / seconds,
                "method": "synthetic full-size batches, three warmup and ten timed updates; not a human result",
            }
        )
    report = {
        "status": "complete" if result.returncode == 0 else "failed",
        "source_digest": digest,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "hardware": "Modal CPU, float32, eight threads",
        "cpu_checks": checks,
        "throughput_preflight": timing,
    }
    (root / "smoke.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=600,
    max_containers=2,
    retries=0,
    volumes={"/artifacts": volume},
)
def stage_inputs(protocol: str) -> dict:
    import json
    import numpy as np
    from behavioral_equivalence.item_context import fit_item_vocabulary, encode_items
    from behavioral_equivalence.content_scaling import fit_duration_scaling, scale_duration
    from behavioral_equivalence.statistics import prediction_scores

    root, config, digest = context()
    if protocol not in config["protocols"]:
        raise ValueError("Frozen protocol required")
    if "input_artifacts" in config:
        upstream = input_path(root, config, protocol)
        if sha256(upstream / "report.json") != config["input_artifacts"][protocol]["report_sha256"]:
            raise ValueError("Bound upstream training-input report changed")
        report = json.loads((upstream / "report.json").read_text())
        if report["status"] != "complete" or any(
            sha256(upstream / name) != checksum
            for name, checksum in report["output_sha256"].items()
        ):
            raise ValueError("Bound upstream training inputs changed")
        return report
    cases = Path("/artifacts") / config["case_root"]
    if (
        sha256(cases / "report.json") != config["case_report_sha256"]
        or sha256(cases / config["case_audit_relative"]) != config["case_audit_sha256"]
    ):
        raise ValueError("Source case construction or independent audit changed")
    source = json.loads((cases / "report.json").read_text())
    intrinsic = Path("/artifacts") / config["context_root"]
    if sha256(intrinsic / "report.json") != config["context_report_sha256"]:
        raise ValueError("Intrinsic case context audit changed")
    intrinsic_report = json.loads((intrinsic / "report.json").read_text())
    audit = json.loads((cases / config["case_audit_relative"]).read_text())
    if source["status"] != "complete" or audit["status"] != "complete":
        raise ValueError("Complete independently audited case inputs required")
    out = root / "inputs" / protocol
    if (out / "report.json").exists():
        report = json.loads((out / "report.json").read_text())
        if report["source_digest"] != digest or any(
            sha256(out / name) != checksum for name, checksum in report["output_sha256"].items()
        ):
            raise ValueError("Staged input changed")
        return report
    pieces = {role: [] for role in ["train", "validation"]}
    for bucket in source["buckets"]:
        name = f"embargo-{config['embargo_hours']:02d}h.npz"
        path = cases / f"bucket-{bucket['bucket']:02d}" / name
        if sha256(path) != bucket["output_sha256"][name]:
            raise ValueError("Audited case artifact changed")
        context_path = intrinsic / f"bucket-{bucket['bucket']:02d}" / name
        if (
            sha256(context_path)
            != intrinsic_report["buckets"][bucket["bucket"]]["output_sha256"][name]
        ):
            raise ValueError("Intrinsic context sidecar changed")
        with np.load(context_path, allow_pickle=False) as sidecar:
            context_ids, context_values = sidecar["case_id"], sidecar["context"]
        with np.load(path, allow_pickle=False) as saved:
            if not np.array_equal(context_ids, saved["case_id"]):
                raise ValueError("Intrinsic context and prediction cases are misaligned")
            role, user_role = saved["role_code"], saved["user_role_code"]
            for r, split in enumerate(["train", "validation"]):
                selected = role == r
                if protocol == "unseen_users":
                    selected &= user_role == r
                part = {
                    key: saved[key][selected]
                    for key in ["case_id", "user_id", "tokens", "tab", "video_id", "target"]
                }
                part["intrinsic_context"] = context_values[selected]
                pieces[split].append(part)
    data = {
        split: {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
        for split, parts in pieces.items()
    }
    del pieces
    vocabulary = fit_item_vocabulary(data["train"]["video_id"], config["maximum_item_vocabulary"])
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "item-vocabulary.npy", vocabulary, allow_pickle=False)
    scaling = fit_duration_scaling(data["train"]["intrinsic_context"])
    (out / "duration-scaling.json").write_text(json.dumps(scaling, indent=2))
    outputs, support = {"item-vocabulary.npy": sha256(out / "item-vocabulary.npy")}, {}
    for split in ["train", "validation"]:
        arrays = data[split]
        arrays["item_id"] = encode_items(vocabulary, arrays["video_id"])
        arrays["intrinsic_context"] = scale_duration(arrays["intrinsic_context"], scaling)
        np.savez_compressed(out / f"{split}.npz", **arrays)
        outputs[f"{split}.npz"] = sha256(out / f"{split}.npz")
        support[split] = {
            "cases": len(arrays["target"]),
            "users": int(len(np.unique(arrays["user_id"]))),
            "unknown_item_fraction": float(np.mean(arrays["item_id"] == 0)),
            "known_intrinsic_duration_fraction": float(
                np.mean(arrays["intrinsic_context"][:, 1] == 0)
            ),
        }
    train = data["train"]
    index = train["tokens"][:, -1].astype(int) * 15 + train["tab"]
    count = np.bincount(index, minlength=18 * 15)
    positive = np.bincount(index, weights=train["target"], minlength=18 * 15)
    probability = (positive + 1) / (count + 2)
    validation = data["validation"]
    forecast = probability[validation["tokens"][:, -1].astype(int) * 15 + validation["tab"]]
    baseline = {
        "probabilities": probability.tolist(),
        "smoothing": "Beta(1,1)",
        "validation": prediction_scores(validation["target"], forecast, validation["user_id"]),
        "scope": "validation only; no final test prediction",
    }
    (out / "markov.json").write_text(json.dumps(baseline, indent=2))
    outputs["markov.json"] = sha256(out / "markov.json")
    outputs["duration-scaling.json"] = sha256(out / "duration-scaling.json")
    report = {
        "status": "complete",
        "protocol": protocol,
        "source_digest": digest,
        "case_report_sha256": config["case_report_sha256"],
        "case_audit_sha256": config["case_audit_sha256"],
        "context_report_sha256": config["context_report_sha256"],
        "support": support,
        "vocabulary_size": len(vocabulary),
        "output_sha256": outputs,
        "scope": "Role-separated training and validation inputs only; calibration/test artifacts are not staged for model fitting.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=21600,
    max_containers=4,
    retries=0,
    volumes={"/artifacts": volume},
)
def fit(protocol: str, architecture: str, seed: int) -> dict:
    import json
    import time
    import numpy as np
    import torch
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.statistics import prediction_scores

    root, config, digest = context()
    if (
        protocol not in config["protocols"]
        or architecture not in config["architectures"]
        or seed not in config["seeds"]
    ):
        raise ValueError("Frozen model/protocol/seed required")
    volume.reload()
    source = input_path(root, config, protocol)
    inputs = json.loads((source / "report.json").read_text())
    upstream_digest = config.get("input_artifacts", {}).get(protocol, {}).get("report_sha256")
    if (
        (upstream_digest is not None and sha256(source / "report.json") != upstream_digest)
        or (upstream_digest is None and inputs["source_digest"] != digest)
        or any(
            sha256(source / name) != inputs["output_sha256"][name]
            for name in ["train.npz", "validation.npz", "item-vocabulary.npy"]
        )
    ):
        raise ValueError("Role-separated training inputs changed")
    out = root / "fits" / f"{protocol}-{architecture}-{seed}"
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if cached["source_digest"] != digest or any(
            sha256(out / name) != checksum for name, checksum in cached["output_sha256"].items()
        ):
            raise ValueError("Cached trained model changed")
        return cached
    out.mkdir(parents=True, exist_ok=True)
    (out / "started.json").write_text(
        json.dumps(
            {
                "status": "running",
                "source_digest": digest,
                "function_call_id": modal.current_function_call_id(),
                "hardware": "Modal CPU, float32, eight threads",
            }
        )
    )
    volume.commit()
    started = time.monotonic()
    torch.set_num_threads(8)
    torch.manual_seed(seed)
    train, validation = {}, {}
    for split, arrays in [("train", train), ("validation", validation)]:
        with np.load(source / f"{split}.npz", allow_pickle=False) as saved:
            for name in saved.files:
                arrays[name] = saved[name]
    gpu = {
        key: torch.as_tensor(
            train[key],
            device="cpu",
            dtype=torch.float32 if key in {"target", "intrinsic_context"} else torch.long,
        )
        for key in ["tokens", "tab", "item_id", "intrinsic_context", "target"]
    }
    model = build_content_model(
        architecture, inputs["vocabulary_size"], hidden=config["hidden"]
    ).to("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    shuffle = torch.Generator(device="cpu").manual_seed(seed)
    epochs, best = [], float("inf")
    for epoch in range(config["epochs"]):
        model.train()
        order = torch.randperm(len(train["target"]), device="cpu", generator=shuffle)
        total_loss = torch.zeros((), device="cpu")
        for offset in range(0, len(order), config["batch_size"]):
            batch = order[offset : offset + config["batch_size"]]
            loss = torch.nn.functional.binary_cross_entropy_with_logits(
                model(
                    gpu["tokens"][batch],
                    gpu["tab"][batch],
                    gpu["item_id"][batch],
                    gpu["intrinsic_context"][batch],
                ),
                gpu["target"][batch],
            )
            if not torch.isfinite(loss):
                raise ArithmeticError("Nonfinite training loss")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * len(batch)
        model.eval()
        prediction = []
        with torch.no_grad():
            for offset in range(0, len(validation["target"]), config["evaluation_batch_size"]):
                section = slice(offset, offset + config["evaluation_batch_size"])
                arguments = [
                    torch.as_tensor(
                        validation[key][section],
                        device="cpu",
                        dtype=torch.float32 if key == "intrinsic_context" else torch.long,
                    )
                    for key in ["tokens", "tab", "item_id", "intrinsic_context"]
                ]
                prediction.append(torch.sigmoid(model(*arguments)).cpu().numpy())
        prediction = np.concatenate(prediction)
        scores = prediction_scores(validation["target"], prediction, validation["user_id"])
        entry = {
            "epoch": epoch + 1,
            "training_nll": float(total_loss.item() / len(order)),
            "validation": scores,
        }
        epochs.append(entry)
        if scores["nll_equal_user_mean"] < best:
            best = scores["nll_equal_user_mean"]
            torch.save(model.state_dict(), out / "best.pt")
            np.savez_compressed(
                out / "validation-predictions.npz",
                case_id=validation["case_id"],
                user_id=validation["user_id"],
                target=validation["target"],
                probability=prediction,
            )
        (out / "progress.json").write_text(json.dumps(epochs, indent=2))
        volume.commit()
        print(
            json.dumps(
                {
                    "protocol": protocol,
                    "architecture": architecture,
                    "seed": seed,
                    "epoch": epoch + 1,
                    "validation_user_nll": scores["nll_equal_user_mean"],
                }
            ),
            flush=True,
        )
    report = {
        "status": "complete",
        "protocol": protocol,
        "architecture": architecture,
        "seed": seed,
        "source_digest": digest,
        "input_report_sha256": sha256(source / "report.json"),
        "parameters": sum(p.numel() for p in model.parameters()),
        "hardware": "Modal CPU, float32, eight threads",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "epochs": epochs,
        "best_epoch": min(epochs, key=lambda e: e["validation"]["nll_equal_user_mean"])["epoch"],
        "output_sha256": {
            name: sha256(out / name) for name in ["best.pt", "validation-predictions.npz"]
        },
        "wall_seconds": time.monotonic() - started,
        "test_predictive_performance": "not evaluated",
        "numerical_scope": "Seeded float32 CPU fitting; bitwise reproducibility across CPU/software environments not asserted.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=86400,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run() -> dict:
    import itertools
    import json

    root, config, digest = context()
    if json.loads((root / "smoke.json").read_text())["status"] != "complete":
        raise ValueError("Passing source-bound CPU and parameter-gradient tests required")
    staged = list(stage_inputs.map(config["protocols"], return_exceptions=True, order_outputs=True))
    failures = [repr(r) for r in staged if isinstance(r, Exception)]
    if failures:
        raise RuntimeError("Role staging failed: " + str(failures))
    jobs = list(itertools.product(config["protocols"], config["architectures"], config["seeds"]))
    if len(jobs) != config["resource_budget"]["training_jobs"]:
        raise ValueError("Frozen training matrix size mismatch")
    fitted = list(fit.starmap(jobs, return_exceptions=True, order_outputs=True))
    failures = [repr(r) for r in fitted if isinstance(r, Exception)]
    result = {
        "status": "incomplete" if failures else "complete",
        "phase": "locked multi-seed training and validation, not final test evaluation",
        "source_digest": digest,
        "config": config,
        "staged": staged,
        "fits": [r for r in fitted if not isinstance(r, Exception)],
        "failures": failures,
        "test_predictive_performance": "not evaluated",
    }
    (root / "manifest.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return {
        "status": result["status"],
        "fits_complete": len(result["fits"]),
        "failures": failures,
        "root": str(root),
    }


@app.local_entrypoint()
def main(stage: str = "smoke"):
    import json
    import sys

    if stage not in {"smoke", "run"}:
        raise ValueError("Stage must be smoke or run")
    _, digest = snapshot()
    if stage == "run":
        sys.path.insert(0, str(ROOT / "src"))
        from behavioral_equivalence.budget import BudgetLedger

        amount = (
            12 * 21600 * (8 * 1.31e-5 + 12 * 2.22e-6)
            + 2 * 600 * (2 * 1.31e-5 + 8 * 2.22e-6)
            + 86400 * (0.5 * 1.31e-5 + 2.22e-6)
            + 2
        )
        print(
            json.dumps(
                BudgetLedger(
                    ROOT / ".campaign/human-training-calendar-v4-replicates-budget.sqlite", 40
                ).reserve(digest, amount)
            ),
            flush=True,
        )
        handle = ROOT / ".campaign" / ("human-training-calendar-" + digest[:12] + ".json")
        if handle.exists():
            call = modal.FunctionCall.from_id(json.loads(handle.read_text())["function_call_id"])
            try:
                print(json.dumps(call.get(timeout=0)), flush=True)
            except TimeoutError:
                print(
                    "Existing training call has not returned; no duplicate submission.", flush=True
                )
            return
    function = smoke if stage == "smoke" else run
    call = function.spawn()
    record = {
        "function_call_id": call.object_id,
        "function_id": function.object_id,
        "source_digest": digest,
    }
    if stage == "run":
        handle.write_text(json.dumps(record, indent=2))
    print(json.dumps(record), flush=True)
    result = call.get()
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "complete":
        raise RuntimeError("Training phase did not complete")
