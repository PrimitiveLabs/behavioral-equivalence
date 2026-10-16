"""Disjoint private random-response calibration for a frozen history-only model."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-random-response-calibration-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.1", "numpy==2.2.6", "scipy==1.15.3", "scikit-learn==1.7.1", "pytest==8.4.1"
    )
    .env({"PYTHONPATH": "/research/src", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_dir(ROOT / "tests", remote_path="/research/tests")
    .add_local_file(
        ROOT / "configs/human-random-calibration-v1.json", remote_path="/research/config.json"
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
        else base / "configs/human-random-calibration-v1.json"
    )
    saved = {"cloud.py": Path(__file__).read_text(), "config.json": config.read_text()}
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    for name in ["test_random_response.py", "test_response_refinement.py"]:
        saved["tests/" + name] = (base / "tests" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=180, retries=0)
def smoke():
    import subprocess

    result = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_random_response.py",
            "/research/tests/test_response_refinement.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(status="complete", source_digest=snapshot()[1], tests=result.stdout)


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=900,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def calibrate(history_root: str, history_report_sha256: str):
    import json
    import hashlib
    import time
    import numpy as np
    import torch
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.quantization import assign_centers
    from behavioral_equivalence.random_response import (
        calibration_discovery_mask,
        response_table,
        supported_witnesses,
    )
    from behavioral_equivalence.response_refinement import group_response_model

    relative = Path(history_root)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not history_root.startswith("human-calendar-training-")
    ):
        raise ValueError("Private immutable history artifact path required")
    volume.reload()
    started = time.monotonic()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    history = Path("/artifacts") / relative
    if checksum(history / "report.json") != history_report_sha256:
        raise ValueError("History report changed")
    report = json.loads((history / "report.json").read_text())
    history_source = json.loads((history / "source.json").read_text())
    if (
        report["status"] != "complete"
        or hashlib.sha256(json.dumps(history_source, sort_keys=True).encode()).hexdigest()
        != report["source_digest"]
        or checksum(history / "model.npz") != report["output_sha256"]["model.npz"]
    ):
        raise ValueError("Completed history model required")
    for name in ["neural.py", "exposure_neural.py", "content_neural.py"]:
        key = "src/behavioral_equivalence/" + name
        if saved[key] != history_source[key]:
            raise ValueError("Frozen encoder implementation changed")
    history_config = json.loads(history_source["config.json"])
    cohort = Path("/artifacts") / history_config["random_cohort_root"]
    if (
        checksum(cohort / "report.json") != history_config["random_cohort_report_sha256"]
        or checksum(cohort / history_config["random_audit_relative"])
        != history_config["random_audit_sha256"]
    ):
        raise ValueError("Audited random cohort changed")
    cohort_report = json.loads((cohort / "report.json").read_text())
    parent = history.parent
    if checksum(parent / "report.json") != report["parent_fit_report_sha256"]:
        raise ValueError("Parent GRU report changed")
    parent_report = json.loads((parent / "report.json").read_text())
    if any(checksum(parent / name) != sha for name, sha in parent_report["output_sha256"].items()):
        raise ValueError("Frozen GRU outputs changed")
    training_root = parent.parents[1]
    training_source = json.loads((training_root / "source.json").read_text())
    if (
        hashlib.sha256(json.dumps(training_source, sort_keys=True).encode()).hexdigest()
        != parent_report["source_digest"]
    ):
        raise ValueError("Parent training source changed")
    training = next(
        json.loads(v)
        for k, v in training_source.items()
        if k.startswith("configs/human-training-calendar")
    )
    inputs = Path("/artifacts") / training["input_artifacts"][report["protocol"]]["path"]
    if (
        checksum(inputs / "report.json")
        != training["input_artifacts"][report["protocol"]]["report_sha256"]
    ):
        raise ValueError("Model vocabulary source changed")
    input_report = json.loads((inputs / "report.json").read_text())
    model = build_content_model("gru", input_report["vocabulary_size"], training["hidden"])
    model.load_state_dict(torch.load(parent / "best.pt", weights_only=True, map_location="cpu"))
    model.eval()
    torch.set_num_threads(8)
    with np.load(history / "model.npz", allow_pickle=False) as arrays:
        centers, mean, scale = [arrays[k] for k in ["centers", "mean", "scale"]]
    out = history / ("random-calibration-" + digest[:12] + "-" + history_report_sha256[:12])
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if (
            cached["source_digest"] != digest
            or cached["history_report_sha256"] != history_report_sha256
            or any(checksum(out / name) != sha for name, sha in cached["output_sha256"].items())
        ):
            raise ValueError("Cached calibration changed")
        return cached
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    k = len(centers)
    tables = {
        name: {
            "counts": np.zeros((k, 75), dtype=np.int64),
            "positives": np.zeros((k, 75)),
            "users": np.zeros((k, 75), dtype=np.int64),
        }
        for name in ["discovery", "estimation"]
    }
    users_seen = set()
    counts = {name: dict(cases=0, users=0) for name in tables}
    for bucket in cohort_report["buckets"]:
        path = cohort / f"bucket-{bucket['bucket']:02d}" / "randomized.npz"
        if checksum(path) != bucket["output_sha256"]["randomized.npz"]:
            raise ValueError("Random exposure file changed")
        with np.load(path, allow_pickle=False) as arrays:
            chosen = arrays["role_code"] == 2
            if report["protocol"] == "unseen_users":
                chosen &= arrays["user_role_code"] == 2
            users = arrays["user_id"][chosen]
            env = arrays["environment"][chosen]
            targets = arrays["target"][chosen]
            tokens = arrays["tokens"][chosen]
        current = set(int(u) for u in np.unique(users))
        if current & users_seen:
            raise ValueError("User appeared in multiple source partitions")
        users_seen.update(current)
        states = []
        batch = config["inference_batch_size"]
        for offset in range(0, len(users), batch):
            with torch.inference_mode():
                z = model.encode(
                    torch.as_tensor(tokens[offset : offset + batch], dtype=torch.long)
                ).numpy()
            states.append(assign_centers(z, mean, scale, centers))
        state = np.concatenate(states) if states else np.array([], dtype=np.int64)
        discovery = calibration_discovery_mask(users, config["split_seed"])
        for name, mask in [("discovery", discovery), ("estimation", ~discovery)]:
            table = response_table(state[mask], env[mask], users[mask], targets[mask], k)
            for key in tables[name]:
                tables[name][key] += table[key]
            counts[name]["cases"] += int(mask.sum())
            counts[name]["users"] += len(np.unique(users[mask]))
    for table in tables.values():
        table["probabilities"] = np.full((k, 75), np.nan)
        np.divide(
            table["positives"],
            table["counts"],
            out=table["probabilities"],
            where=table["counts"] > 0,
        )
    blocks = tuple(tuple(b) for b in report["initial_blocks"])
    witnesses = supported_witnesses(
        tables["discovery"],
        blocks,
        min_users=config["minimum_users_per_state_environment"],
        min_cases=config["minimum_cases_per_state_environment"],
    )
    fitted = group_response_model(
        tables["estimation"]["counts"], tables["estimation"]["positives"], blocks
    )
    arrays = {
        f"{name}_{key}": value for name, table in tables.items() for key, value in table.items()
    }
    arrays.update(
        {
            f"estimated_{key}": fitted[key]
            for key in ["parent", "abstract", "mapping", "absolute_discrepancy"]
        }
    )
    np.savez_compressed(out / "tables.npz", **arrays)
    (out / "witnesses.json").write_text(json.dumps(witnesses, indent=2))
    support = {}
    for name, table in tables.items():
        supported = (table["counts"] >= config["minimum_cases_per_state_environment"]) & (
            table["users"] >= config["minimum_users_per_state_environment"]
        )
        supported[:, 4::5] = False
        support[name] = dict(
            **counts[name],
            supported_known_duration_state_cells=int(supported.sum()),
            populated_state_environment_cells=int(np.count_nonzero(table["counts"])),
        )
    result = dict(
        status="complete",
        source_digest=digest,
        history_report_sha256=history_report_sha256,
        protocol=report["protocol"],
        seed=report["seed"],
        calibration_split_seed=config["split_seed"],
        support=support,
        supported_merged_state_pair_witnesses=len(witnesses),
        output_sha256={name: checksum(out / name) for name in ["tables.npz", "witnesses.json"]},
        root=str(out),
        wall_seconds=time.monotonic() - started,
        final_test_scoring="none",
        scope=config["scope"],
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(stage: str = "smoke", history_root: str = "", history_report_sha256: str = ""):
    import json
    import hashlib
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    if stage not in {"smoke", "calibrate"}:
        raise ValueError("Smoke or calibration required")
    if stage == "calibrate" and (
        len(history_report_sha256) != 64
        or any(c not in "0123456789abcdef" for c in history_report_sha256)
    ):
        raise ValueError("Pinned history report SHA256 required")
    digest = snapshot()[1]
    identity = hashlib.sha256(
        json.dumps([stage, digest, history_root, history_report_sha256]).encode()
    ).hexdigest()
    amount = 0.03 if stage == "smoke" else 900 * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1
    reservation = BudgetLedger(
        ROOT / ".campaign/random-response-calibration-budget.sqlite", 3
    ).reserve(identity, amount)
    state = ROOT / ".campaign" / ("random-calibration-" + identity[:24] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = (
            smoke.spawn()
            if stage == "smoke"
            else calibrate.spawn(history_root, history_report_sha256)
        )
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id, source_digest=digest, reservation=reservation
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    print(json.dumps(call.get()), flush=True)
