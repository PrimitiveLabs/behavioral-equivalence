"""Retrodictive validation variance proxy and outcome-free response test eligibility."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-response-precision-v1")
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
        ROOT / "configs/human-response-precision-v1.json", remote_path="/research/config.json"
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
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "config.json": (
            base / "config.json"
            if base == Path("/research")
            else base / "configs/human-response-precision-v1.json"
        ).read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    for name in ["test_case_inference.py", "test_precision_planning.py"]:
        saved["tests/" + name] = (base / "tests" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(12288, 12288),
    timeout=900,
    retries=0,
    volumes={"/artifacts": volume},
)
def run():
    import json
    import hashlib
    import subprocess
    import numpy as np
    import torch
    from behavioral_equivalence.content_neural import build_content_model
    from behavioral_equivalence.quantization import assign_centers
    from behavioral_equivalence.case_inference import conditional_response_forecasts
    from behavioral_equivalence.precision_planning import (
        user_nll_differences,
        normal_precision_plan,
    )

    volume.reload()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    tests = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_case_inference.py",
            "/research/tests/test_precision_planning.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    torch.set_num_threads(8)
    rows = []
    for spec in config["inputs"]:
        history = Path("/artifacts") / spec["history_root"]
        frontier = Path("/artifacts") / spec["frontier_root"]
        if (
            checksum(history / "report.json") != spec["history_report_sha256"]
            or checksum(frontier / "report.json") != spec["frontier_report_sha256"]
        ):
            raise ValueError("Pinned history/frontier report changed")
        h = json.loads((history / "report.json").read_text())
        f = json.loads((frontier / "report.json").read_text())
        h_source = json.loads((history / "source.json").read_text())
        if (
            h["status"] != "complete"
            or f["status"] != "complete"
            or h["protocol"] != "known_users"
            or h["seed"] != spec["seed"]
            or f["seed"] != spec["seed"]
            or f["input_reports"]["history"] != spec["history_report_sha256"]
            or checksum(history / "model.npz") != h["output_sha256"]["model.npz"]
            or checksum(frontier / "models.npz") != f["output_sha256"]["models.npz"]
        ):
            raise ValueError("Source-bound completed fitted models required")
        for name in ["neural.py", "exposure_neural.py", "content_neural.py"]:
            key = "src/behavioral_equivalence/" + name
            if saved[key] != h_source[key]:
                raise ValueError("Frozen neural implementation changed")
        parent = history.parent
        if checksum(parent / "report.json") != h["parent_fit_report_sha256"]:
            raise ValueError("Parent fit changed")
        parent_report = json.loads((parent / "report.json").read_text())
        if any(
            checksum(parent / name) != sha for name, sha in parent_report["output_sha256"].items()
        ):
            raise ValueError("Parent weights changed")
        training_source = json.loads((parent.parents[1] / "source.json").read_text())
        training = next(
            json.loads(v)
            for k, v in training_source.items()
            if k.startswith("configs/human-training-calendar")
        )
        inputs = Path("/artifacts") / training["input_artifacts"]["known_users"]["path"]
        if checksum(inputs / "report.json") != parent_report["input_report_sha256"]:
            raise ValueError("Model vocabulary source changed")
        input_report = json.loads((inputs / "report.json").read_text())
        model = build_content_model("gru", input_report["vocabulary_size"], training["hidden"])
        model.load_state_dict(torch.load(parent / "best.pt", weights_only=True, map_location="cpu"))
        model.eval()
        hc = json.loads(h_source["config.json"])
        cohort = Path("/artifacts") / hc["random_cohort_root"]
        if (
            checksum(cohort / "report.json") != hc["random_cohort_report_sha256"]
            or checksum(cohort / hc["random_audit_relative"]) != hc["random_audit_sha256"]
        ):
            raise ValueError("Audited random cohort changed")
        cohort_report = json.loads((cohort / "report.json").read_text())
        with np.load(history / "model.npz", allow_pickle=False) as a:
            centers, mean, scale = [a[k] for k in ["centers", "mean", "scale"]]
        with np.load(frontier / "models.npz", allow_pickle=False) as a:
            parameters = {key: a[key] for key in a.files}
        pieces = []
        test_users = set()
        test_cases = 0
        test_ids = hashlib.sha256()
        validation_users = set()
        for bucket in cohort_report["buckets"]:
            path = cohort / f"bucket-{bucket['bucket']:02d}" / "randomized.npz"
            if checksum(path) != bucket["output_sha256"]["randomized.npz"]:
                raise ValueError("Random cohort bucket changed")
            with np.load(path, allow_pickle=False) as a:
                known = a["duration_group"] < 4
                validation = (a["role_code"] == 1) & known
                future = (a["role_code"] == 3) & known
                users = a["user_id"][validation]
                targets = a["target"][validation]
                tokens = a["tokens"][validation]
                env = a["environment"][validation]
                test_users.update(int(u) for u in np.unique(a["user_id"][future]))
                ids = a["case_id"][future]
                test_ids.update(ids.astype("<u8").tobytes())
                test_cases += len(ids)
            current = set(int(u) for u in np.unique(users))
            if current & validation_users:
                raise ValueError("Users crossed source buckets")
            validation_users.update(current)
            states = []
            for offset in range(0, len(users), 2048):
                with torch.inference_mode():
                    z = model.encode(
                        torch.as_tensor(tokens[offset : offset + 2048], dtype=torch.long)
                    ).numpy()
                states.append(assign_centers(z, mean, scale, centers))
            state = np.concatenate(states) if states else np.empty(0, dtype=np.int64)
            forecasts = {}
            for name in [config["primary_candidate"], config["primary_reference"]]:
                forecasts[name] = conditional_response_forecasts(
                    state,
                    env,
                    parameters["fine_to_coarse"],
                    parameters[name + "_mapping"],
                    parameters[name + "_abstract"],
                )
            pieces.append(
                dict(
                    user=users,
                    target=targets,
                    candidate=forecasts[config["primary_candidate"]],
                    reference=forecasts[config["primary_reference"]],
                )
            )
        arrays = {key: np.concatenate([p[key] for p in pieces]) for key in pieces[0]}
        differences = user_nll_differences(
            arrays["target"], arrays["reference"], arrays["candidate"], arrays["user"]
        )
        plan = normal_precision_plan(differences, config["assumed_gaps_nats"])
        rows.append(
            dict(
                seed=spec["seed"],
                protocol="known_users",
                validation_cases=len(arrays["target"]),
                **plan,
                test_eligibility=dict(
                    cases=test_cases,
                    users=len(test_users),
                    case_id_sha256=test_ids.hexdigest(),
                    selection="Recorded test role and known intrinsic duration; no predictions or outcomes used",
                ),
                history_report_sha256=spec["history_report_sha256"],
                frontier_report_sha256=spec["frontier_report_sha256"],
            )
        )
    if len({json.dumps(r["test_eligibility"], sort_keys=True) for r in rows}) != 1:
        raise ValueError("Fixed test eligibility differs across fitted models")
    root = Path("/artifacts") / ("human-response-precision-" + digest[:16])
    root.mkdir(exist_ok=True)
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    report = dict(
        status="complete",
        source_digest=digest,
        tests=tests.stdout,
        rows=rows,
        scope=config["scope"],
        validation_mean_loss_exported=False,
        final_test_predictions_generated=0,
        final_test_metrics_generated=0,
        root=str(root),
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    digest = snapshot()[1]
    reservation = BudgetLedger(ROOT / ".campaign/validation-precision-budget.sqlite", 0.5).reserve(
        "response-" + digest, 900 * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1
    )
    state = ROOT / ".campaign" / ("response-precision-" + digest[:24] + ".json")
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
    print(state.read_text(), flush=True)
    print(json.dumps(call.get()), flush=True)
