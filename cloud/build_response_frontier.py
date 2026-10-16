"""Private known-user observed-response frontiers with disjoint estimation."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-response-frontier-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_dir(ROOT / "tests", remote_path="/research/tests")
    .add_local_file(
        ROOT / "configs/human-response-frontier-v1.json", remote_path="/research/config.json"
    )
    .add_local_file(
        ROOT / "configs/human-response-frontier-inputs-v1.json", remote_path="/research/inputs.json"
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

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    saved = {"cloud.py": Path(__file__).read_text()}
    for name, local in [
        ("config.json", "human-response-frontier-v1.json"),
        ("inputs.json", "human-response-frontier-inputs-v1.json"),
    ]:
        saved[name] = (
            base / name if base == Path("/research") else base / "configs" / local
        ).read_text()
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    for name in ["test_response_coarsening.py", "test_response_frontier.py"]:
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
            "/research/tests/test_response_coarsening.py",
            "/research/tests/test_response_frontier.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(status="complete", source_digest=snapshot()[1], tests=result.stdout)


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=180,
    retries=0,
    max_containers=2,
    volumes={"/artifacts": volume},
)
def construct(seed: int):
    import json
    import numpy as np
    from behavioral_equivalence.response_coarsening import coarsen_response_table
    from behavioral_equivalence.response_frontier import response_frontier
    from behavioral_equivalence.response_refinement import group_response_model

    volume.reload()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    spec = next(x for x in json.loads(saved["inputs.json"]) if x["seed"] == seed)
    roots = {}
    reports = {}
    for kind in ["history", "calibration", "support"]:
        relative = Path(spec[kind + "_root"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not str(relative).startswith("human-calendar-training-")
        ):
            raise ValueError("Private artifact path required")
        root = Path("/artifacts") / relative
        if checksum(root / "report.json") != spec[kind + "_report_sha256"]:
            raise ValueError("Pinned upstream report changed")
        report = json.loads((root / "report.json").read_text())
        if (
            report["status"] != "complete"
            or report["protocol"] != "known_users"
            or report["seed"] != seed
            or any(checksum(root / name) != sha for name, sha in report["output_sha256"].items())
        ):
            raise ValueError("Complete checksum-bound upstream artifacts required")
        if kind != "history" and report["history_report_sha256"] != spec["history_report_sha256"]:
            raise ValueError("Upstream history models differ")
        roots[kind] = root
        reports[kind] = report
    if reports["calibration"]["calibration_split_seed"] != 20261015:
        raise ValueError("Frozen user split changed")
    out = roots["history"] / ("response-frontier-" + digest[:16])
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if cached["source_digest"] != digest or any(
            checksum(out / name) != sha for name, sha in cached["output_sha256"].items()
        ):
            raise ValueError("Cached frontier changed")
        return cached
    k = config["parent_states"]
    tables = {}
    with (
        np.load(roots["calibration"] / "tables.npz", allow_pickle=False) as fine,
        np.load(roots["support"] / "support.npz", allow_pickle=False) as support,
    ):
        mapping = support[f"k{k}_map"]
        for name in ["discovery", "estimation"]:
            tables[name] = coarsen_response_table(
                fine[name + "_counts"],
                fine[name + "_positives"],
                mapping,
                support[f"k{k}_{name}_counts"],
                support[f"k{k}_{name}_users"],
            )
    initial = tuple(
        tuple(range(start, start + config["initial_block_size"]))
        for start in range(0, k, config["initial_block_size"])
    )
    rows = response_frontier(
        tables["discovery"],
        tables["estimation"],
        initial,
        config["split_budgets"],
        config["control_seed"],
        min_cases=config["minimum_cases"],
        min_users=config["minimum_users"],
    )
    arrays = {"fine_to_coarse": mapping}
    records = []
    for row in rows:
        model = group_response_model(
            tables["estimation"]["counts"], tables["estimation"]["positives"], row["blocks"]
        )
        for key in ["abstract", "mapping"]:
            arrays[row["name"] + "_" + key] = model[key]
        arrays["parent"] = model["parent"]
        records.append(
            dict(
                **row,
                latent_states=len(row["blocks"]),
                maximum_point_discrepancy=model["maximum_point_discrepancy"],
            )
        )
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    np.savez_compressed(out / "models.npz", **arrays)
    (out / "partitions.json").write_text(json.dumps(records, indent=2))
    result = dict(
        status="complete",
        source_digest=digest,
        protocol="known_users",
        seed=seed,
        input_reports={kind: spec[kind + "_report_sha256"] for kind in roots},
        records=[
            {
                key: r[key]
                for key in [
                    "name",
                    "latent_states",
                    "split_budget",
                    "random_fill_states",
                    "maximum_point_discrepancy",
                ]
            }
            for r in records
        ],
        primary_candidate=config["primary_candidate"],
        primary_reference=config["primary_reference"],
        output_sha256={name: checksum(out / name) for name in ["models.npz", "partitions.json"]},
        root=str(out),
        final_test_scoring="none",
        scope=config["scope"],
        search_scope=config["search_scope"],
    )
    (out / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(stage: str = "smoke", seed: int = 20261015):
    import json
    import hashlib
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    if stage not in {"smoke", "construct"}:
        raise ValueError("Smoke or construct stage required")
    saved, digest = snapshot()
    if seed not in [r["seed"] for r in json.loads(saved["inputs.json"])]:
        raise ValueError("Fixed known-user seed required")
    identity = hashlib.sha256(json.dumps([stage, digest, seed]).encode()).hexdigest()
    amount = 0.03 if stage == "smoke" else 180 * (1.31e-5 + 2 * 2.22e-6) + 0.05
    reservation = BudgetLedger(ROOT / ".campaign/response-frontier-budget.sqlite", 1).reserve(
        identity, amount
    )
    state = ROOT / ".campaign" / ("response-frontier-" + identity[:24] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = smoke.spawn() if stage == "smoke" else construct.spawn(seed)
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
