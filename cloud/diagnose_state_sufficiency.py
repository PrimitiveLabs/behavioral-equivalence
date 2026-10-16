"""Post-hoc within-state history/context diagnostics on saved human forecasts."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-state-sufficiency-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/tools:/research/src", "OPENBLAS_NUM_THREADS": "2"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tools/state_sufficiency.py", remote_path="/research/tools/state_sufficiency.py"
    )
    .add_local_file(
        ROOT / "tests/test_state_sufficiency.py",
        remote_path="/research/tests/test_state_sufficiency.py",
    )
    .add_local_file(
        ROOT / "configs/human-supplementary-final-v1.json", remote_path="/research/config.json"
    )
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 * 1024**2):
            h.update(chunk)
    return h.hexdigest()


def snapshot():
    import hashlib
    import json

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    source = {
        "cloud.py": Path(__file__).read_text(),
        "tools/state_sufficiency.py": (base / "tools/state_sufficiency.py").read_text(),
        "tests/test_state_sufficiency.py": (base / "tests/test_state_sufficiency.py").read_text(),
        "config.json": (
            base / ("config.json" if base != ROOT else "configs/human-supplementary-final-v1.json")
        ).read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        source["src/" + p.name] = p.read_text()
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
def diagnose(inputs):
    import json
    import time
    import subprocess
    import numpy as np
    from state_sufficiency import cell_totals, combine_cells
    from behavioral_equivalence.case_inference import available_history_lengths

    started = time.monotonic()
    volume.reload()
    source, digest = snapshot()
    cfg = json.loads(source["config.json"])
    expected = {(p, s) for p in ["known_users", "unseen_users"] for s in range(20261015, 20261020)}
    if len(inputs) != 10 or {(r["protocol"], r["seed"]) for r in inputs} != expected:
        raise ValueError("Complete fixed fit matrix required")
    tests = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_state_sufficiency.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    cases = Path("/artifacts") / cfg["case_root"]
    context = Path("/artifacts") / cfg["context_root"]
    if (
        checksum(cases / "report.json") != cfg["case_report_sha256"]
        or checksum(context / "report.json") != cfg["context_report_sha256"]
    ):
        raise ValueError("Audited case/context manifest changed")
    cr = json.loads((cases / "report.json").read_text())
    ir = json.loads((context / "report.json").read_text())
    results = []
    for entry in inputs:
        primary = Path(entry["primary_root"])
        supplement = Path(entry["supplementary_root"])
        if (
            checksum(primary / "report.json") != entry["primary_report_sha256"]
            or checksum(supplement / "report.json") != entry["supplementary_report_sha256"]
        ):
            raise ValueError("Saved forecast report changed")
        pr = json.loads((primary / "report.json").read_text())
        sr = json.loads((supplement / "report.json").read_text())
        if (pr["protocol"], pr["seed"]) != (entry["protocol"], entry["seed"]) or (
            sr["protocol"],
            sr["seed"],
        ) != (entry["protocol"], entry["seed"]):
            raise ValueError("Forecast fit identities differ")
        blocks = {key: [] for key in ["history_length", "context_missingness", "tab"]}
        seen = set()
        total = 0
        for bucket in range(32):
            # Primary forecasts and supplementary state IDs use the same canonical cases.
            primary_name = f"bucket-{bucket:02d}-test-forecasts.npz"
            supplement_name = f"bucket-{bucket:02d}-factorized.npz"
            if (
                checksum(primary / primary_name) != pr["output_sha256"][primary_name]
                or checksum(supplement / supplement_name) != sr["output_sha256"][supplement_name]
            ):
                raise ValueError("Forecast/state artifact changed")
            name = "embargo-24h.npz"
            packed = cases / f"bucket-{bucket:02d}" / name
            intrinsic = context / f"bucket-{bucket:02d}" / name
            if (
                checksum(packed) != cr["buckets"][bucket]["output_sha256"][name]
                or checksum(intrinsic) != ir["buckets"][bucket]["output_sha256"][name]
            ):
                raise ValueError("Canonical source bucket changed")
            with (
                np.load(primary / primary_name, allow_pickle=False) as p,
                np.load(supplement / supplement_name, allow_pickle=False) as s,
                np.load(packed, allow_pickle=False) as a,
                np.load(intrinsic, allow_pickle=False) as c,
            ):
                ids = p["case_id"]
                users = p["user_id"]
                states = s["k256_state"]
                if (
                    not np.array_equal(ids, s["case_id"])
                    or not np.array_equal(users, s["user_id"])
                    or not np.array_equal(a["case_id"], c["case_id"])
                ):
                    raise ValueError("Case identities differ across frozen sources")
                ix = np.flatnonzero(np.isin(a["case_id"], ids))
                if not np.array_equal(a["case_id"][ix], ids) or not np.array_equal(
                    a["user_id"][ix], users
                ):
                    raise ValueError("Forecast origins differ from original source order")
                unique = set(map(int, np.unique(users)))
                if seen & unique:
                    raise ValueError("Original users cross source buckets")
                seen.update(unique)
                lengths = available_history_lengths(a["tokens"][ix])
                history = np.select(
                    [lengths == 0, lengths <= 8, lengths <= 16], [0, 1, 2], default=3
                )
                categories = {
                    "history_length": (history, 4),
                    "context_missingness": (c["context"][ix, 1].astype(int), 2),
                    "tab": (a["tab"][ix], 15),
                }
                for axis, (values, count) in categories.items():
                    blocks[axis].append(
                        cell_totals(users, states, values, p["target"], p["parent_k256"], count)
                    )
                total += len(ids)
        result = dict(
            protocol=entry["protocol"],
            seed=entry["seed"],
            cases=total,
            users=len(seen),
            axes={axis: combine_cells(values) for axis, values in blocks.items()},
        )
        results.append(result)
        print(
            json.dumps(dict(protocol=entry["protocol"], seed=entry["seed"], status="complete")),
            flush=True,
        )
    out = Path("/artifacts") / ("state-sufficiency-" + digest[:16])
    out.mkdir(exist_ok=True)
    report = dict(
        status="complete",
        source_digest=digest,
        inputs=inputs,
        results=results,
        tests=tests.stdout,
        wall_seconds=time.monotonic() - started,
        scope="Post-hoc descriptive within-parent-state response heterogeneity by history length, context missingness and tab. Equal users within each cell; published rates require50users and2000cases. No causal, Markov-sufficiency, significance or adjusted-discovery claim; unsupported comparisons are retained.",
    )
    (out / "source.json").write_text(json.dumps(source, sort_keys=True))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    import json
    import hashlib
    import fcntl
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    inputs = []
    for protocol in ["known_users", "unseen_users"]:
        for seed in range(20261015, 20261020):
            primary = (
                ROOT / "reports/final-human" / f"one_step-{protocol}-{seed}-24.json"
            ).read_bytes()
            supplement = (
                ROOT / "reports/supplementary-final" / f"{protocol}-{seed}.json"
            ).read_bytes()
            p, s = json.loads(primary), json.loads(supplement)
            if p["status"] != "complete" or s["status"] != "complete":
                raise ValueError("Complete frozen scoring reports required")
            inputs.append(
                dict(
                    protocol=protocol,
                    seed=seed,
                    primary_root=p["root"],
                    supplementary_root=s["root"],
                    primary_report_sha256=hashlib.sha256(primary).hexdigest(),
                    supplementary_report_sha256=hashlib.sha256(supplement).hexdigest(),
                )
            )
    _, digest = snapshot()
    batch = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    lock = (ROOT / ".campaign/state-sufficiency.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reservation = BudgetLedger(ROOT / ".campaign/calibrated-parent-budget.sqlite", 25).reserve(
        "state-sufficiency-" + digest + "-" + batch, 0.25
    )
    path = ROOT / ".campaign" / ("state-sufficiency-" + digest[:16] + "-" + batch[:16] + ".json")
    if path.exists():
        call = modal.FunctionCall.from_id(json.loads(path.read_text())["function_call_id"])
    else:
        with path.with_suffix(".submission-attempt").open("x") as marker:
            marker.write(digest + "\n")
        call = diagnose.spawn(inputs)
        path.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id, source_digest=digest, reservation=reservation
                ),
                indent=2,
            )
        )
    print(path.read_text(), flush=True)
    result = call.get()
    (ROOT / "reports/preflight" / ("state-sufficiency-" + digest[:16] + ".json")).write_text(
        json.dumps(result, indent=2)
    )
    print(
        json.dumps(
            dict(
                status=result["status"],
                fits=len(result["results"]),
                wall_seconds=result["wall_seconds"],
            )
        ),
        flush=True,
    )
