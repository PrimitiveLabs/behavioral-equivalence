"""Independent PRISM MDP check of small interval models with zero lower edges."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-interval-vertex-crosscheck-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("default-jre-headless", "curl")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pydantic==2.11.7", "pytest==8.4.1")
    .run_commands(
        "curl -fL https://github.com/prismmodelchecker/prism/releases/download/v4.11/prism-4.11-linux64-x86.tar.gz -o /tmp/prism.tar.gz",
        "echo '6d4677b6ab64daa16a8bb5ad05f79c46313ad67835cb8445ee770e7f57f611c8  /tmp/prism.tar.gz' | sha256sum -c -",
        "mkdir /opt/prism && tar -xzf /tmp/prism.tar.gz -C /opt/prism --strip-components=1",
        "cd /opt/prism && ./install.sh",
    )
    .env({"PYTHONPATH": "/research/src:/research/tools", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tools/interval_vertices.py", remote_path="/research/tools/interval_vertices.py"
    )
    .add_local_file(
        ROOT / "tests/test_interval_vertices.py",
        remote_path="/research/tests/test_interval_vertices.py",
    )
)


def snapshot():
    import hashlib
    import json

    base = Path("/research") if Path("/research/tools/interval_vertices.py").exists() else ROOT
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "tools/interval_vertices.py": (base / "tools/interval_vertices.py").read_text(),
        "tests/test_interval_vertices.py": (base / "tests/test_interval_vertices.py").read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=900,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def verify():
    import hashlib
    import json
    import time
    import subprocess
    from fractions import Fraction
    import numpy as np
    from behavioral_equivalence.uncertainty import IntervalChain
    from behavioral_equivalence.integer_bounds import robust_enclosures
    from behavioral_equivalence.prism import check
    from interval_vertices import write_vertex_mdp, vertex_reachability

    started = time.monotonic()
    saved, digest = snapshot()
    root = Path("/artifacts") / ("interval-vertex-crosscheck-" + digest[:16])
    root.mkdir(exist_ok=True)
    tests = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_interval_vertices.py",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    fixtures = [
        ("two", [[0, 0.25], [0, 1]], [[0.75, 1], [0, 1]]),
        (
            "simplex",
            [[0, 0, 0], [0, 0.25, 0.25], [0, 0, 1]],
            [[1, 1, 1], [0.5, 0.5, 0.75], [0, 0, 1]],
        ),
        (
            "capped",
            [[0.25, 0, 0.25], [0, 0.25, 0.25], [0, 0, 1]],
            [[0.5, 0.5, 0.5], [0.5, 0.5, 0.75], [0, 0, 1]],
        ),
        (
            "four",
            [[0, 0.25, 0, 0.25], [0.25, 0, 0.25, 0], [0, 0, 0.5, 0.25], [0, 0, 0, 1]],
            [[0.5, 0.5, 0.25, 0.75], [0.5, 0.5, 0.5, 0.5], [0.25, 0.5, 0.75, 0.5], [0, 0, 0, 1]],
        ),
    ]
    horizons = [0, 1, 5, 10, 25, 50]
    records = []
    outputs = {}
    maximum = 0
    for name, lo, hi in fixtures:
        n = len(lo)
        model = IntervalChain(np.array(lo), np.array(hi), ("active",) * (n - 1) + ("complete",))
        cert = robust_enclosures(model, "complete", horizons)
        for horizon in horizons:
            for maximize, mode in [(False, "min"), (True, "max")]:
                exact = vertex_reachability(model, "complete", horizon, maximize)
                for state in range(n):
                    low = Fraction(
                        cert["values"][str(horizon)][mode + "_lower"][state], 1 << cert["bits"]
                    )
                    high = Fraction(
                        cert["values"][str(horizon)][mode + "_upper"][state], 1 << cert["bits"]
                    )
                    if not low <= exact[state] <= high:
                        raise ValueError(
                            "Independent exact vertex oracle outside directed robust bounds"
                        )
                    path = root / f"{name}-state{state}.prism"
                    counts = write_vertex_mdp(model, path, state)
                    log = root / f"{name}-state{state}-{mode}-h{horizon}.log"
                    value = check(path, "complete", horizon, log, extremum=mode)
                    error = abs(value - float(exact[state]))
                    maximum = max(maximum, error)
                    if error > 1e-10:
                        raise ValueError("PRISM MDP differs from exact interval vertex semantics")
                    records.append(
                        dict(
                            fixture=name,
                            state=state,
                            horizon=horizon,
                            extremum=mode,
                            prism=value,
                            exact_numerator=str(exact[state].numerator),
                            exact_denominator=str(exact[state].denominator),
                            directed_lower=str(low),
                            directed_upper=str(high),
                            vertex_counts=counts,
                        )
                    )
                    for artifact in [path, log]:
                        outputs[artifact.name] = hashlib.sha256(artifact.read_bytes()).hexdigest()
    result = dict(
        status="complete",
        source_digest=digest,
        fixtures=len(fixtures),
        checks=len(records),
        horizons=horizons,
        records=records,
        output_sha256=outputs,
        tests=tests.stdout,
        maximum_prism_difference=maximum,
        wall_seconds=time.monotonic() - started,
        backend="Pinned PRISM4.11 explicit ordinary MDP engine, standard bounded Pmin/Pmax",
        scope="Small dyadic interval-simplex fixtures, including represented zero lower edges; exact vertex enumeration and independent PRISM MDP solving. Exponential vertex construction restricted to at most8 coordinates. Does not claim external-checker coverage of every large uncertainty-benchmark model.",
    )
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    (root / "report.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    source, digest = snapshot()
    state = ROOT / ".campaign" / ("interval-vertex-crosscheck-" + digest[:16] + ".json")
    print(
        json.dumps(
            BudgetLedger(ROOT / ".campaign/uncertainty-budget.sqlite", 8).reserve(
                "interval-vertex-crosscheck-" + digest, 0.05
            )
        ),
        flush=True,
    )
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        with state.with_suffix(".submission-attempt").open("x") as marker:
            marker.write(digest + "\n")
        call = verify.spawn()
        state.write_text(
            json.dumps(dict(source_digest=digest, function_call_id=call.object_id), indent=2)
        )
    print(state.read_text(), flush=True)
    (
        ROOT / "reports/source-snapshots" / ("interval-vertex-crosscheck-" + digest[:16] + ".json")
    ).write_text(json.dumps(source, indent=2) + "\n")
    result = call.get()
    (
        ROOT / "reports/preflight" / ("interval-vertex-crosscheck-" + digest[:16] + ".json")
    ).write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ["status", "checks", "maximum_prism_difference", "wall_seconds"]}
        ),
        flush=True,
    )
