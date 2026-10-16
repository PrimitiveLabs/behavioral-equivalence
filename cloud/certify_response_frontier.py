"""Exact-normalized one-step certificates for fitted response point models."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-response-certificates-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_dir(ROOT / "tests", remote_path="/research/tests")
    .add_local_file(
        ROOT / "configs/human-response-certificate-v1.json", remote_path="/research/config.json"
    )
    .add_local_file(
        ROOT / "configs/human-response-certificate-inputs-v1.json",
        remote_path="/research/inputs.json",
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
        ("config.json", "human-response-certificate-v1.json"),
        ("inputs.json", "human-response-certificate-inputs-v1.json"),
    ]:
        saved[name] = (
            base / name if base == Path("/research") else base / "configs" / local
        ).read_text()
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    for name in ["test_response_chain.py"]:
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
            "/research/tests/test_response_chain.py",
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
    import gzip
    from fractions import Fraction
    import numpy as np
    from behavioral_equivalence.response_chain import response_chain_pair
    from behavioral_equivalence.integer_bounds import certify_abstraction, verify_abstraction

    volume.reload()
    saved, digest = snapshot()
    spec = next(x for x in json.loads(saved["inputs.json"]) if x["seed"] == seed)
    relative = Path(spec["root"])
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or not str(relative).startswith("human-calendar-training-")
    ):
        raise ValueError("Private frontier artifact path required")
    root = Path("/artifacts") / relative
    if checksum(root / "report.json") != spec["report_sha256"]:
        raise ValueError("Frontier report changed")
    report = json.loads((root / "report.json").read_text())
    if (
        report["status"] != "complete"
        or report["seed"] != seed
        or report["protocol"] != "known_users"
        or len(report["records"]) != 20
        or any(checksum(root / name) != sha for name, sha in report["output_sha256"].items())
    ):
        raise ValueError("Completed checksum-bound frontier required")
    out = root / ("integer-certificates-" + digest[:16])
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if (
            cached["source_digest"] != digest
            or checksum(out / "certificates.jsonl.gz") != cached["certificate_sha256"]
        ):
            raise ValueError("Cached certificate evidence changed")
        return cached
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    partitions = json.loads((root / "partitions.json").read_text())
    properties = {"positive": ("positive",), "negative": ("negative",)}

    def pair(arrays, name, environment):
        row = next(r for r in partitions if r["name"] == name)
        return response_chain_pair(
            arrays["parent"][:, environment],
            arrays[name + "_abstract"][:, environment],
            tuple(tuple(b) for b in row["blocks"]),
        )

    count = 0
    primary_radius = Fraction(0)
    with np.load(root / "models.npz", allow_pickle=False) as arrays:
        with (out / "certificates.jsonl.gz").open("wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as archive:
                for row in partitions:
                    for environment in range(75):
                        parent, abstract, blocks = pair(arrays, row["name"], environment)
                        certificate = certify_abstraction(
                            parent, abstract, blocks, [1], bits=48, properties=properties
                        )
                        if row["name"] == report["primary_candidate"]:
                            value = certificate["epsilon"]
                            primary_radius = max(
                                primary_radius,
                                Fraction(
                                    int(value["numerator_hex"], 16),
                                    int(value["denominator_hex"], 16),
                                ),
                            )
                        record = dict(
                            name=row["name"], environment=environment, certificate=certificate
                        )
                        archive.write((json.dumps(record, sort_keys=True) + "\n").encode())
                        count += 1
        replayed = 0
        with gzip.open(out / "certificates.jsonl.gz", "rt") as archive:
            for line in archive:
                row = json.loads(line)
                parent, abstract, _ = pair(arrays, row["name"], row["environment"])
                verify_abstraction(parent, abstract, row["certificate"])
                replayed += 1
    if count != 1500 or replayed != count:
        raise ValueError("All20 models and75 contexts require replay")
    result = dict(
        status="complete",
        source_digest=digest,
        frontier_report_sha256=spec["report_sha256"],
        protocol="known_users",
        seed=seed,
        certificates=count,
        replayed=replayed,
        primary_max_context_radius=dict(
            numerator_hex=hex(primary_radius.numerator),
            denominator_hex=hex(primary_radius.denominator),
            display=float(primary_radius),
        ),
        certificate_sha256=checksum(out / "certificates.jsonl.gz"),
        root=str(out),
        scope="Exact-normalized serialized fitted point chains; one-step positive/negative outcomes over all75 fixed contexts. Arithmetic replay uses existing verifier, not independent theorem review. No human calibration, trajectory, confidence or causal guarantee.",
        final_test_scoring="none",
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
    reservation = BudgetLedger(ROOT / ".campaign/response-certificate-budget.sqlite", 1).reserve(
        identity, amount
    )
    state = ROOT / ".campaign" / ("response-certificate-" + identity[:24] + ".json")
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
