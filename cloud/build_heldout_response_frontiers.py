"""Six calibration-group holdouts with neutral certificates and exact witnesses."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-heldout-response-frontiers-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "scipy==1.15.3", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_dir(ROOT / "tests", remote_path="/research/tests")
    .add_local_file(
        ROOT / "configs/human-heldout-frontier-v1.json", remote_path="/research/config.json"
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
            else base / "configs/human-heldout-frontier-v1.json"
        ).read_text(),
    }
    for p in sorted((base / "src/behavioral_equivalence").glob("*.py")):
        saved["src/behavioral_equivalence/" + p.name] = p.read_text()
    for name in [
        "test_heldout_response.py",
        "test_heldout_frontier.py",
        "test_neutral_response.py",
        "test_response_frontier.py",
    ]:
        saved["tests/" + name] = (base / "tests" / name).read_text()
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(image=image, cpu=(1, 1), memory=(2048, 2048), timeout=180, retries=0)
def smoke():
    import subprocess

    r = subprocess.run(
        [
            "python",
            "-m",
            "pytest",
            "-q",
            "-W",
            "error",
            "/research/tests/test_heldout_response.py",
            "/research/tests/test_heldout_frontier.py",
            "/research/tests/test_neutral_response.py",
            "/research/tests/test_response_frontier.py",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return dict(status="complete", source_digest=snapshot()[1], tests=r.stdout)


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(4096, 4096),
    timeout=600,
    retries=0,
    max_containers=2,
    volumes={"/artifacts": volume},
)
def construct(seed: int):
    import json
    import gzip
    from fractions import Fraction
    import numpy as np
    from behavioral_equivalence.response_coarsening import coarsen_response_table
    from behavioral_equivalence.heldout_response import response_group_folds, heldout_point_model
    from behavioral_equivalence.heldout_frontier import heldout_refinement_frontier
    from behavioral_equivalence.neutral_response import (
        neutral_training_probabilities,
        response_gap_witness,
    )
    from behavioral_equivalence.response_chain import binary_response_chain, response_chain_pair
    from behavioral_equivalence.core import quotient
    from behavioral_equivalence.integer_bounds import certify_abstraction, verify_abstraction

    volume.reload()
    saved, digest = snapshot()
    config = json.loads(saved["config.json"])
    spec = next(s for s in config["inputs"] if s["seed"] == seed)
    roots = {}
    reports = {}
    for kind in ["history", "calibration", "support", "prior"]:
        relative = Path(spec[kind + "_root"])
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not str(relative).startswith("human-calendar-training-")
        ):
            raise ValueError("Private frozen input path required")
        root = Path("/artifacts") / relative
        if checksum(root / "report.json") != spec[kind + "_report_sha256"]:
            raise ValueError("Pinned upstream report changed")
        r = json.loads((root / "report.json").read_text())
        if (
            r["status"] != "complete"
            or r["seed"] != seed
            or r["protocol"] != "known_users"
            or any(checksum(root / name) != sha for name, sha in r["output_sha256"].items())
        ):
            raise ValueError("Complete source-bound upstream artifacts required")
        if kind != "history" and r["history_report_sha256"] != spec["history_report_sha256"]:
            raise ValueError("History state maps differ")
        roots[kind] = root
        reports[kind] = r
    out = roots["history"] / ("heldout-response-frontier-" + digest[:16])
    if (out / "report.json").exists():
        cached = json.loads((out / "report.json").read_text())
        if cached["source_digest"] != digest or any(
            checksum(out / name) != sha for name, sha in cached["output_sha256"].items()
        ):
            raise ValueError("Cached heldout frontier changed")
        return {k: v for k, v in cached.items() if k != "records"}
    with np.load(roots["prior"] / "priors.npz", allow_pickle=False) as a:
        prior = a["coarse_probabilities"]
        training_counts = a["coarse_counts"]
        fine_to_coarse = a["fine_to_coarse"]
    neutral = neutral_training_probabilities(prior, training_counts)
    tables = {}
    with (
        np.load(roots["calibration"] / "tables.npz", allow_pickle=False) as fine,
        np.load(roots["support"] / "support.npz", allow_pickle=False) as support,
    ):
        if not np.array_equal(fine_to_coarse, support["k32_map"]):
            raise ValueError("Training and calibration state coarsenings differ")
        for name in ["discovery", "estimation"]:
            tables[name] = coarsen_response_table(
                fine[name + "_counts"],
                fine[name + "_positives"],
                fine_to_coarse,
                support[f"k32_{name}_counts"],
                support[f"k32_{name}_users"],
            )
    folds = response_group_folds()
    if [(tuple(f["query"]), tuple(f["heldout"])) for f in config["folds"]] != [
        (f["query"], f["heldout"]) for f in folds
    ]:
        raise ValueError("Every fixed two-of-four group holdout is required")
    partitions = []
    arrays = {"fine_to_coarse": fine_to_coarse, "neutral_parent": neutral}
    summaries = []
    epsilon = config["neutral_epsilon"]
    budget = Fraction.from_float(float(epsilon))
    for fold_index, fold in enumerate(folds):
        prefix = "q" + "".join(map(str, fold["query"]))
        rows = heldout_refinement_frontier(
            neutral,
            tables["discovery"],
            tables["estimation"],
            fold["query"],
            config["split_budgets"],
            config["control_seed"] + fold_index * config["seed_fold_offset"],
            epsilon=epsilon,
            min_cases=config["minimum_cases"],
            min_users=config["minimum_users"],
        )
        if len(rows) != 20:
            raise ValueError("All targeted and matched-control variants required")
        for split_budget in config["split_budgets"]:
            if len({len(r["blocks"]) for r in rows if r["split_budget"] == split_budget}) != 1:
                raise ValueError("Control capacities differ")
        for row in rows:
            value = row["neutral_certificate"]["epsilon"]
            radius = Fraction(int(value["numerator_hex"], 16), int(value["denominator_hex"], 16))
            if radius > budget:
                raise ValueError("Neutral precision was not preserved")
            model = heldout_point_model(
                prior, tables["estimation"], row["blocks"], fold["query"], config["prior_mass"]
            )
            arrays[prefix + "_parent"] = model["parent"]
            key = prefix + "_" + row["name"]
            arrays[key + "_abstract"] = model["abstract"]
            arrays[key + "_mapping"] = model["mapping"]
            partitions.append(
                dict(fold=prefix, query_groups=fold["query"], heldout_groups=fold["heldout"], **row)
            )
            summaries.append(
                dict(
                    fold=prefix,
                    name=row["name"],
                    query_groups=fold["query"],
                    heldout_groups=fold["heldout"],
                    latent_states=len(row["blocks"]),
                    split_budget=row["split_budget"],
                    directed_witnesses=len(row["trace"]),
                    neutral_repair_states=row["neutral_repair_states"],
                    capacity_fill_states=row["capacity_fill_states"],
                    neutral_radius_for_display=float(radius),
                )
            )
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    np.savez_compressed(out / "models.npz", **arrays)
    (out / "partitions.json").write_text(json.dumps(partitions, indent=2))
    count = 0
    maxima = {}

    def pair(a, record, environment):
        key = record["fold"] + "_" + record["name"]
        return response_chain_pair(
            a[record["fold"] + "_parent"][:, environment],
            a[key + "_abstract"][:, environment],
            tuple(tuple(b) for b in record["blocks"]),
        )

    with (out / "certificates.jsonl.gz").open("wb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0, filename="") as archive:
            for row in partitions:
                key = row["fold"] + "_" + row["name"]
                maxima[key] = {
                    scope: Fraction(0) for scope in ["all", "known", "queried", "heldout"]
                }
                for environment in range(75):
                    parent, abstract, blocks = pair(arrays, row, environment)
                    certificate = certify_abstraction(
                        parent,
                        abstract,
                        blocks,
                        [1],
                        bits=48,
                        properties={"positive": ("positive",), "negative": ("negative",)},
                    )
                    value = certificate["epsilon"]
                    radius = Fraction(
                        int(value["numerator_hex"], 16), int(value["denominator_hex"], 16)
                    )
                    group = environment % 5
                    for scope, include in [
                        ("all", True),
                        ("known", group < 4),
                        ("queried", group in row["query_groups"]),
                        ("heldout", group in row["heldout_groups"]),
                    ]:
                        if include:
                            maxima[key][scope] = max(maxima[key][scope], radius)
                    archive.write(
                        (
                            json.dumps(
                                dict(
                                    fold=row["fold"],
                                    name=row["name"],
                                    environment=environment,
                                    certificate=certificate,
                                ),
                                sort_keys=True,
                            )
                            + "\n"
                        ).encode()
                    )
                    count += 1
    lookup = {(r["fold"], r["name"]): r for r in partitions}
    replayed = 0
    neutral_replayed = 0
    witnesses_replayed = 0
    with np.load(out / "models.npz", allow_pickle=False) as a:
        with gzip.open(out / "certificates.jsonl.gz", "rt") as archive:
            for line in archive:
                record = json.loads(line)
                row = lookup[(record["fold"], record["name"])]
                parent, abstract, _ = pair(a, row, record["environment"])
                verify_abstraction(parent, abstract, record["certificate"])
                replayed += 1
        for row in json.loads((out / "partitions.json").read_text()):
            parent = binary_response_chain(a["neutral_parent"])
            blocks = tuple(tuple(b) for b in row["blocks"]) + ((32,), (33,))
            abstract, _ = quotient(parent, blocks)
            verify_abstraction(parent, abstract, row["neutral_certificate"])
            neutral_replayed += 1
            for witness in row.get("witness_certificates", []):
                expected = response_gap_witness(
                    tables["discovery"]["counts"],
                    tables["discovery"]["positives"],
                    witness["states"],
                    witness["environment"],
                    epsilon,
                )
                if (
                    expected != witness
                    or not witness["exceeds_budget"]
                    or witness["environment"] % 5 not in row["query_groups"]
                ):
                    raise ValueError("Fitted-count witness certificate differs")
                witnesses_replayed += 1
    if count != 9000 or replayed != count or neutral_replayed != 120:
        raise ValueError("Complete six-fold/context certificate replay required")
    for row in summaries:
        row["one_step_context_radii"] = {
            scope: dict(
                numerator_hex=hex(value.numerator),
                denominator_hex=hex(value.denominator),
                display=float(value),
            )
            for scope, value in maxima[row["fold"] + "_" + row["name"]].items()
        }
    report = dict(
        status="complete",
        source_digest=digest,
        seed=seed,
        protocol="known_users",
        input_reports={kind: spec[kind + "_report_sha256"] for kind in roots},
        folds=6,
        models=len(partitions),
        point_certificates=count,
        point_certificates_replayed=replayed,
        neutral_certificates_replayed=neutral_replayed,
        fitted_count_witnesses_replayed=witnesses_replayed,
        initial_neutral_states=next(
            r["latent_states"] for r in summaries if r["name"] == "unrefined"
        ),
        records=summaries,
        output_sha256={
            name: checksum(out / name)
            for name in ["models.npz", "partitions.json", "certificates.jsonl.gz"]
        },
        root=str(out),
        heldout_random_calibration_outcomes_used=False,
        final_test_scoring="none",
        scope=config["scope"],
    )
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return {k: v for k, v in report.items() if k != "records"}


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
    config = json.loads(saved["config.json"])
    if seed not in [r["seed"] for r in config["inputs"]]:
        raise ValueError("Fixed known-user seed required")
    identity = hashlib.sha256(json.dumps([stage, digest, seed]).encode()).hexdigest()
    amount = 0.03 if stage == "smoke" else 600 * (1.31e-5 + 4 * 2.22e-6) + 0.05
    reservation = BudgetLedger(ROOT / ".campaign/heldout-response-budget.sqlite", 4).reserve(
        identity, amount
    )
    state = ROOT / ".campaign" / ("heldout-frontier-" + identity[:24] + ".json")
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
