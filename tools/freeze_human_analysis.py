"""Bind the joint human analysis and reserve the complete fixed final job matrix."""

from pathlib import Path
import hashlib
import importlib.util
import json

ROOT = Path(__file__).resolve().parents[1]
PHASES = ("one_step", "direct_horizon", "response_all_groups", "response_heldout_groups")


def final_jobs(bindings):
    if set(bindings) != set(PHASES):
        raise ValueError("All four controller bindings required")
    rows = []
    for phase in PHASES:
        digest = bindings[phase]["source_digest"]
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("Full source digest required")
        protocols = ["known_users", "unseen_users"] if phase in PHASES[:2] else ["known_users"]
        for protocol in protocols:
            for seed in range(20261015, 20261020):
                for embargo in [0, 24, 48] if phase == "one_step" else [24]:
                    if phase == "one_step":
                        identity = (
                            f"human-final-one-step-{digest}-{protocol}-{seed}-embargo-{embargo:02d}"
                        )
                        state = f"human-final-one-step-{digest[:20]}-{protocol}-{seed}-embargo-{embargo:02d}.json"
                        controller = "cloud/evaluate_human_one_step.py"
                        arguments = [
                            "--protocol",
                            protocol,
                            "--seed",
                            str(seed),
                            "--embargo-hours",
                            str(embargo),
                        ]
                    elif phase == "direct_horizon":
                        identity = f"final-horizon-{digest}-{protocol}-{seed}"
                        state = f"final-horizon-{digest[:16]}-{protocol}-{seed}.json"
                        controller = "cloud/evaluate_human_horizons.py"
                        arguments = ["--protocol", protocol, "--seed", str(seed)]
                    else:
                        identity = f"final-{phase}-{digest}-{seed}-test"
                        state = identity + ".json"
                        controller = "cloud/evaluate_human_responses.py"
                        arguments = ["--phase", phase, "--seed", str(seed)]
                    timeout = 1800 if phase == "one_step" else 3600
                    amount = timeout * (8 * 1.31e-5 + 12 * 2.22e-6) + 0.1
                    rows.append(
                        dict(
                            phase=phase,
                            protocol=protocol,
                            seed=seed,
                            embargo_hours=embargo,
                            reservation_id=identity,
                            reservation_usd=amount,
                            state_file=state,
                            controller=controller,
                            arguments=arguments,
                        )
                    )
    return rows


def main():
    import argparse
    import io
    import sqlite3
    import sys
    from datetime import datetime, timezone
    import modal

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.analysis_freeze import REQUIRED_COMPONENTS, verify_analysis_freeze
    from behavioral_equivalence.budget import BudgetLedger

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--publish",
        action="store_true",
        help="Publish immutable private lock and reserve all final jobs",
    )
    args = parser.parse_args()
    v = modal.Volume.from_name("behavioral-equivalence-research")
    modules = {}
    for name, file in [
        ("one_step", "evaluate_human_one_step.py"),
        ("direct_horizon", "evaluate_human_horizons.py"),
        ("responses", "evaluate_human_responses.py"),
    ]:
        spec = importlib.util.spec_from_file_location("freeze_" + name, ROOT / "cloud" / file)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules[name] = module
    bindings = {}
    snapshots = {}
    configs = {}
    uploads = {}
    artifacts = {}
    cache = {}

    def read(path):
        if path not in cache:
            cache[path] = b"".join(v.read_file(path))
        return cache[path]

    def bind_remote(path, sha, role):
        raw = read(path)
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError("Input identity changed: " + path)
        artifacts.setdefault(path, dict(path=path, sha256=sha, role=role))
        return json.loads(raw)

    def bind_local(raw, name, role):
        sha = hashlib.sha256(raw).hexdigest()
        path = "human-final-analysis-freeze-v1/inputs/" + sha + "/" + name
        uploads[path] = raw
        cache[path] = raw
        artifacts[path] = dict(path=path, sha256=sha, role=role)
        return path

    for phase in PHASES:
        saved, digest = (
            modules[phase].snapshot()
            if phase in PHASES[:2]
            else modules["responses"].snapshot(phase)
        )
        configs[phase] = json.loads(saved["config.json"])
        snapshots[phase] = saved
        bindings[phase] = dict(
            source_digest=digest,
            config_sha256=hashlib.sha256(saved["config.json"].encode()).hexdigest(),
        )
        bind_local(
            json.dumps(saved, indent=2).encode(),
            phase + "-source.json",
            {
                "one_step": "one_step_comparators",
                "direct_horizon": "direct_horizon_comparators",
                "response_all_groups": "response_all_groups",
                "response_heldout_groups": "response_heldout_groups",
            }[phase],
        )
    for path, role in [
        ("spec/human-confirmatory-analysis.md", "statistical_analysis"),
        ("spec/human-intervention-analysis.md", "statistical_analysis"),
        ("spec/human-final-analysis-freeze.md", "statistical_analysis"),
        ("spec/user-cluster-transition-uncertainty.md", "transition_uncertainty"),
        ("configs/ablation-plan-v1.json", "ablations"),
        ("tools/freeze_human_analysis.py", "statistical_analysis"),
        ("tools/run_final_human_jobs.py", "statistical_analysis"),
    ]:
        bind_local((ROOT / path).read_bytes(), Path(path).name, role)
    one = configs["one_step"]
    training_sources = {}
    for probe in one["probes"]:
        for arch, sha in probe["fit_report_sha256"].items():
            bind_remote(
                probe["training_root"]
                + "/fits/"
                + f"{probe['protocol']}-{arch}-{probe['seed']}"
                + "/report.json",
                sha,
                "one_step_comparators",
            )
        bind_remote(
            probe["nested_root"] + "/report.json",
            probe["nested_report_sha256"],
            "one_step_comparators",
        )
        if probe["training_root"] not in training_sources:
            source = json.loads(read(probe["training_root"] + "/source.json"))
            training_sources[probe["training_root"]] = next(
                json.loads(text)
                for name, text in source.items()
                if name.startswith("configs/human-training-calendar")
            )
    for training in training_sources.values():
        bind_remote(
            training["case_root"] + "/report.json",
            training["case_report_sha256"],
            "case_definitions",
        )
        bind_remote(
            training["case_root"] + "/" + training["case_audit_relative"],
            training["case_audit_sha256"],
            "case_definitions",
        )
        bind_remote(
            training["context_root"] + "/report.json",
            training["context_report_sha256"],
            "case_definitions",
        )
    horizon = configs["direct_horizon"]
    bind_remote(
        horizon["cohort_root"] + "/report.json", horizon["cohort_report_sha256"], "case_definitions"
    )
    bind_remote(
        horizon["cohort_root"] + "/" + horizon["audit_relative"],
        horizon["audit_report_sha256"],
        "case_definitions",
    )
    for probe in horizon["probes"]:
        for head in probe["heads"].values():
            bind_remote(
                head["root"] + "/report.json", head["report_sha256"], "direct_horizon_comparators"
            )
    for phase in PHASES[2:]:
        for probe in configs[phase]["probes"]:
            bind_remote(
                probe["history_root"] + "/report.json", probe["history_report_sha256"], phase
            )
            bind_remote(
                probe["frontier_root"] + "/report.json", probe["frontier_report_sha256"], phase
            )
            hc = json.loads(json.loads(read(probe["history_root"] + "/source.json"))["config.json"])
            bind_remote(
                hc["random_cohort_root"] + "/report.json",
                hc["random_cohort_report_sha256"],
                "case_definitions",
            )
            bind_remote(
                hc["random_cohort_root"] + "/" + hc["random_audit_relative"],
                hc["random_audit_sha256"],
                "case_definitions",
            )
    uncertainty = json.loads(
        (ROOT / "reports/preflight/human-cluster-uncertainty-completion.json").read_text()
    )
    for record in uncertainty["records"]:
        path = record["root"].removeprefix("/artifacts/") + "/report.json"
        raw = read(path)
        bind_remote(path, hashlib.sha256(raw).hexdigest(), "transition_uncertainty")
    manifest = dict(
        schema_version="human-final-analysis-freeze-v1",
        status="frozen",
        created_at=datetime.now(timezone.utc).isoformat(),
        test_results_inspected=False,
        components=sorted(REQUIRED_COMPONENTS),
        controllers=bindings,
        fits=[
            dict(protocol=p, seed=s)
            for p in ["known_users", "unseen_users"]
            for s in range(20261015, 20261020)
        ],
        artifacts=list(artifacts.values()),
        registration_scope="Internal pre-test analysis lock, not independent review or external preregistration",
        known_pretest_findings=[
            "All10primarystrict outputs retain256states and fail2xcompression",
            "Heldout response seed20261019targeted/control tables are structurally identical in every fold",
            "All K256pooled count-model emission intervals are vacuous",
        ],
        provider_inference_calls=0,
    )
    for phase in PHASES:
        verify_analysis_freeze(
            manifest,
            source_digest=bindings[phase]["source_digest"],
            config_bytes=snapshots[phase]["config.json"].encode(),
            read_artifact=read,
            phase=phase,
        )
    jobs = final_jobs(bindings)
    with sqlite3.connect(ROOT / ".campaign/nested-human-budget.sqlite") as db:
        used = db.execute("select sum(micros) from reservation").fetchone()[0]
        cap = db.execute("select micros from ceiling").fetchone()[0]
    import math

    needed = sum(math.ceil(row["reservation_usd"] * 1e6) for row in jobs)
    if used + needed > cap:
        raise ValueError("Whole final matrix exceeds existing unallocated campaign allowance")
    if not args.publish:
        print(
            json.dumps(
                dict(
                    status="validated_not_published",
                    artifacts=len(artifacts),
                    jobs=len(jobs),
                    additional_reservation_usd=needed / 1e6,
                ),
                indent=2,
            )
        )
        return
    destination = "human-final-analysis-freeze-v1/freeze.json"
    try:
        v.read_file(destination).__next__()
        raise ValueError("Existing joint lock found, refuse to overwrite")
    except FileNotFoundError:
        pass
    raw = json.dumps(manifest, indent=2).encode() + b"\n"
    sha = hashlib.sha256(raw).hexdigest()
    uploads[destination] = raw
    uploads["human-final-analysis-freeze-v1/freeze-" + sha + ".json"] = raw
    with v.batch_upload() as batch:
        for path, content in uploads.items():
            batch.put_file(io.BytesIO(content), "/" + path)
    assert hashlib.sha256(b"".join(v.read_file(destination))).hexdigest() == sha
    ledger = BudgetLedger(ROOT / ".campaign/nested-human-budget.sqlite", 40)
    for row in jobs:
        row["reservation"] = ledger.reserve(row["reservation_id"], row["reservation_usd"])
    (ROOT / "reports/final-analysis-freeze.json").write_bytes(raw)
    (ROOT / "reports/final-human-job-plan.json").write_text(
        json.dumps(dict(freeze_sha256=sha, status="reserved_not_submitted", jobs=jobs), indent=2)
        + "\n"
    )
    print(
        json.dumps(
            dict(
                status="locked_and_reserved",
                freeze_sha256=sha,
                artifacts=len(artifacts),
                jobs=len(jobs),
                total_reserved_usd=jobs[-1]["reservation"]["total_reserved_usd"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
