"""Require the complete supplementary fitted-model matrix before test scoring."""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def build_config(root):
    root = Path(root)
    cfg = json.loads((root / "configs/human-supplementary-heads-v1.json").read_text())
    design = json.loads((root / "configs/human-supplementary-calibration-v1.json").read_text())
    probes = []
    for probe in cfg["parent_probes"]:
        protocol, seed = probe["protocol"], probe["seed"]
        entry = dict(probe)
        heads = {}
        for architecture in cfg["architectures"]:
            path = root / "reports/supplementary-heads" / f"{protocol}-{architecture}-{seed}.json"
            raw = path.read_bytes()
            report = json.loads(raw)
            if (
                report["status"] != "complete"
                or report["source_digest"]
                != "5d9fae59fec2dbbfeafa086f78d98e16347ff7c3a020fb22cc43c89b5f5a3e5a"
                or report["protocol"] != protocol
                or report["architecture"] != architecture
                or report["seed"] != seed
                or len(report["records"]) != 6
                or any(r["status"] != "complete" for r in report["records"])
            ):
                raise ValueError("Every fixed trained endpoint and parent fit required")
            if [r["endpoint"] for r in report["records"]] != [e["name"] for e in cfg["endpoints"]]:
                raise ValueError("Frozen endpoint identities and order required")
            heads[architecture] = dict(
                root=f"supplementary-heads-{report['source_digest'][:16]}/{protocol}-{architecture}-{seed}",
                report_sha256=hashlib.sha256(raw).hexdigest(),
                model_sha256=report["output_sha256"]["model.npz"],
                source_digest=report["source_digest"],
            )
        path = root / "reports/supplementary-calibration" / f"{protocol}-{seed}.json"
        raw = path.read_bytes()
        report = json.loads(raw)
        if (
            report["status"] != "complete"
            or report["source_digest"]
            != "68edad5c7113cb6d09663a86f3c465e0293d9625475e7c8f82999e39646dc531"
            or report["protocol"] != protocol
            or report["seed"] != seed
            or report["certificates_replayed"] != 120
        ):
            raise ValueError("Complete fixed supplementary calibration required")
        entry.update(
            heads=heads,
            calibration=dict(
                root=report["root"].removeprefix("/artifacts/"),
                report_sha256=hashlib.sha256(raw).hexdigest(),
                output_sha256=report["output_sha256"],
                source_digest=report["source_digest"],
            ),
        )
        probes.append(entry)
    expected = {(p, s) for p in cfg["protocols"] for s in cfg["seeds"]}
    if len(probes) != 10 or {(p["protocol"], p["seed"]) for p in probes} != expected:
        raise ValueError("Both protocols and all5 seeds required")
    return dict(
        schema_version="supplementary-final-v1",
        status="execution_frozen",
        probes=probes,
        endpoints=cfg["endpoints"],
        case_root=cfg["case_root"],
        case_report_sha256=cfg["case_report_sha256"],
        context_root=cfg["context_root"],
        context_report_sha256=cfg["context_report_sha256"],
        target_root=cfg["target_attachment_root"],
        target_report_canonical_digest=cfg["target_attachment_report_canonical_digest"],
        fitting_cohort_root=cfg["fitting_cohort_root"],
        fitting_cohort_canonical_digest=cfg["fitting_cohort_canonical_digest"],
        state_counts=design["state_counts"],
        embargo_hours=24,
        inference_batch_size=4096,
        bootstrap_replications=2000,
        minimum_users=50,
        minimum_cases=2000,
        probability_floor=1e-6,
        primary_start_count=64,
        primary_epsilon=0.01,
        comparisons="Every model versus full endpoint-specific k256 emission parent, plus fixed refined64 versus matched random and unconstrained feature controls",
        uncertainty="Parent intervals for all fixed counts; uniform-family abstraction envelope for refined64/.01, separate from nominal certificate",
        forecast_archive="Lossless factorization: direct neural endpoint probabilities; per-case state IDs plus shared emission/quotient tables for discrete models",
        scope="Exploratory supplementary six-endpoint study, specified after primary click scoring began. Cannot replace or rescue primary click or intervention hypotheses. Pointwise user-bootstrap intervals are descriptive, not multiplicity-adjusted discoveries.",
    )


def main():
    cfg = build_config(ROOT)
    raw = json.dumps(cfg, indent=2).encode() + b"\n"
    digest = hashlib.sha256(raw).hexdigest()
    path = ROOT / "configs/human-supplementary-final-v1.json"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("Refuse to overwrite an existing supplementary test freeze")
    path.write_bytes(raw)
    snapshot = (
        ROOT / "reports/source-snapshots" / ("supplementary-final-config-" + digest[:16] + ".json")
    )
    snapshot.write_bytes(raw)
    print(
        json.dumps(dict(path=str(path.relative_to(ROOT)), sha256=digest, probes=len(cfg["probes"])))
    )


if __name__ == "__main__":
    main()
