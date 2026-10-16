"""Audit and summarize the completed full synthetic uncertainty benchmark."""

from pathlib import Path
import modal

app = modal.App("behavioral-equivalence-uncertainty-summary-v2")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
ROOT = Path(__file__).resolve().parents[1]
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .env({"PYTHONPATH": "/research/src"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


def audit_digest() -> str:
    import hashlib

    return hashlib.sha256(
        Path(__file__).read_bytes()
        + Path("/research/src/behavioral_equivalence/uncertainty_analysis.py").read_bytes()
    ).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(2048, 2048),
    timeout=120,
    retries=0,
    max_containers=4,
    volumes={"/artifacts": volume},
)
def audit_world(campaign: str, key: str) -> dict:
    import json
    from behavioral_equivalence.uncertainty_analysis import summarize_world

    if (
        not campaign.replace("-", "").isalnum()
        or len(key) != 64
        or any(c not in "0123456789abcdef" for c in key)
    ):
        raise ValueError("Valid campaign and SHA256 world key required")
    root = Path("/artifacts") / campaign
    protocol = json.loads((root / "protocol.json").read_text())
    destination = root / ("summary-audit-" + audit_digest()[:16]) / (key + ".json")
    if destination.exists():
        return json.loads(destination.read_text())
    result = summarize_world(root / "worlds" / key, protocol)
    destination.parent.mkdir(exist_ok=True)
    destination.write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.function(
    image=image,
    cpu=(0.5, 0.5),
    memory=(1024, 1024),
    timeout=3600,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def summarize(campaign: str) -> dict:
    import itertools
    import json
    import time
    import numpy as np

    if not campaign.replace("-", "").isalnum():
        raise ValueError("Invalid campaign")
    root = Path("/artifacts") / campaign
    manifest = json.loads((root / "manifest.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    if (
        manifest["status"] != "complete"
        or manifest["failed"]
        or manifest["protocol_digest"] != protocol["protocol_digest"]
    ):
        raise ValueError("Complete, source-bound campaign required")
    began = time.monotonic()
    expected = {
        json.dumps(dict(family=f, size=n, seed=s), sort_keys=True)
        for f, n, s in itertools.product(protocol["families"], protocol["sizes"], protocol["seeds"])
    }
    results = list(
        audit_world.map(
            [campaign] * len(manifest["results"]),
            [r["key"] for r in manifest["results"]],
            return_exceptions=True,
            order_outputs=True,
        )
    )
    if any(isinstance(r, Exception) for r in results):
        raise RuntimeError(
            "One or more per-world audits failed; successful sidecars remain cached: "
            + str([repr(r) for r in results if isinstance(r, Exception)])
        )
    if {json.dumps(r["spec"], sort_keys=True) for r in results} != expected or len(results) != len(
        expected
    ):
        raise ValueError("Incomplete or duplicated world matrix")
    file_hashes = {
        manifest["results"][i]["key"]: r["world_report_sha256"] for i, r in enumerate(results)
    }
    reports = results
    groups = {}
    dataset_records = []
    paired = {}
    for report in results:
        dataset_records.extend(report["coverage_records"])
        for row in report["rows"]:
            key = (
                row["method"],
                row["size"],
                row["samples_per_row"],
                row["property"],
                row["horizon"],
            )
            groups.setdefault(key, []).append(row)
            pair_key = (
                row["family"],
                row["size"],
                row["seed"],
                row["samples_per_row"],
                row["property"],
                row["horizon"],
            )
            paired.setdefault(pair_key, {})[row["method"]] = row["width"]
    aggregates = []
    for key, rows in sorted(groups.items()):
        method, size, m, target, h = key
        aggregates.append(
            {
                "method": method,
                "size": size,
                "samples_per_row": m,
                "property": target,
                "horizon": h,
                "independent_worlds": len(rows),
                "mean_active_state_width": float(np.mean([r["width"] for r in rows])),
                "mean_active_state_vacuous_fraction": float(
                    np.mean([r["vacuous_fraction"] for r in rows])
                ),
            }
        )
    rng = np.random.default_rng(20261015)
    width_differences = []
    for size, m, target, h in itertools.product(
        protocol["sizes"],
        protocol["samples_per_row"],
        ["complete", "disengage", "survival"],
        protocol["horizons"],
    ):
        strata = []
        for family in protocol["families"]:
            values = np.array(
                [
                    paired[(family, size, seed, m, target, h)]["clopper_pearson_float_quantiles"]
                    - paired[(family, size, seed, m, target, h)]["certified_hoeffding"]
                    for seed in protocol["seeds"]
                ]
            )
            strata.append(values)
        draws = sum(v[rng.integers(0, len(v), (2000, len(v)))].mean(axis=1) for v in strata) / len(
            strata
        )
        width_differences.append(
            {
                "size": size,
                "samples_per_row": m,
                "property": target,
                "horizon": h,
                "cp_minus_hoeffding_mean_width": float(np.mean(np.concatenate(strata))),
                "paired_stratified_world_bootstrap_95_ci": np.quantile(
                    draws, [0.025, 0.975]
                ).tolist(),
            }
        )
    coverage = []
    for method, size, m in itertools.product(
        ["certified_hoeffding", "clopper_pearson_float_quantiles"],
        protocol["sizes"],
        protocol["samples_per_row"],
    ):
        rows = [
            r
            for r in dataset_records
            if r["method"] == method and r["spec"]["size"] == size and r["samples_per_row"] == m
        ]
        coverage.append(
            {
                "method": method,
                "size": size,
                "samples_per_row": m,
                "sampled_datasets": len(rows),
                "kernel_inclusion_count": sum(r["kernel_included"] for r in rows),
                "all_probability_enclosures_contained_count": sum(
                    r["all_state_property_enclosures_contained"] for r in rows
                ),
                "proven_noncoverage_count": sum(r["proven_noncoverage_count"] for r in rows),
                "unresolved_comparisons": sum(r["unresolved_count"] for r in rows),
            }
        )
    summary = {
        "status": "complete",
        "campaign_id": campaign,
        "protocol_digest": protocol["protocol_digest"],
        "worlds": len(reports),
        "paired_transition_datasets": len(dataset_records) // 2,
        "method_dataset_evaluations": len(dataset_records),
        "verified_certificate_artifacts": len(dataset_records),
        "coverage": coverage,
        "widths": aggregates,
        "paired_width_comparisons": width_differences,
        "world_report_sha256": file_hashes,
        "wall_seconds": time.monotonic() - began,
        "limitations": [
            "Finite Monte Carlo coverage counts are descriptive, not a proof of nominal coverage.",
            "The exact Hoeffding argument requires fixed-design iid row samples; no human coverage claim follows.",
            "Clopper-Pearson beta quantiles lack outward-rounding certification.",
            "PRISM 4.11 does not independently check the represented zero-lower-bound rows in this suite.",
            "Rectangular envelopes can be more conservative than fixed-kernel uncertainty.",
            "Width confidence intervals condition on fixed generator families and use paired world-level stratified resampling.",
        ],
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    (root / "summary-source.py").write_bytes(Path(__file__).read_bytes())
    (root / "summary-helper-source.py").write_bytes(
        Path("/research/src/behavioral_equivalence/uncertainty_analysis.py").read_bytes()
    )
    volume.commit()
    return {
        "status": summary["status"],
        "worlds": len(reports),
        "paired_transition_datasets": len(dataset_records) // 2,
        "method_dataset_evaluations": len(dataset_records),
        "report": str(root / "summary.json"),
        "wall_seconds": summary["wall_seconds"],
    }


@app.local_entrypoint()
def main(campaign: str):
    import json

    call = summarize.spawn(campaign)
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": summarize.object_id}),
        flush=True,
    )
    print(json.dumps(call.get(), indent=2), flush=True)
