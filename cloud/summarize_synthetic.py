"""Aggregate the complete source-bound synthetic campaign and integer audit."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-synthetic-summary-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=1800,
    retries=0,
    volumes={"/artifacts": volume},
)
def summarize():
    import csv
    import gzip
    import hashlib
    import itertools
    import json
    import time
    import numpy as np
    from behavioral_equivalence.synthetic_analysis import world_method_rows

    started = time.monotonic()
    volume.reload()
    root = Path("/artifacts/synthetic-main-v3-d9f1fdcc7c35")
    protocol_raw = (root / "protocol.json").read_bytes()
    protocol = json.loads(protocol_raw)
    source_raw = (root / "timeout-recovery-v1/reconciled-manifest.json").read_bytes()
    reconciled = json.loads(source_raw)
    audit_root = root / "integer-campaign-audit-6d9dbce18e2635f0"
    audit_raw = (audit_root / "manifest.json").read_bytes()
    audit = json.loads(audit_raw)
    if (
        reconciled["status"] != "complete"
        or reconciled["worlds"] != 1200
        or audit["status"] != "complete"
        or audit["worlds_audited"] != 1200
        or audit["verified_certificates"] != 26400
        or reconciled["protocol_digest"] != protocol["protocol_digest"]
        or audit["source_protocol_digest"] != protocol["protocol_digest"]
    ):
        raise ValueError("Complete matching frozen campaign and audit required")
    specs = [
        dict(family=f, size=n, perturbation=e, seed=s)
        for f, n, e, s in itertools.product(
            protocol["families"], protocol["sizes"], protocol["perturbations"], protocol["seeds"]
        )
    ]
    expected = {
        hashlib.sha256(json.dumps(s, sort_keys=True).encode()).hexdigest(): s for s in specs
    }
    checked = {r["key"]: r for r in audit["records"]}
    completed = {r["key"]: r for r in reconciled["results"]}
    if len(expected) != 1200 or set(expected) != set(checked) or set(expected) != set(completed):
        raise ValueError("Duplicate, missing or unexpected synthetic worlds")
    saved = {
        "summarize_synthetic.py": Path(__file__).read_text(),
        "synthetic_analysis.py": Path(
            "/research/src/behavioral_equivalence/synthetic_analysis.py"
        ).read_text(),
    }
    digest = hashlib.sha256(
        json.dumps(saved, sort_keys=True).encode() + source_raw + audit_raw
    ).hexdigest()
    out = root / ("summary-" + digest[:16])
    out.mkdir(exist_ok=True)
    (out / "source.json").write_text(json.dumps(saved, indent=2))
    rows, properties, bindings = [], [], []
    prism_checks, maximum_prism_difference = 0, 0.0
    for key in sorted(expected):
        raw = (root / "worlds" / key / "report.json").read_bytes()
        ar = (audit_root / "worlds" / key / "report.json").read_bytes()
        world, aw = json.loads(raw), json.loads(ar)
        if (
            hashlib.sha256(raw).hexdigest() != checked[key]["source_report_sha256"]
            or hashlib.sha256(ar).hexdigest() != checked[key]["audit_report_sha256"]
            or world["source_digest"] != protocol["source_digest"]
            or world["protocol_digest"] != protocol["protocol_digest"]
            or world["spec"] != expected[key]
            or len(world["methods"]) != protocol["expected_method_variants_per_world"]
        ):
            raise ValueError("World or audit no longer matches frozen provenance")
        mr, pr = world_method_rows(world, aw, protocol["horizons"])
        rows.extend(mr)
        properties.extend(pr)
        for verify in [world["original_verification"]] + [
            m["independent_verification"] for m in world["methods"]
        ]:
            if verify["status"] != "passed" or len(verify["checks"]) != 2:
                raise ValueError("Independent PRISM check missing")
            for check in verify["checks"]:
                if (
                    check["horizon"] != 50
                    or check["state"] != 0
                    or check["difference"] > protocol["numerical_tolerance"]
                ):
                    raise ValueError("PRISM check scope or result differs from frozen protocol")
                prism_checks += 1
                maximum_prism_difference = max(maximum_prism_difference, check["difference"])
        bindings.append(
            dict(
                world=key,
                source_report_sha256=checked[key]["source_report_sha256"],
                audit_report_sha256=checked[key]["audit_report_sha256"],
            )
        )

    def table(name, data):
        with gzip.open(out / name, "wt", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader()
            w.writerows(data)

    table("methods.csv.gz", rows)
    table("properties.csv.gz", properties)
    grouped = {}
    for row in rows:
        group = (row["method"], row["parameters"], row["size"], row["perturbation"])
        grouped.setdefault(group, []).append(row)
    cells = []
    for (method, params, size, perturbation), data in sorted(grouped.items()):
        cell = dict(
            method=method,
            parameters=params,
            size=size,
            perturbation=perturbation,
            worlds=len(data),
            scope="descriptive fixed benchmark cells; paired seeds reused across conditions",
        )
        for name in [
            "compression_ratio",
            "states",
            "partition_seconds",
            "property_seconds",
            "prism_seconds",
            "process_cumulative_peak_rss_kib",
            "transition_edges",
        ]:
            values = [r[name] for r in data]
            cell[name] = dict(
                mean=float(np.mean(values)),
                median=float(np.median(values)),
                minimum=float(min(values)),
                maximum=float(max(values)),
            )
        cells.append(cell)
    exceedances = [r for r in rows if r["budget_strictly_satisfied"] is False]
    exact = [r for r in rows if r["method"] == "exact_rational"]
    summary = dict(
        status="complete",
        analysis_digest=digest,
        original_source_digest=protocol["source_digest"],
        original_protocol_digest=protocol["protocol_digest"],
        reconciled_manifest_sha256=hashlib.sha256(source_raw).hexdigest(),
        integer_audit_manifest_sha256=hashlib.sha256(audit_raw).hexdigest(),
        worlds=len(expected),
        method_variants=len(rows),
        verified_integer_certificates=audit["verified_certificates"],
        unconditional_property_origin_maximum_comparisons=len(properties) * 3,
        conditional_property_origin_maximum_comparisons=len(properties),
        floating_bound_violations=sum(
            r["unconditional_violations"] + r["conditional_violation"] for r in properties
        ),
        floating_numerical_tolerance=protocol["numerical_tolerance"],
        prism_point_checks=prism_checks,
        maximum_prism_point_difference=maximum_prism_difference,
        prism_scope="State zero, completion/disengagement, horizon 50, original and all 22 quotients; not every property/origin",
        integer_scope=audit["scope"],
        strict_requested_budget_exceedances=exceedances,
        maximum_exact_budget_excess_for_display=max(
            r["exact_budget_excess_for_display"] for r in exceedances
        )
        if exceedances
        else 0,
        serialized_exact_quotients_with_nonzero_radius=sum(
            not r["epsilon_is_exactly_zero"] for r in exact
        ),
        maximum_serialized_exact_radius_for_display=max(
            r["epsilon_exact_normalized_for_display"] for r in exact
        ),
        cells=cells,
        artifact_bindings=bindings,
        memory_scope="Worker process cumulative high-water mark; never isolated per-method peak memory",
        runtime_scope="Recorded task measurements under static and documented increased concurrency, including preemptions and cached coordinator reconciliation; no controlled isolated hardware-speedup claim",
        recovery_scope="Original terminal coordinator timeout preserved; reconciliation reused the completed canonical source report. No new numerical computation inferred from the cached response.",
        table_scope="Floating screening errors and their recorded floating coupling/conditioning bounds; exact normalized-radius audit reported separately. CSV display values do not replace integer certificates.",
        wall_seconds=time.monotonic() - started,
        root=str(out),
        output_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()
        },
    )
    if (
        len(exceedances) != audit["strict_requested_epsilon_exceedances"]
        or summary["serialized_exact_quotients_with_nonzero_radius"]
        != audit["serialized_exact_quotients_with_nonzero_radius"]
    ):
        raise AssertionError("Audit counts changed during aggregation")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    volume.commit()
    return {
        k: v
        for k, v in summary.items()
        if k not in ("cells", "artifact_bindings", "strict_requested_budget_exceedances")
    }


@app.local_entrypoint()
def main():
    import json

    call = summarize.spawn()
    print(
        json.dumps(dict(function_call_id=call.object_id, function_id=summarize.object_id)),
        flush=True,
    )
    print(json.dumps(call.get()), flush=True)
