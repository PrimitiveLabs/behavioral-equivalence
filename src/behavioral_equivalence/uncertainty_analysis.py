"""Per-world, checkpointable audit of retained uncertainty certificates."""

import gzip
import hashlib
import json
from pathlib import Path
import time


def summarize_world(path: Path, protocol: dict) -> dict:
    began = time.monotonic()
    raw = (path / "report.json").read_bytes()
    report = json.loads(raw)
    if report["status"] != "complete" or report["protocol_digest"] != protocol["protocol_digest"]:
        raise ValueError("World evidence differs from frozen protocol")
    expected = {
        (m, method)
        for m in protocol["samples_per_row"]
        for method in ["certified_hoeffding", "clopper_pearson_float_quantiles"]
    }
    if {(r["samples_per_row"], r["method"]) for r in report["records"]} != expected or len(
        report["records"]
    ) != len(expected):
        raise ValueError("Unpaired method/sample cases")
    rows, coverage, io = [], [], []
    expected_properties = {
        (target, h)
        for target in ["complete", "disengage", "survival"]
        for h in protocol["horizons"]
    }
    for r in report["records"]:
        if (
            {(p["property"], p["horizon"]) for p in r["properties"]} != expected_properties
            or len(r["properties"]) != len(expected_properties)
            or any(p["states"] != report["spec"]["size"] for p in r["properties"])
        ):
            raise ValueError("Missing, duplicated or wrong-size property comparisons")
        start = time.monotonic()
        packed = (path / r["certificate"]).read_bytes()
        if hashlib.sha256(packed).hexdigest() != r["certificate_sha256"]:
            raise ValueError("Certificate artifact changed")
        cert = json.loads(gzip.decompress(packed))["evidence"]
        io.append(
            {
                "certificate": r["certificate"],
                "bytes": len(packed),
                "read_parse_seconds": time.monotonic() - start,
            }
        )
        envelope = cert["uniform_abstraction"]
        active = [
            (b, len(block))
            for b, block in enumerate(envelope["partition"])
            if envelope["labels"][b] == "active"
        ]
        n_active = sum(weight for _, weight in active)
        coverage.append(
            {
                "spec": report["spec"],
                "method": r["method"],
                "samples_per_row": r["samples_per_row"],
                "kernel_included": r["kernel_membership"],
                "all_state_property_enclosures_contained": all(
                    p["enclosure_containment_count"] == p["states"] for p in r["properties"]
                ),
                "proven_noncoverage_count": sum(
                    p["proven_noncoverage_count"] for p in r["properties"]
                ),
                "unresolved_count": sum(p["unresolved_count"] for p in r["properties"]),
            }
        )
        for p in r["properties"]:
            target = "terminal" if p["property"] == "survival" else p["property"]
            bound = cert["robust_values"][target]
            values = bound["values"][str(p["horizon"])]
            denominator = 1 << bound["bits"]
            widths = [
                (values["max_upper"][b] - values["min_lower"][b]) / denominator for b, _ in active
            ]
            width = sum(w * weight for w, (_, weight) in zip(widths, active)) / n_active
            vacuous = (
                sum(weight for w, (_, weight) in zip(widths, active) if w >= 0.999999) / n_active
            )
            rows.append(
                {
                    "method": r["method"],
                    "size": report["spec"]["size"],
                    "family": report["spec"]["family"],
                    "seed": report["spec"]["seed"],
                    "samples_per_row": r["samples_per_row"],
                    "property": p["property"],
                    "horizon": p["horizon"],
                    "width": width,
                    "vacuous_fraction": vacuous,
                }
            )
    return {
        "status": "complete",
        "spec": report["spec"],
        "world_report_sha256": hashlib.sha256(raw).hexdigest(),
        "protocol_digest": protocol["protocol_digest"],
        "rows": rows,
        "coverage_records": coverage,
        "io": io,
        "wall_seconds": time.monotonic() - began,
    }
