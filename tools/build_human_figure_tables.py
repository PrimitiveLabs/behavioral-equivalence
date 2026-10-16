"""Complete-case aggregate inputs for measured human Figures 2 and 6."""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def frontier_rows(report, nested):
    records = {"nested." + r["name"]: r for r in nested["records"]}
    rows = []
    for key, metric in report["metrics"].items():
        if not key.startswith("all/"):
            continue
        name = key.removeprefix("all/")
        record = records.get(name)
        states = (
            record["reduced_latent_states"]
            if record
            else 256
            if name == "parent_k256"
            else int(name.removeprefix("teacher_k"))
            if name.startswith("teacher_k")
            else int(name.removeprefix("pooled_count_k"))
            if name.startswith("pooled_count_k")
            else None
        )
        family = (
            record["kind"]
            if record
            else "parent"
            if name == "parent_k256"
            else "independent_teacher_quantization"
            if name.startswith("teacher_k")
            else "pooled_count_model"
            if name.startswith("pooled_count_k")
            else "sequence_or_markov"
        )
        comparison = (
            report["comparisons"]["all"].get(name + "-vs-parent")
            if name != "parent_k256"
            else dict(candidate_minus_reference_nats=0, ci95=[0, 0])
        )
        rows.append(
            dict(
                protocol=report["protocol"],
                seed=report["seed"],
                method=name,
                family=family,
                latent_states=states,
                primary=name == report["primary"],
                users=metric["users"],
                cases=metric["cases"],
                nll_equal_user_mean=metric["nll_equal_user_mean"],
                brier_equal_user_mean=metric["brier_equal_user_mean"],
                candidate_minus_parent_nats=comparison["candidate_minus_reference_nats"],
                ci95=comparison["ci95"],
            )
        )
    return rows


def annotated_row(values, **identity):
    """Retain statistical RNG identity and reject conflicting cohort metadata."""
    result = dict(values)
    if "seed" in result:
        result["bootstrap_seed"] = result["seed"]
    for key, value in identity.items():
        if key != "seed" and key in result and result[key] != value:
            raise ValueError("Conflicting figure cohort identity: " + key)
        result[key] = value
    return result


def main():
    frozen = json.loads((ROOT / "reports/final-analysis-freeze.json").read_text())
    freeze_digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    required = [
        ("one_step", p, s)
        for p in ["known_users", "unseen_users"]
        for s in range(20261015, 20261020)
    ] + [
        (phase, "known_users", s)
        for phase in ["response_all_groups", "response_heldout_groups"]
        for s in range(20261015, 20261020)
    ]
    frontier, calibration, uncertainty, responses, inputs = [], [], [], [], []
    for phase, protocol, seed in required:
        path = ROOT / "reports/final-human" / f"{phase}-{protocol}-{seed}-24.json"
        if not path.exists():
            raise ValueError(
                f"Complete fixed matrices required before human figures: missing {path.name}"
            )
        raw = path.read_bytes()
        report = json.loads(raw)
        if (
            report["source_digest"] != frozen["controllers"][phase]["source_digest"]
            or report["analysis_freeze_digest"] != freeze_digest
        ):
            raise ValueError("Figure report does not match joint analysis lock")
        inputs.append(
            dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(raw).hexdigest())
        )
        if phase == "one_step":
            nested_path = ROOT / "reports/preflight" / f"human-nested-{protocol}-{seed}.json"
            if (
                hashlib.sha256(nested_path.read_bytes()).hexdigest()
                != report["nested_report_sha256"]
            ):
                raise ValueError("Figure partition provenance changed")
            frontier.extend(frontier_rows(report, json.loads(nested_path.read_text())))
            for name in [
                "markov",
                "context",
                "gru",
                "transformer",
                "parent_k256",
                report["primary"],
            ]:
                metric = report["metrics"]["all/" + name]
                for cell in metric["calibration_bins"]:
                    calibration.append(
                        annotated_row(cell, protocol=protocol, seed=seed, method=name)
                    )
            for states, diagnostic in report["pooled_interval_diagnostics"].items():
                uncertainty.append(
                    annotated_row(
                        diagnostic, protocol=protocol, seed=seed, latent_states=int(states)
                    )
                )
        else:
            contrast = report["comparisons"]["aggregate"][
                report["primary_candidate"] + "-vs-reference"
            ]
            responses.append(
                annotated_row(
                    contrast,
                    phase=phase,
                    seed=seed,
                    users=report["users"],
                    cases=report["cases"],
                    candidate=report["primary_candidate"],
                    reference=report["primary_reference"],
                )
            )
    result = dict(
        status="complete",
        analysis_freeze_digest=freeze_digest,
        inputs=inputs,
        frontier=frontier,
        calibration=calibration,
        uncertainty=uncertainty,
        responses=responses,
        scope="Complete ten-fit primary24hprediction matrix and both five-seed response matrices. Across-seed displays are descriptive; paired confidence intervals remain conditional on original user clusters. No raw human records or private user statistics.",
    )
    out = ROOT / "reports/figures/human-final-tables.json"
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            dict(
                status="complete",
                frontier_rows=len(frontier),
                calibration_cells=len(calibration),
                response_comparisons=len(responses),
                sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
            )
        )
    )


if __name__ == "__main__":
    main()
