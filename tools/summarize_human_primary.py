"""Apply the frozen primary decisions to every required final human report."""

from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def response_decision(rows, expected_seeds):
    observed = {r["seed"]: r for r in rows}
    if len(observed) != len(rows) or set(observed) - set(expected_seeds):
        raise ValueError("Distinct fixed response seeds required")
    complete = set(observed) == set(expected_seeds)
    return dict(
        status="complete" if complete else "incomplete",
        scientific_outcome=("supported" if all(r["ci95"][1] < 0 for r in rows) else "unsupported")
        if complete
        else "not_evaluated",
        required_fits=len(expected_seeds),
        observed_fits=len(rows),
        missing_seeds=sorted(set(expected_seeds) - set(observed)),
        rows=rows,
        scope="Intersection-union strict improvement over all five fixed seeds, paired original-user bootstrap. Secondary contrasts do not replace this decision; no arbitrary-interface or zero-shot claim.",
    )


def main():
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.human_comparison import HumanPrimaryResult, decide_primary

    frozen = json.loads((ROOT / "reports/final-analysis-freeze.json").read_text())
    freeze_digest = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    seeds = tuple(range(20261015, 20261020))
    primary = []
    response = {phase: [] for phase in ["response_all_groups", "response_heldout_groups"]}
    for path in (ROOT / "reports/final-human").glob("*.json"):
        report = json.loads(path.read_text())
        if report["analysis_freeze_digest"] != freeze_digest:
            raise ValueError("Final report belongs to another analysis lock")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if report.get("feedback_embargo_hours") == 24 and "primary" in report:
            if report["source_digest"] != frozen["controllers"]["one_step"]["source_digest"]:
                raise ValueError("One-step source differs from joint lock")
            n = json.loads(
                (
                    ROOT
                    / "reports/preflight"
                    / f"human-nested-{report['protocol']}-{report['seed']}.json"
                ).read_text()
            )
            if (
                hashlib.sha256(
                    (
                        ROOT
                        / "reports/preflight"
                        / f"human-nested-{report['protocol']}-{report['seed']}.json"
                    ).read_bytes()
                ).hexdigest()
                != report["nested_report_sha256"]
            ):
                raise ValueError("Primary partition provenance changed")
            row = next(r for r in n["records"] if "nested." + r["name"] == report["primary"])
            contrast = report["comparisons"]["all"][report["primary"] + "-vs-parent"]
            support = report["metrics"]["all/parent_k256"]
            primary.append(
                HumanPrimaryResult(
                    protocol=report["protocol"],
                    seed=report["seed"],
                    phase="confirmatory",
                    parent_latent_states=256,
                    reduced_latent_states=row["reduced_latent_states"],
                    candidate_minus_reference_nats=contrast["candidate_minus_reference_nats"],
                    ci95=tuple(contrast["ci95"]),
                    users=support["users"],
                    cases=support["cases"],
                    report_sha256=sha,
                )
            )
        phase = report.get("phase")
        if phase in response:
            if report["source_digest"] != frozen["controllers"][phase]["source_digest"]:
                raise ValueError("Response source differs from joint lock")
            contrast = report["comparisons"]["aggregate"][
                report["primary_candidate"] + "-vs-reference"
            ]
            response[phase].append(
                dict(
                    seed=report["seed"],
                    ci95=contrast["ci95"],
                    candidate_minus_reference_nats=contrast["candidate_minus_reference_nats"],
                    users=report["users"],
                    cases=report["cases"],
                    report_sha256=sha,
                )
            )
    result = dict(
        analysis_freeze_digest=freeze_digest,
        one_step=decide_primary(tuple(primary), seeds),
        responses={phase: response_decision(rows, seeds) for phase, rows in response.items()},
    )
    (ROOT / "reports/final-human-primary-decisions.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
