"""Fresh small scientific rerun across all generator and abstraction families."""

from pathlib import Path
import hashlib
import json
import numpy as np
from behavioral_equivalence.synthetic import FAMILIES, generate
from behavioral_equivalence.benchmark import evaluate_world
from behavioral_equivalence.core import exact_partition, quotient, reachability
from behavioral_equivalence.integer_bounds import certify_abstraction, verify_abstraction
from behavioral_equivalence.prism import write_model, check


def run(destination, prism_executable):
    out = Path(destination)
    out.mkdir(parents=True, exist_ok=False)
    protocol = {
        "horizons": [5, 10, 25, 50],
        "epsilon_grid": [0.01, 0.03],
        "state_divisors": [2, 4],
        "predictive_feature_horizons": [1, 5],
        "numerical_tolerance": 1e-10,
    }
    worlds = []
    external = []
    certificate_count = 0
    for family in FAMILIES:
        for perturbation in [0, 0.01]:
            spec = dict(family=family, size=32, seed=20261024, perturbation=perturbation)
            result = evaluate_world(spec, protocol)
            if result["bound_violations"] or len({r["method"] for r in result["methods"]}) != 6:
                raise ValueError("Fresh six-method preservation comparison failed")
            worlds.append(result)
            world = generate(**spec)
            blocks = exact_partition([world.chain])
            abstract, _ = quotient(world.chain, blocks)
            cert = certify_abstraction(world.chain, abstract, blocks, protocol["horizons"])
            verify_abstraction(world.chain, abstract, cert)
            name = family + ("-perturbed" if perturbation else "-exact")
            (out / (name + "-certificate.json")).write_text(json.dumps(cert, indent=2))
            certificate_count += 1
            path = out / (name + ".prism")
            write_model(world.chain, path)
            for target in ["complete", "disengage"]:
                h = 10
                predicted = float(reachability(world.chain, target, h)[0])
                observed = check(
                    path,
                    target,
                    h,
                    out / (name + "-" + target + ".log"),
                    executable=prism_executable,
                )
                delta = abs(predicted - observed)
                if delta > 1e-10:
                    raise ValueError("Fresh independent PRISM prediction differs")
                external.append(
                    dict(
                        family=family,
                        perturbation=perturbation,
                        target=target,
                        horizon=h,
                        python=predicted,
                        prism=observed,
                        difference=delta,
                    )
                )
    if not np.isfinite([r["difference"] for r in external]).all():
        raise ValueError("Finite solver output required")
    report = dict(
        status="complete",
        worlds=len(worlds),
        mechanism_families=list(FAMILIES),
        method_families=6,
        certificate_replays=certificate_count,
        external_queries=len(external),
        maximum_external_difference=max(r["difference"] for r in external),
        external=external,
        world_results=worlds,
        scope="Fresh 32-state scientific subset, all five mechanisms, exact/perturbed worlds, all six abstraction method families and all four horizons; 20 independent PRISM point queries. Not a repeat of the full 1200-world matrix or human training.",
    )
    report["artifacts"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()
    }
    (out / "report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prism", default="/opt/prism/bin/prism")
    args = parser.parse_args()
    result = run(args.output, args.prism)
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ["external", "world_results", "artifacts"]}
        )
    )
