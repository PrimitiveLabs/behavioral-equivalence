"""Fail-closed input contract for final human inference."""

import hashlib
import json


REQUIRED_COMPONENTS = frozenset(
    {
        "one_step_comparators",
        "direct_horizon_comparators",
        "response_all_groups",
        "response_heldout_groups",
        "transition_uncertainty",
        "statistical_analysis",
        "case_definitions",
        "ablations",
    }
)


def verify_analysis_freeze(
    manifest, *, source_digest, config_bytes, read_artifact, phase="one_step"
):
    """Verify an externally stored freeze before reading final prediction cases.

    read_artifact returns bytes for a declared private relative artifact path.
    Source snapshots bind executable code; report digests transitively bind
    model weights and case arrays, which inference checks before using them.
    This is an integrity gate, not scientific review or preregistration proof.
    """
    phases = {"one_step", "direct_horizon", "response_all_groups", "response_heldout_groups"}
    controllers = manifest.get("controllers", {})
    if set(controllers) != phases or phase not in phases:
        raise ValueError("All four final controller identities required")
    for binding in controllers.values():
        if any(
            not isinstance(binding.get(key), str)
            or len(binding[key]) != 64
            or any(c not in "0123456789abcdef" for c in binding[key])
            for key in ["source_digest", "config_sha256"]
        ):
            raise ValueError("Controller source and configuration hashes required")
    binding = controllers[phase]
    if (
        manifest.get("schema_version") != "human-final-analysis-freeze-v1"
        or manifest.get("status") != "frozen"
        or manifest.get("test_results_inspected") is not False
        or manifest.get("components") != sorted(REQUIRED_COMPONENTS)
        or binding["source_digest"] != source_digest
        or binding["config_sha256"] != hashlib.sha256(config_bytes).hexdigest()
    ):
        raise ValueError("Complete matching final analysis freeze required")
    expected = {(p, s) for p in ["known_users", "unseen_users"] for s in range(20261015, 20261020)}
    fits = manifest.get("fits", [])
    if len(fits) != 10 or {(f["protocol"], f["seed"]) for f in fits} != expected:
        raise ValueError("All ten protocol and seed combinations required")
    artifacts = manifest.get("artifacts", [])
    if not artifacts or len({a["path"] for a in artifacts}) != len(artifacts):
        raise ValueError("Unique frozen input artifacts required")
    roles = set()
    for artifact in artifacts:
        path = artifact["path"]
        if not isinstance(path, str) or path.startswith("/") or ".." in path.split("/"):
            raise ValueError("Private artifact paths must be relative without traversal")
        if hashlib.sha256(read_artifact(path)).hexdigest() != artifact["sha256"]:
            raise ValueError("Frozen input artifact changed")
        roles.add(artifact["role"])
    if not REQUIRED_COMPONENTS <= roles:
        raise ValueError("Every final analysis component needs bound evidence")
    config = json.loads(config_bytes)
    if config.get("split") != "test" or config.get("noninferiority_margin_nats") != 0.01:
        raise ValueError("Final test and fixed noninferiority margin required")
    probes = config.get("probes", [])
    phase_expected = (
        expected
        if phase in {"one_step", "direct_horizon"}
        else {pair for pair in expected if pair[0] == "known_users"}
    )
    if (
        len(probes) != len(phase_expected)
        or {(p["protocol"], p["seed"]) for p in probes} != phase_expected
    ):
        raise ValueError("Final inference config must retain the full fit matrix")
    return hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
