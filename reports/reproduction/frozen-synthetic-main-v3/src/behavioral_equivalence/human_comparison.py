"""Confirmatory decision over every fixed seed and evaluation protocol."""

from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator


class HumanPrimaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    protocol: Literal["known_users", "unseen_users"]
    seed: int
    phase: Literal["confirmatory"]
    parent_latent_states: Literal[256]
    reduced_latent_states: int = Field(ge=1, le=256)
    candidate_minus_reference_nats: float = Field(allow_inf_nan=False)
    ci95: tuple[float, float]
    users: int = Field(ge=2)
    cases: int = Field(ge=2)
    report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_summary(self) -> "HumanPrimaryResult":
        import math

        if not all(math.isfinite(x) for x in self.ci95) or self.ci95[0] > self.ci95[1]:
            raise ValueError("Finite ordered confidence interval required")
        if self.users > self.cases:
            raise ValueError("User count exceeds prediction cases")
        return self


def decide_primary(
    results: tuple[HumanPrimaryResult, ...],
    expected_seeds: tuple[int, ...],
    margin_nats: float = 0.01,
) -> dict:
    """Require all ten fixed fits; no favorable seed or grid substitution."""
    import math

    if (
        not expected_seeds
        or len(set(expected_seeds)) != len(expected_seeds)
        or any(not isinstance(s, int) or isinstance(s, bool) for s in expected_seeds)
        or not math.isfinite(margin_nats)
        or margin_nats <= 0
    ):
        raise ValueError("Distinct integer seeds and positive finite margin required")
    expected = {(p, s) for p in ("known_users", "unseen_users") for s in expected_seeds}
    observed = {(r.protocol, r.seed): r for r in results}
    if len(observed) != len(results):
        raise ValueError("Duplicate protocol and seed result")
    if set(observed) - expected:
        raise ValueError("Results outside fixed confirmatory matrix")
    rows = [
        dict(
            protocol=r.protocol,
            seed=r.seed,
            compression_met=r.reduced_latent_states <= 128,
            noninferiority_met=r.ci95[1] < margin_nats,
            candidate_minus_reference_nats=r.candidate_minus_reference_nats,
            ci95=list(r.ci95),
            report_sha256=r.report_sha256,
        )
        for _, r in sorted(observed.items())
    ]
    complete = set(observed) == expected
    passed = complete and all(r["compression_met"] and r["noninferiority_met"] for r in rows)
    return dict(
        status="complete" if complete else "incomplete",
        scientific_outcome=("supported" if passed else "unsupported")
        if complete
        else "not_evaluated",
        required_fits=len(expected),
        observed_fits=len(observed),
        missing=[dict(protocol=p, seed=s) for p, s in sorted(expected - set(observed))],
        margin_nats=margin_nats,
        maximum_reduced_latent_states=128,
        observed_compression_failures=sum(not r["compression_met"] for r in rows),
        observed_noninferiority_failures=sum(not r["noninferiority_met"] for r in rows),
        rows=rows,
        scope="Intersection-union decision for fixed fitted models and evaluated data distribution. No claim about all possible training seeds, human psychological states, or intervention transfer.",
        verification="Report identities supplied by caller; numerical certificates and provenance require separate verification.",
    )
