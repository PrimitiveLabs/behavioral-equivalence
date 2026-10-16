"""Uniform uncertainty-abstraction evaluation against known model probabilities."""

from fractions import Fraction
from .core import Chain, Partition
from .uncertainty import IntervalChain
from .uncertainty_abstraction import abstract_envelope, verify_envelope
from .integer_bounds import integer_kernel, reachability_enclosures, robust_enclosures


def assess_intervals(
    concrete: Chain,
    uncertainty: IntervalChain,
    blocks: Partition,
    horizons: list[int],
    bits: int = 48,
) -> tuple[dict, dict]:
    if uncertainty.labels != concrete.labels:
        raise ValueError("Concrete and uncertain labels must match")
    p = integer_kernel(concrete)
    member = all(
        Fraction.from_float(float(uncertainty.lower[s, j]))
        <= Fraction(dict(row.terms).get(j, 0), row.total)
        <= Fraction.from_float(float(uncertainty.upper[s, j]))
        for s, row in enumerate(p)
        for j in range(concrete.size)
    )
    abstract, certificate = abstract_envelope(uncertainty, blocks)
    verify_envelope(uncertainty, certificate)
    mapping = {s: b for b, block in enumerate(blocks) for s in block}
    properties = {
        "complete": frozenset({"complete"}),
        "disengage": frozenset({"disengage"}),
        "terminal": frozenset({"complete", "disengage"}),
    }
    records = []
    tables = {}
    denominator = 1 << bits
    for target, labels in properties.items():
        # Relabel only for this property, preserving the actual interval rows.
        model = IntervalChain(
            abstract.lower,
            abstract.upper,
            tuple("target" if label in labels else "other" for label in abstract.labels),
        )
        bounds = robust_enclosures(model, "target", horizons, bits)
        truth = reachability_enclosures(p, concrete.labels, labels, horizons, bits)
        tables[target] = bounds
        for h in horizons:
            lo = bounds["values"][str(h)]["min_lower"]
            hi = bounds["values"][str(h)]["max_upper"]
            exact_lo, exact_hi = truth[h]
            covered = [
                lo[mapping[s]] <= exact_lo[s] and hi[mapping[s]] >= exact_hi[s]
                for s in range(concrete.size)
            ]
            disproved = [
                lo[mapping[s]] > exact_hi[s] or hi[mapping[s]] < exact_lo[s]
                for s in range(concrete.size)
            ]
            records.append(
                {
                    "property": "survival" if target == "terminal" else target,
                    "horizon": h,
                    "states": concrete.size,
                    "enclosure_containment_count": sum(covered),
                    "proven_noncoverage_count": sum(disproved),
                    "unresolved_count": concrete.size - sum(covered) - sum(disproved),
                    "average_width_all_concrete_states": sum(
                        hi[mapping[s]] - lo[mapping[s]] for s in range(concrete.size)
                    )
                    / (denominator * concrete.size),
                    "maximum_width": max(hi[b] - lo[b] for b in range(len(blocks))) / denominator,
                }
            )
    if member and any(r["proven_noncoverage_count"] for r in records):
        raise AssertionError("Uniform model certificate contradicts an in-set concrete model")
    return {
        "kernel_membership": member,
        "concrete_states": concrete.size,
        "abstract_states": len(blocks),
        "properties": records,
        "conditional_certificate_contradictions": 0,
        "represented_zero_lower_edges": int(((abstract.lower == 0) & (abstract.upper > 0)).sum()),
        "statistical_unit": "one sampled transition dataset; state/property/horizon checks are dependent",
        "numeric_status": "directed integer enclosures for binary-rational kernels and bounds",
        "semantics": "uniform hidden-state envelope with per-step rectangular choices; no added nominal epsilon",
    }, {"uniform_abstraction": certificate, "robust_values": tables}
