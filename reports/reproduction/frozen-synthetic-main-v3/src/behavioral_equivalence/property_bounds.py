"""Certificates for an explicit finite property suite, distinct from bisimulation.

Events concern label visits at decision-indexed graph times. Bounds are uniform
over related concrete states but say nothing about unlisted path properties.
"""

from dataclasses import dataclass, asdict
from fractions import Fraction
import numpy as np
from .core import Chain, Partition, check_partition
from .integer_bounds import integer_kernel, exact_radius


@dataclass(frozen=True)
class LabelEvent:
    mode: str
    targets: tuple[str, ...]
    horizon: int
    threshold: int = 1

    def __post_init__(self):
        if self.mode not in {"reachability", "occupancy", "count_at_least"}:
            raise ValueError("Unsupported label event")
        if not self.targets or len(set(self.targets)) != len(self.targets):
            raise ValueError("Distinct nonempty target labels required")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon < 0:
            raise ValueError("Nonnegative integer horizon required")
        if (
            not isinstance(self.threshold, int)
            or isinstance(self.threshold, bool)
            or self.threshold < 1
        ):
            raise ValueError("Positive integer count threshold required")
        if self.mode != "count_at_least" and self.threshold != 1:
            raise ValueError("Only count events accept a threshold")


def _catalog(events: tuple[LabelEvent, ...]) -> dict:
    if not events or len(set(events)) != len(events):
        raise ValueError("Distinct nonempty event suite required")
    groups = {}
    for i, e in enumerate(events):
        mode = "occupancy" if e.mode == "occupancy" else "count"
        groups.setdefault((mode, e.targets), []).append((i, e))
    return groups


def event_enclosures(chain: Chain, events: tuple[LabelEvent, ...], bits: int = 48) -> dict:
    if not isinstance(bits, int) or isinstance(bits, bool) or not 8 <= bits <= 62:
        raise ValueError("Integer precision from 8 to 62 required")
    rows = integer_kernel(chain)
    d = 1 << bits
    result = {}
    for (mode, targets), requested in _catalog(events).items():
        goal = tuple(label in targets for label in chain.labels)
        maximum_h = max(e.horizon for _, e in requested)
        rmax = (
            1 if mode == "occupancy" else min(max(e.threshold for _, e in requested), maximum_h + 1)
        )
        lower = [tuple(d for _ in rows)] + [
            tuple(d if g and r == 1 else 0 for g in goal) for r in range(1, rmax + 1)
        ]
        upper = lower.copy()
        for h in range(maximum_h + 1):
            for index, e in requested:
                if e.horizon == h:
                    if e.threshold > rmax:
                        result[index] = {"lower": [0] * chain.size, "upper": [0] * chain.size}
                    else:
                        result[index] = {
                            "lower": list(lower[e.threshold]),
                            "upper": list(upper[e.threshold]),
                        }
            if h == maximum_h:
                break
            next_lower = [lower[0]]
            next_upper = [upper[0]]
            for remaining in range(1, rmax + 1):
                lo, hi = [], []
                for s, row in enumerate(rows):
                    previous = 1 if mode == "occupancy" else remaining - int(goal[s])
                    numerator_lo = sum(w * lower[previous][j] for j, w in row.terms)
                    numerator_hi = sum(w * upper[previous][j] for j, w in row.terms)
                    lo.append(numerator_lo // row.total)
                    hi.append((numerator_hi + row.total - 1) // row.total)
                next_lower.append(tuple(lo))
                next_upper.append(tuple(hi))
            lower, upper = next_lower, next_upper
    return result


def event_predictions(chain: Chain, events: tuple[LabelEvent, ...]) -> np.ndarray:
    """Floating screening only; exact integer verification decides acceptance."""
    result = np.zeros((len(events), chain.size))
    for (mode, targets), requested in _catalog(events).items():
        goal = np.array([label in targets for label in chain.labels])
        maximum_h = max(e.horizon for _, e in requested)
        rmax = (
            1 if mode == "occupancy" else min(max(e.threshold for _, e in requested), maximum_h + 1)
        )
        values = np.zeros((rmax + 1, chain.size))
        values[0] = 1
        values[1] = goal
        for h in range(maximum_h + 1):
            for i, e in requested:
                if e.horizon == h and e.threshold <= rmax:
                    result[i] = values[e.threshold]
            if h == maximum_h:
                break
            weighted = np.einsum("ij,rj->ri", chain.transitions, values)
            nxt = np.ones_like(values)
            for r in range(1, rmax + 1):
                nxt[r] = (
                    weighted[1]
                    if mode == "occupancy"
                    else np.where(goal, weighted[r - 1], weighted[r])
                )
            values = nxt
    return result


def _fraction(value: Fraction) -> dict:
    return {
        "numerator_hex": format(value.numerator, "x"),
        "denominator_hex": format(value.denominator, "x"),
    }


def certify_properties(
    concrete: Chain,
    abstract: Chain,
    blocks: Partition,
    events: tuple[LabelEvent, ...],
    bits: int = 48,
    origins: tuple[int, ...] | None = None,
) -> dict:
    check_partition(concrete, blocks)
    if abstract.labels != tuple(concrete.labels[b[0]] for b in blocks):
        raise ValueError("Abstract labels differ from the partition")
    origins = tuple(range(concrete.size)) if origins is None else origins
    if (
        not origins
        or len(set(origins)) != len(origins)
        or any(
            not isinstance(s, int) or isinstance(s, bool) or not 0 <= s < concrete.size
            for s in origins
        )
    ):
        raise ValueError("Distinct nonempty concrete prediction origins required")
    p = event_enclosures(concrete, events, bits)
    q = event_enclosures(abstract, events, bits)
    radius, _ = exact_radius(integer_kernel(concrete), integer_kernel(abstract), blocks)
    mapping = {s: b for b, block in enumerate(blocks) for s in block}
    d = 1 << bits
    records = []
    for i, event in enumerate(events):
        differences = [
            max(
                0,
                p[i]["upper"][s] - q[i]["lower"][mapping[s]],
                q[i]["upper"][mapping[s]] - p[i]["lower"][s],
            )
            for s in origins
        ]
        units = max(differences)
        coupled = 1 - (1 - radius) ** event.horizon
        bound = min(Fraction(units, d), coupled)
        records.append(
            {
                "event": asdict(event),
                "direct_max_error_units": units,
                "witness_state": origins[differences.index(units)],
                "error_bound": _fraction(bound),
                "concrete_enclosure": p[i],
                "abstract_enclosure": q[i],
            }
        )
    return {
        "schema_version": "finite-property-certificate-v1",
        "concrete_digest": concrete.digest(),
        "abstract_digest": abstract.digest(),
        "partition": [list(b) for b in blocks],
        "bits": bits,
        "concrete_origins": list(origins),
        "one_step_radius": _fraction(radius),
        "properties": records,
        "kernel_semantics": "binary-rational weights normalized exactly within each row",
        "scope": "only the enumerated label events from the listed deterministic prediction origins and their related abstract states; not bisimulation or a human guarantee",
    }


def verify_properties(concrete: Chain, abstract: Chain, certificate: dict) -> None:
    events = tuple(
        LabelEvent(**dict(r["event"], targets=tuple(r["event"]["targets"])))
        for r in certificate["properties"]
    )
    expected = certify_properties(
        concrete,
        abstract,
        tuple(tuple(b) for b in certificate["partition"]),
        events,
        certificate["bits"],
        tuple(certificate["concrete_origins"]),
    )
    # JSON converts event target tuples to lists; normalize that representation.
    import json

    if json.loads(json.dumps(expected)) != json.loads(json.dumps(certificate)):
        raise ValueError("Finite property certificate differs from recomputation")
