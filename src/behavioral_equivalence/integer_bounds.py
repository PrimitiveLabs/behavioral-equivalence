"""Exact integer audit of finite-horizon probabilities and abstraction error.

Float inputs define exact binary-rational weights. Chain rows are normalized
with integer arithmetic. Probability enclosures use directed integer division,
so their validity does not depend on floating-point model-checking tolerance.
"""

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from .core import Chain, Partition, check_partition
from .uncertainty import IntervalChain


@dataclass(frozen=True)
class IntegerRow:
    terms: tuple[tuple[int, int], ...]
    total: int


def integer_kernel(chain: Chain) -> tuple[IntegerRow, ...]:
    rows = []
    for row in chain.transitions:
        ratios = [(j, *float(p).as_integer_ratio()) for j, p in enumerate(row) if p]
        common = max(d for _, _, d in ratios)
        terms = tuple((j, n * (common // d)) for j, n, d in ratios)
        rows.append(IntegerRow(terms, sum(w for _, w in terms)))
    return tuple(rows)


def exact_radius(
    concrete: tuple[IntegerRow, ...], abstract: tuple[IntegerRow, ...], blocks: Partition
) -> tuple[Fraction, int]:
    mapping = {s: i for i, b in enumerate(blocks) for s in b}
    q = [dict(row.terms) for row in abstract]
    maximum = Fraction()
    witness = 0
    for s, row in enumerate(concrete):
        b = mapping[s]
        aggregated = {}
        for u, w in row.terms:
            aggregated[mapping[u]] = aggregated.get(mapping[u], 0) + w
        numerator = sum(
            abs(aggregated.get(j, 0) * abstract[b].total - q[b].get(j, 0) * row.total)
            for j in aggregated.keys() | q[b].keys()
        )
        value = Fraction(numerator, 2 * row.total * abstract[b].total)
        if value > maximum:
            maximum = value
            witness = s
    return maximum, witness


def _round_dot(row: IntegerRow, values: tuple[int, ...], up: bool) -> int:
    numerator = sum(w * values[j] for j, w in row.terms)
    return (numerator + row.total - 1) // row.total if up else numerator // row.total


def reachability_enclosures(
    rows: tuple[IntegerRow, ...],
    labels: tuple[str, ...],
    targets: frozenset[str],
    horizons: list[int],
    bits: int = 48,
) -> dict:
    if not isinstance(bits, int) or not 1 <= bits <= 62 or not horizons:
        raise ValueError("Positive fixed-point precision and horizons required")
    if any(not isinstance(h, int) or isinstance(h, bool) or h < 0 for h in horizons):
        raise ValueError("Nonnegative integer horizons required")
    denominator = 1 << bits
    goal = tuple(label in targets for label in labels)
    lower = tuple(denominator if flag else 0 for flag in goal)
    upper = lower
    result = {0: (lower, upper)} if 0 in horizons else {}
    for h in range(1, max(horizons) + 1):
        lower = tuple(
            denominator if goal[s] else _round_dot(row, lower, False) for s, row in enumerate(rows)
        )
        upper = tuple(
            denominator if goal[s] else _round_dot(row, upper, True) for s, row in enumerate(rows)
        )
        if h in horizons:
            result[h] = (lower, upper)
    return result


def _encoded(value: Fraction) -> dict:
    return {
        "numerator_hex": format(value.numerator, "x"),
        "denominator_hex": format(value.denominator, "x"),
    }


def _decoded(value: dict) -> Fraction:
    return Fraction(int(value["numerator_hex"], 16), int(value["denominator_hex"], 16))


def certify_abstraction(
    concrete: Chain,
    abstract: Chain,
    blocks: Partition,
    horizons: list[int],
    bits: int = 48,
    properties: dict[str, tuple[str, ...]] | None = None,
) -> dict:
    check_partition(concrete, blocks)
    if abstract.labels != tuple(concrete.labels[b[0]] for b in blocks):
        raise ValueError("Abstract labels do not match relation")
    p = integer_kernel(concrete)
    q = integer_kernel(abstract)
    epsilon, witness = exact_radius(p, q, blocks)
    catalog = (
        properties
        if properties is not None
        else {
            "complete": ("complete",),
            "disengage": ("disengage",),
            "terminal": ("complete", "disengage"),
        }
    )
    tables = {}
    for name, targets in catalog.items():
        selected = sorted(set(horizons) | {max(0, h - 1) for h in horizons})
        table = reachability_enclosures(q, abstract.labels, frozenset(targets), selected, bits)
        tables[name] = {
            str(h): {"lower": list(lo), "upper": list(hi)} for h, (lo, hi) in table.items()
        }
    result = {
        "schema_version": "integer-abstraction-v1",
        "concrete_digest": concrete.digest(),
        "abstract_digest": abstract.digest(),
        "partition": [list(b) for b in blocks],
        "bits": bits,
        "horizons": horizons,
        "properties": {k: list(v) for k, v in catalog.items()},
        "epsilon": _encoded(epsilon),
        "epsilon_witness": witness,
        "deltas": {str(h): _encoded(1 - (1 - epsilon) ** h) for h in horizons},
        "abstract_enclosures": tables,
        "kernel_semantics": "exact binary-rational weights normalized within each row",
        "initial_scope": "all related deterministic initial state pairs; arbitrary initial laws require coupling",
        "proof_revision": "integer-bounds-v1",
    }
    if "complete" in tables and "active" in concrete.labels:
        condition_rows = []
        for row in q:
            terms = tuple((j, w) for j, w in row.terms if abstract.labels[j] == "active")
            condition_rows.append(IntegerRow(terms, sum(w for _, w in terms)))
        supported = []
        for s, row in enumerate(p):
            mass = sum(w for j, w in row.terms if concrete.labels[j] == "active")
            supported.append(concrete.labels[s] == "active" and mass > 0)
        result["conditioning"] = {
            "concrete_defined": supported,
            "abstract_masses": [
                _encoded(Fraction(r.total, q[s].total)) for s, r in enumerate(condition_rows)
            ],
        }
        conditional = {}
        for h in horizons:
            if h < 1:
                continue
            previous = tables["complete"][str(h - 1)]
            conditional[str(h)] = {
                "lower": [
                    _round_dot(r, tuple(previous["lower"]), False) if r.total else None
                    for r in condition_rows
                ],
                "upper": [
                    _round_dot(r, tuple(previous["upper"]), True) if r.total else None
                    for r in condition_rows
                ],
            }
        result["abstract_enclosures"]["complete_given_active_next"] = conditional
    return result


def state_interval(
    certificate: dict, state: int, target: str, horizon: int
) -> tuple[Fraction, Fraction]:
    """Materialize a bound after verifying the certificate against actual inputs."""
    block = next(i for i, b in enumerate(certificate["partition"]) if state in b)
    denominator = 1 << certificate["bits"]
    delta = _decoded(certificate["deltas"][str(horizon)])
    if target == "complete_given_active_next":
        if not certificate["conditioning"]["concrete_defined"][state]:
            raise ValueError("Concrete conditioning event is undefined for this origin")
        d = _decoded(certificate["conditioning"]["abstract_masses"][block])
        epsilon = _decoded(certificate["epsilon"])
        if d == 0:
            return Fraction(), Fraction(1)
        delta = min(Fraction(1), (delta + epsilon) / (d - epsilon)) if d > epsilon else Fraction(1)
    if target == "survival":
        values = certificate["abstract_enclosures"]["terminal"][str(horizon)]
        lo, hi = denominator - values["upper"][block], denominator - values["lower"][block]
    else:
        values = certificate["abstract_enclosures"][target][str(horizon)]
        lo, hi = values["lower"][block], values["upper"][block]
    return max(Fraction(), Fraction(lo, denominator) - delta), min(
        Fraction(1), Fraction(hi, denominator) + delta
    )


def verify_abstraction(concrete: Chain, abstract: Chain, certificate: dict) -> None:
    expected = certify_abstraction(
        concrete,
        abstract,
        tuple(tuple(b) for b in certificate["partition"]),
        certificate["horizons"],
        certificate["bits"],
        {k: tuple(v) for k, v in certificate["properties"].items()},
    )
    if expected != certificate:
        raise ValueError("Integer certificate differs from recomputation")


@dataclass(frozen=True)
class IntegerIntervalRow:
    lower: tuple[int, ...]
    upper: tuple[int, ...]
    total: int


def interval_kernel(model: IntervalChain) -> tuple[IntegerIntervalRow, ...]:
    rows = []
    for lower, upper in zip(model.lower, model.upper):
        ratios = [float(x).as_integer_ratio() for x in [*lower, *upper]]
        common = max(d for _, d in ratios)
        weights = tuple(n * (common // d) for n, d in ratios)
        n = len(lower)
        lo, hi = weights[:n], weights[n:]
        if sum(lo) > common or sum(hi) < common:
            raise ValueError("Interval row is exactly infeasible despite floating tolerance")
        rows.append(IntegerIntervalRow(lo, hi, common))
    return tuple(rows)


def _extreme(row: IntegerIntervalRow, values: tuple[int, ...], order: list[int], up: bool) -> int:
    numerator = sum(w * v for w, v in zip(row.lower, values))
    remaining = row.total - sum(row.lower)
    for j in order:
        added = min(remaining, row.upper[j] - row.lower[j])
        remaining -= added
        numerator += added * values[j]
        if remaining == 0:
            break
    return (numerator + row.total - 1) // row.total if up else numerator // row.total


def robust_enclosures(
    model: IntervalChain, target: str, horizons: list[int], bits: int = 48
) -> dict:
    rows = interval_kernel(model)
    if not isinstance(bits, int) or not 1 <= bits <= 62 or not horizons:
        raise ValueError("Positive precision and horizons required")
    if any(not isinstance(h, int) or isinstance(h, bool) or h < 0 for h in horizons):
        raise ValueError("Nonnegative integer horizons required")
    denominator = 1 << bits
    goal = tuple(label == target for label in model.labels)
    initial = tuple(denominator if g else 0 for g in goal)
    vectors = [initial] * 4
    result = {}
    if 0 in horizons:
        result["0"] = {
            k: list(initial) for k in ["min_lower", "min_upper", "max_lower", "max_upper"]
        }
    for h in range(1, max(horizons) + 1):
        next_vectors = []
        for column, values in enumerate(vectors):
            maximize = column >= 2
            round_up = column % 2 == 1
            order = sorted(range(len(values)), key=lambda j: values[j], reverse=maximize)
            next_vectors.append(
                tuple(
                    denominator if goal[s] else _extreme(row, values, order, round_up)
                    for s, row in enumerate(rows)
                )
            )
        vectors = next_vectors
        if h in horizons:
            result[str(h)] = dict(
                zip(
                    ["min_lower", "min_upper", "max_lower", "max_upper"], [list(v) for v in vectors]
                )
            )
    payload = {"lower": model.lower.tolist(), "upper": model.upper.tolist(), "labels": model.labels}
    return {
        "schema_version": "integer-robust-v1",
        "model_digest": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "target": target,
        "horizons": horizons,
        "bits": bits,
        "values": result,
        "semantics": "row-rectangular per-step adversary; float endpoints are exact binary rationals",
        "statistical_scope": "conditional on kernel membership in the specified feasible uncertainty set",
    }
