"""Small exact interval-row vertex oracle and equivalent PRISM MDP export."""

from fractions import Fraction
from itertools import product
from pathlib import Path
import re
import numpy as np


def row_vertices(lower, upper):
    lo, hi = map(lambda a: np.asarray(a, dtype=float), (lower, upper))
    if (
        lo.ndim != 1
        or hi.shape != lo.shape
        or not 1 <= len(lo) <= 8
        or any(not np.isfinite(a).all() for a in [lo, hi])
        or np.any(lo < 0)
        or np.any(hi > 1)
        or np.any(lo > hi)
    ):
        raise ValueError("Feasible interval row of at most8 coordinates required")
    low = tuple(Fraction.from_float(float(v)) for v in lo)
    high = tuple(Fraction.from_float(float(v)) for v in hi)
    if sum(low) > 1 or sum(high) < 1:
        raise ValueError("Interval row simplex is empty")
    result = set()
    for free in range(len(low)):
        fixed = [j for j in range(len(low)) if j != free]
        for choices in product([0, 1], repeat=len(fixed)):
            row = list(low)
            for j, choice in zip(fixed, choices):
                row[j] = high[j] if choice else low[j]
            row[free] = 1 - sum(row[j] for j in fixed)
            if low[free] <= row[free] <= high[free]:
                result.add(tuple(row))
    if not result:
        raise AssertionError("Nonempty bounded row simplex must have a vertex")
    return tuple(sorted(result))


def vertex_reachability(model, target, horizon, maximize):
    if type(horizon) is not int or horizon < 0:
        raise ValueError("Nonnegative integer horizon required")
    rows = [row_vertices(lo, hi) for lo, hi in zip(model.lower, model.upper)]
    goal = [label == target for label in model.labels]
    values = tuple(Fraction(int(g)) for g in goal)
    for _ in range(horizon):
        values = tuple(
            Fraction(1)
            if goal[s]
            else (max if maximize else min)(
                sum(p * v for p, v in zip(row, values)) for row in candidates
            )
            for s, candidates in enumerate(rows)
        )
    return values


def write_vertex_mdp(model, destination, initial_state):
    if type(initial_state) is not int or not 0 <= initial_state < len(model.labels):
        raise ValueError("Valid deterministic initial state required")
    if any(not re.fullmatch("[A-Za-z_][A-Za-z0-9_]*", label) for label in model.labels):
        raise ValueError("Simple PRISM label identifiers required")
    lines = [
        "mdp",
        "",
        "module interval_vertices",
        f"s : [0..{len(model.labels) - 1}] init {initial_state};",
    ]
    vertex_counts = []
    for s, (lo, hi) in enumerate(zip(model.lower, model.upper)):
        vertices = row_vertices(lo, hi)
        vertex_counts.append(len(vertices))
        for i, row in enumerate(vertices):
            terms = [f"{float(p):.17g}:(s'={j})" for j, p in enumerate(row) if p]
            lines.append(f"[row{s}action{i}] s={s} -> " + " + ".join(terms) + ";")
    lines += ["endmodule", ""]
    for label in sorted(set(model.labels)):
        lines.append(
            f'label "{label}" = '
            + " | ".join(f"s={s}" for s, value in enumerate(model.labels) if value == label)
            + ";"
        )
    Path(destination).write_text("\n".join(lines) + "\n")
    return vertex_counts
