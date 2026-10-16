"""PRISM 4.11 subprocess adapter with explicit initial state and retained logs."""

from pathlib import Path
import re
import subprocess
from .core import Chain
from .uncertainty import IntervalChain


def write_model(model: Chain | IntervalChain, destination: Path, initial_state: int = 0) -> None:
    n = len(model.labels)
    if not 0 <= initial_state < n:
        raise ValueError("Initial state out of bounds")
    lines = [
        "dtmc",
        "",
        "module behavioral",
        f"  s : [0..{n - 1}] init {initial_state};",
    ]
    for state in range(n):
        terms = []
        for successor in range(n):
            if isinstance(model, Chain):
                probability = model.transitions[state, successor]
                if probability > 0:
                    terms.append(f"{probability:.17g}:(s'={successor})")
            elif model.upper[state, successor] > 0:
                lo, hi = model.lower[state, successor], model.upper[state, successor]
                terms.append(f"[{lo:.17g},{hi:.17g}]:(s'={successor})")
        lines.append(f"  [] s={state} -> " + " + ".join(terms) + ";")
    lines.extend(["endmodule", ""])
    for label in sorted(set(model.labels)):
        # Labels are input data, so restrict interpolation into PRISM syntax.
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", label):
            raise ValueError("PRISM labels must be simple identifiers")
        states = " | ".join(f"s={s}" for s, value in enumerate(model.labels) if value == label)
        lines.append(f'label "{label}" = {states};')
    destination.write_text("\n".join(lines) + "\n")


def check(
    model_path: Path,
    target: str,
    horizon: int,
    log_path: Path,
    executable: str = "/opt/prism/bin/prism",
    extremum: str | None = None,
) -> float:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", target) or horizon < 0:
        raise ValueError("Invalid property")
    if extremum not in (None, "min", "max"):
        raise ValueError("Extremum must be min or max")
    operator = "P" + (extremum or "")
    prop = f'{operator}=? [ F<={horizon} "{target}" ]'
    result = subprocess.run(
        [executable, str(model_path), "-pf", prop, "-explicit"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    log_path.write_text(result.stdout + "\n" + result.stderr)
    if result.returncode:
        raise RuntimeError(f"PRISM failed with code {result.returncode}; see {log_path}")
    match = re.search(r"^Result:\s+([0-9.eE+\-]+)", result.stdout, re.MULTILINE)
    if match is None:
        raise RuntimeError(f"PRISM returned no scalar result; see {log_path}")
    return float(match.group(1))
