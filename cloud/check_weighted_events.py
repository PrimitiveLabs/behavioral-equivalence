"""Pinned Lean proof checking on Modal, with no human data or model inference."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-weighted-events-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "zstd", "ca-certificates")
    .run_commands(
        "curl -fL https://github.com/leanprover/lean4/releases/download/v4.19.0/lean-4.19.0-linux.tar.zst -o /tmp/lean.tar.zst",
        "tar --zstd -xf /tmp/lean.tar.zst -C /opt",
    )
    .env({"PATH": "/opt/lean-4.19.0-linux/bin:/usr/local/bin:/usr/bin:/bin"})
    .run_commands(
        "git clone --depth 1 --branch v4.19.0 https://github.com/leanprover-community/mathlib4.git /opt/mathlib",
        "python -c \"import subprocess; assert subprocess.check_output(['git','-C','/opt/mathlib','rev-parse','HEAD'],text=True).strip()=='c44e0c8ee63ca166450922a373c7409c5d26b00b'\"",
        "cd /opt/mathlib && lake exe cache get Mathlib/Data/Real/Basic.lean Mathlib/Tactic/Linarith.lean Mathlib/Tactic/Ring.lean Mathlib/Tactic/FieldSimp.lean Mathlib/Algebra/Order/BigOperators/Group/Finset.lean Mathlib/Algebra/BigOperators/Field.lean",
    )
    .add_local_file(
        ROOT / "formal/WeightedEvents.lean", remote_path="/research/WeightedEvents.lean"
    )
    .add_local_file(ROOT / "configs/weighted-events-v1.json", remote_path="/research/config.json")
)


def snapshot():
    import json
    import hashlib

    base = Path("/research") if Path("/research/WeightedEvents.lean").exists() else ROOT
    saved = {
        "cloud.py": Path(__file__).read_text(),
        "proof.lean": (
            base / "WeightedEvents.lean"
            if base == Path("/research")
            else base / "formal/WeightedEvents.lean"
        ).read_text(),
        "config.json": (
            base / "config.json"
            if base == Path("/research")
            else base / "configs/weighted-events-v1.json"
        ).read_text(),
    }
    return saved, hashlib.sha256(json.dumps(saved, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=300,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def check():
    import subprocess
    import json
    import hashlib
    import re
    import time

    started = time.monotonic()
    saved, digest = snapshot()
    cfg = json.loads(saved["config.json"])
    if re.search(r"\b(sorry|admit|axiom)\b", saved["proof.lean"]):
        raise ValueError("Admitted or new-axiom declarations are forbidden")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd="/opt/mathlib", text=True
    ).strip()
    if commit != cfg["mathlib_commit"]:
        raise ValueError("Pinned mathlib checkout changed")
    result = subprocess.run(
        ["lake", "env", "lean", "/research/WeightedEvents.lean"],
        cwd="/opt/mathlib",
        capture_output=True,
        text=True,
    )
    negative_path = Path("/tmp/invalid-coupling-control.lean")
    negative_path.write_text("example : False := by exact True.intro\n")
    negative = subprocess.run(
        ["lake", "env", "lean", str(negative_path)],
        cwd="/opt/mathlib",
        capture_output=True,
        text=True,
    )
    passed = (
        result.returncode == 0
        and "sorryAx" not in result.stdout + result.stderr
        and negative.returncode != 0
    )
    root = Path("/artifacts") / ("weighted-events-" + digest[:16])
    root.mkdir(exist_ok=True)
    (root / "source.json").write_text(json.dumps(saved, indent=2))
    (root / "lean-output.txt").write_text(result.stdout + result.stderr)
    report = dict(
        status="complete" if passed else "failed",
        source_digest=digest,
        mathlib_commit=commit,
        lean_version=subprocess.check_output(["lean", "--version"], text=True).strip(),
        returncode=result.returncode,
        invalid_proof_control_returncode=negative.returncode,
        invalid_proof_control_output=negative.stdout + negative.stderr,
        compiler_output=result.stdout + result.stderr,
        proof_sha256=hashlib.sha256(saved["proof.lean"].encode()).hexdigest(),
        theorems=cfg["theorems"],
        scope=cfg["scope"],
        wall_seconds=time.monotonic() - started,
        root=str(root),
    )
    (root / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    saved, digest = snapshot()
    reservation = BudgetLedger(ROOT / ".campaign/uncertainty-budget.sqlite", 8).reserve(
        "weighted-events-" + digest, 0.02
    )
    state = ROOT / ".campaign" / ("weighted-events-" + digest[:16] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = check.spawn()
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id, source_digest=digest, reservation=reservation
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    result = call.get()
    (ROOT / "reports/preflight" / ("weighted-events-" + digest[:16] + ".json")).write_text(
        json.dumps(result, indent=2)
    )
    (ROOT / "reports/source-snapshots" / ("weighted-events-" + digest[:16] + ".json")).write_text(
        json.dumps(saved, indent=2)
    )
    print(json.dumps(result), flush=True)
