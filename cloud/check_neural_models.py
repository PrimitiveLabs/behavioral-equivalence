"""CPU component tests independent of GPU allocation availability."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-neural-component-tests-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6", "pytest==8.4.1")
    .env({"PYTHONPATH": "/research/src", "OMP_NUM_THREADS": "8", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
    .add_local_file(
        ROOT / "tests/test_exposure_neural.py",
        remote_path="/research/tests/test_exposure_neural.py",
    )
    .add_local_file(
        ROOT / "tests/test_content_neural.py", remote_path="/research/tests/test_content_neural.py"
    )
)


@app.function(
    image=image,
    cpu=(8, 8),
    memory=(4096, 4096),
    timeout=300,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run() -> dict:
    import hashlib
    import json
    import subprocess
    import time
    import torch
    from behavioral_equivalence.content_neural import build_content_model

    sources = {
        "check_neural_models.py": Path(__file__).read_text(),
        "test_exposure_neural.py": Path("/research/tests/test_exposure_neural.py").read_text(),
        "test_content_neural.py": Path("/research/tests/test_content_neural.py").read_text(),
    }
    for name in ["neural.py", "exposure_neural.py", "content_neural.py"]:
        sources[name] = (Path("/research/src/behavioral_equivalence") / name).read_text()
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    result = subprocess.run(
        ["python", "-m", "pytest", "-q", "-W", "error", "/research/tests"],
        capture_output=True,
        text=True,
        timeout=90,
    )
    torch.set_num_threads(8)
    timings = []
    for architecture in ["context", "gru", "transformer"]:
        model = build_content_model(architecture, 65535, hidden=128)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        inputs = [
            torch.randint(1, 18, (512, 32)),
            torch.randint(0, 15, (512,)),
            torch.randint(0, 65536, (512,)),
            torch.zeros((512, 2)),
        ]
        targets = torch.arange(512).float() % 2
        for step in range(13):
            if step == 3:
                began = time.monotonic()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(model(*inputs), targets)
            loss.backward()
            optimizer.step()
        seconds = (time.monotonic() - began) / 10
        timings.append(
            {
                "architecture": architecture,
                "training_step_seconds": seconds,
                "cases_per_second": 512 / seconds,
                "cpu_threads": 8,
                "parameters": sum(p.numel() for p in model.parameters()),
            }
        )
    report = {
        "status": "complete" if result.returncode == 0 else "failed",
        "source_digest": digest,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "throughput_preflight": timings,
        "scope": "CPU component tests and synthetic full-size throughput; no human fitting or final test prediction",
    }
    out = Path("/artifacts/neural-component-tests-v1") / digest
    out.mkdir(parents=True, exist_ok=True)
    (out / "source.json").write_text(json.dumps(sources, indent=2))
    (out / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    import json

    call = run.spawn()
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": run.object_id}), flush=True
    )
    result = call.get()
    print(json.dumps(result, indent=2), flush=True)
    if result["status"] != "complete":
        raise RuntimeError("Neural component tests failed")
