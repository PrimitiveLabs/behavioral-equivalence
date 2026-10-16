"""Measure cloud CPU training kernels before selecting a hardware backend."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-cpu-training-throughput-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.7.1", "numpy==2.2.6")
    .env({"PYTHONPATH": "/research/src", "OMP_NUM_THREADS": "16", "OPENBLAS_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image,
    cpu=(16, 16),
    memory=(4096, 4096),
    timeout=300,
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def run():
    import hashlib
    import json
    import time
    import torch
    from behavioral_equivalence.content_neural import build_content_model

    cpuinfo = Path("/proc/cpuinfo").read_text()
    flags = (
        next(line for line in cpuinfo.splitlines() if line.startswith("flags"))
        .split(":", 1)[1]
        .split()
    )
    bf16 = "avx512_bf16" in flags or "amx_bf16" in flags
    model_name = (
        next(line for line in cpuinfo.splitlines() if line.startswith("model name"))
        .split(":", 1)[1]
        .strip()
    )
    records = []
    for threads in [8]:
        torch.set_num_threads(threads)
        for architecture in ["gru", "transformer"]:
            for mixed in [False, True]:
                torch.manual_seed(20261015)
                model = build_content_model(architecture, 65535, 128)
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
                    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=mixed):
                        loss = torch.nn.functional.binary_cross_entropy_with_logits(
                            model(*inputs), targets
                        )
                    if not torch.isfinite(loss):
                        raise ArithmeticError("Nonfinite CPU benchmark loss")
                    loss.backward()
                    optimizer.step()
                seconds = (time.monotonic() - began) / 10
                records.append(
                    {
                        "architecture": architecture,
                        "threads": threads,
                        "autocast_bfloat16": mixed,
                        "seconds_per_update": seconds,
                        "cases_per_second": 512 / seconds,
                    }
                )
    sources = {"benchmark_cpu_training.py": Path(__file__).read_text()}
    for name in ["neural.py", "exposure_neural.py", "content_neural.py"]:
        sources[name] = (Path("/research/src/behavioral_equivalence") / name).read_text()
    digest = hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
    report = {
        "status": "complete",
        "source_digest": digest,
        "cpu_model": model_name,
        "hardware_bfloat16": bf16,
        "records": records,
        "scope": "Synthetic throughput and finite-loss check only; no human model evaluation.",
    }
    out = Path("/artifacts/cpu-training-throughput-v1") / digest
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
    print(json.dumps(call.get(), indent=2), flush=True)
