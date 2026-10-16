"""Measured Figure 1: a complete synthetic graph and its audited quotient."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-figure01-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "matplotlib==3.10.3")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=300,
    retries=0,
    volumes={"/artifacts": volume},
)
def render():
    import csv
    import gzip
    import hashlib
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch
    from behavioral_equivalence.core import Chain, quotient, reachability
    from behavioral_equivalence.integer_bounds import verify_abstraction

    volume.reload()
    root = Path("/artifacts/synthetic-main-v3-d9f1fdcc7c35")
    spec = dict(family="trust", size=32, perturbation=0, seed=10000)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    source = root / "worlds" / key
    raw = (source / "report.json").read_bytes()
    report = json.loads(raw)
    audit_root = root / "integer-campaign-audit-6d9dbce18e2635f0"
    audit_raw = (audit_root / "manifest.json").read_bytes()
    audit = json.loads(audit_raw)
    entry = next(r for r in audit["records"] if r["key"] == key)
    audit_path = audit_root / "worlds" / key
    ar = (audit_path / "report.json").read_bytes()
    if (
        audit["status"] != "complete"
        or audit["worlds_audited"] != 1200
        or hashlib.sha256(raw).hexdigest() != entry["source_report_sha256"]
        or hashlib.sha256(ar).hexdigest() != entry["audit_report_sha256"]
        or report["status"] != "complete"
        or report["spec"] != spec
    ):
        raise ValueError("Completed and audited measured world required")
    for name in ["core.py", "integer_bounds.py"]:
        saved = (root / "source" / name) if name == "core.py" else (audit_root / "source.json")
        if name == "core.py":
            text = saved.read_text()
        else:
            text = json.loads(saved.read_text())[name]
        if text != (Path("/research/src/behavioral_equivalence") / name).read_text():
            raise ValueError("Figure numeric source differs from audited experiment")
    with np.load(source / "world.npz", allow_pickle=False) as a:
        concrete = Chain(a["p"], tuple(a["labels"].tolist()), a["initial"])
    if concrete.digest() != report["world_digest"]:
        raise ValueError("Measured world changed")
    index = next(i for i, m in enumerate(report["methods"]) if m["method"] == "exact_rational")
    blocks = tuple(tuple(b) for b in report["methods"][index]["partition"])
    abstract, _ = quotient(concrete, blocks)
    detail = json.loads(ar)["methods"][index]
    packed = (audit_path / detail["certificate"]).read_bytes()
    if hashlib.sha256(packed).hexdigest() != detail["certificate_sha256"]:
        raise ValueError("Measured quotient certificate changed")
    certificate = json.loads(gzip.decompress(packed))
    verify_abstraction(concrete, abstract, certificate)
    digest = hashlib.sha256(Path(__file__).read_bytes() + raw + ar + packed).hexdigest()
    out = root / ("figure01-" + digest[:12])
    out.mkdir(exist_ok=False)
    (out / "render-source.py").write_bytes(Path(__file__).read_bytes())

    def table(name, rows):
        with (out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    mapping = {s: b for b, block in enumerate(blocks) for s in block}
    table(
        "states.csv",
        [
            dict(state=s, label=concrete.labels[s], quotient_state=mapping[s])
            for s in range(concrete.size)
        ],
    )
    edges = []
    for model_name, model in [("original", concrete), ("quotient", abstract)]:
        edges.extend(
            dict(model=model_name, source=s, target=t, probability=float(model.transitions[s, t]))
            for s, t in zip(*np.nonzero(model.transitions))
        )
    table("edges.csv", edges)
    properties = []
    for target in ["complete", "disengage"]:
        for h in [5, 10, 25, 50]:
            full = float(reachability(concrete, target, h)[0])
            reduced = float(reachability(abstract, target, h)[mapping[0]])
            properties.append(
                dict(
                    target=target,
                    horizon=h,
                    original_state=0,
                    quotient_state=mapping[0],
                    original_probability=full,
                    quotient_probability=reduced,
                    floating_difference=abs(full - reduced),
                )
            )
    table("properties.csv", properties)
    plt.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 10, "pdf.fonttype": 42, "svg.fonttype": "none"}
    )
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 6.7))
    angles = np.linspace(np.pi / 2, np.pi / 2 + 2 * np.pi, len(blocks), endpoint=False)
    centers = np.column_stack([np.cos(angles), np.sin(angles)])
    original_positions = np.zeros((concrete.size, 2))
    for b, block in enumerate(blocks):
        offsets = np.linspace(0, 2 * np.pi, len(block), endpoint=False)
        original_positions[list(block)] = centers[b] + 0.105 * np.column_stack(
            [np.cos(offsets), np.sin(offsets)]
        )
    palette = {"active": "#54778c", "complete": "#39826a", "disengage": "#b16b55"}

    def graph(ax, model, positions, original):
        for s, t in zip(*np.nonzero(model.transitions)):
            p = float(model.transitions[s, t])
            if s == t:
                x, y = positions[s]
                start, end = (x - 0.025, y + 0.025), (x + 0.025, y + 0.025)
                curve = -1.7
            else:
                start, end = positions[s], positions[t]
                curve = 0.10 if s < t else -0.10
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    arrowstyle="-|>",
                    connectionstyle=f"arc3,rad={curve}",
                    mutation_scale=3 if original else 7,
                    shrinkA=4 if original else 12,
                    shrinkB=4 if original else 12,
                    linewidth=(0.15 + 0.5 * p) if original else (0.4 + 3 * p),
                    color="#7c8790",
                    alpha=0.16 if original else 0.55,
                    zorder=1,
                )
            )
        ax.scatter(
            positions[:, 0],
            positions[:, 1],
            s=42 if original else 430,
            c=[palette[label] for label in model.labels],
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        if original:
            for b, center in enumerate(centers):
                ax.text(
                    *(center * 1.23),
                    f"C{b}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="#313940",
                )
        else:
            for b, center in enumerate(centers):
                ax.text(
                    *center, str(b), ha="center", va="center", fontsize=10, color="white", zorder=4
                )
        ax.set(xlim=(-1.47, 1.47), ylim=(-1.47, 1.47), aspect="equal")
        ax.axis("off")

    graph(axes[0], concrete, original_positions, True)
    graph(axes[1], abstract, centers, False)
    axes[0].set_title(
        f"Original: {concrete.size} states, {np.count_nonzero(concrete.transitions):,} edges",
        loc="left",
        fontsize=13,
    )
    axes[1].set_title(
        f"Quotient: {abstract.size} states, {np.count_nonzero(abstract.transitions):,} edges",
        loc="left",
        fontsize=13,
    )
    fig.suptitle(
        "Different states, the same specified future behavior", x=0.05, ha="left", fontsize=17
    )
    eps = detail["epsilon_approximation_for_display"]
    fig.text(
        0.05,
        0.105,
        f"Measured trust mechanism, seed 10000, no perturbation. {concrete.size / abstract.size:g}x state reduction.",
        fontsize=10,
    )
    fig.text(
        0.05,
        0.071,
        "Every nonzero edge is shown. Original edge widths: 0.15 + 0.5p; quotient: 0.4 + 3p. Layout groups the audited classes.",
        fontsize=9,
        color="#505860",
    )
    fig.text(
        0.05,
        0.038,
        f"Blue: active. Green: complete. Brown: disengage. Serialized quotient one-step error: {eps:.3g}. Synthetic model only.",
        fontsize=9,
        color="#505860",
    )
    fig.subplots_adjust(left=0.04, right=0.97, bottom=0.16, top=0.84, wspace=0.16)
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(out / ("figure-01-abstraction." + suffix), dpi=180)
    plt.close(fig)
    caption = (
        "# Figure 1. Measured behavioral-state abstraction\n\n"
        f"The frozen trust benchmark with 32 states, perturbation 0 and seed 10000 reduces to {abstract.size} exact rational equivalence classes. "
        "Both panels contain every nonzero transition. Node locations group the original states by their audited quotient classes; positions have no psychological meaning. "
        "Separate edge-width formulas keep the dense original readable and must not be interpreted as a shared visual probability scale. The CSV records every actual probability. "
        f"The floating serialized quotient has measured exact-normalized one-step radius {eps:.17g}; the accompanying integer certificate accounts for this numerical error. "
        "The property table records completion and disengagement forecasts at horizons 5, 10, 25 and 50 from original state zero and its related abstract state. "
        "This is a synthetic stochastic mechanism, not a learned representation of human cognition.\n"
    )
    (out / "caption.md").write_text(caption)
    manifest = dict(
        status="rendered_pending_visual_qa",
        render_digest=digest,
        spec=spec,
        source_world_sha256=hashlib.sha256(raw).hexdigest(),
        audit_manifest_sha256=hashlib.sha256(audit_raw).hexdigest(),
        certificate_sha256=detail["certificate_sha256"],
        original_states=concrete.size,
        reduced_states=abstract.size,
        output_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.iterdir())
            if p.is_file()
        },
        root=str(out),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return manifest


@app.local_entrypoint()
def main():
    import json

    call = render.spawn()
    print(
        json.dumps(dict(function_call_id=call.object_id, function_id=render.object_id)), flush=True
    )
    print(json.dumps(call.get()), flush=True)
