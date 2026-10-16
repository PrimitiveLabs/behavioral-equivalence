"""Measured Figure 5: fixed witness and complete matched-control comparisons."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-figure05-v1")
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
    max_containers=1,
    retries=0,
    volumes={"/artifacts": volume},
)
def render(campaign: str, analysis_directory: str):
    import csv
    import hashlib
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from behavioral_equivalence.synthetic import generate
    from behavioral_equivalence.intervention_space import environment_chain
    from behavioral_equivalence.core import quotient, reachability

    if not campaign.replace("-", "").isalnum() or not analysis_directory.replace("-", "").isalnum():
        raise ValueError("Valid source and analysis names required")
    volume.reload()
    root = Path("/artifacts") / campaign
    raw = (root / analysis_directory / "summary.json").read_bytes()
    summary = json.loads(raw)
    if (
        summary["status"] != "complete"
        or summary["audited_worlds"] != 400
        or summary["model_certificate_bindings_checked"] != 25600
    ):
        raise ValueError("Complete audited intervention analysis required")
    manifest_raw = (root / "manifest.json").read_bytes()
    if hashlib.sha256(manifest_raw).hexdigest() != summary["manifest_sha256"]:
        raise ValueError("Full benchmark differs from audited analysis")
    full_manifest = json.loads(manifest_raw)
    frozen = json.loads((root / "source.json").read_text())
    for name in ["core.py", "synthetic.py", "intervention_space.py"]:
        if frozen[name] != (Path("/research/src/behavioral_equivalence") / name).read_text():
            raise ValueError("Witness computation differs from frozen campaign")
    spec = dict(family="trust", size=128, seed=40000, surface="localized")
    identity = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    example_raw = (root / "worlds" / identity / "report.json").read_bytes()
    example = json.loads(example_raw)
    selection = next(s for s in example["searches"] if s["method"] == "rbf_ucb")
    a, b = selection["witness_states"]
    world = generate("trust", 128, 40000)
    original = tuple(tuple(block) for block in selection["partition_maps"]["unchanged"])
    refined = tuple(tuple(block) for block in selection["partition_maps"]["targeted_witness"])
    if not any(a in block and b in block for block in original):
        raise AssertionError("Witness states were not initially equivalent")
    changed = environment_chain(world, tuple(selection["witness_point"]), "localized")
    before, _ = quotient(changed, original)
    after, _ = quotient(changed, refined)

    def values(model, blocks=None):
        p = reachability(model, "complete", 10)
        return [
            float(
                p[s]
                if blocks is None
                else p[next(i for i, block in enumerate(blocks) if s in block)]
            )
            for s in [a, b]
        ]

    stages = [
        ("Neutral training environment", values(world.chain)),
        ("Changed environment: true states", values(changed)),
        ("Changed environment: merged forecast", values(before, original)),
        ("Changed environment: refined forecast", values(after, refined)),
    ]
    if abs(stages[0][1][0] - stages[0][1][1]) > 1e-12:
        raise AssertionError("Neutral example does not preserve equivalence")
    digest = hashlib.sha256(
        Path(__file__).read_bytes() + raw + example_raw + manifest_raw
    ).hexdigest()
    out = root / ("figure05-" + digest[:12])
    out.mkdir(exist_ok=True)
    (out / "render-source.py").write_bytes(Path(__file__).read_bytes())

    def table(name, rows):
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    example_rows = [
        dict(stage=stage, state=state, completion_probability=p)
        for stage, probabilities in stages
        for state, p in zip([a, b], probabilities)
    ]
    table("witness-probabilities.csv", example_rows)
    paired = [
        dict(
            comparison=i,
            scope=r["scope"],
            candidate="/".join(r["candidate"]),
            reference="/".join(r["reference"]),
            **seed,
        )
        for i, r in enumerate(summary["comparisons"])
        for seed in r["seed_effects"]
    ]
    table("paired-seed-effects.csv", paired)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.edgecolor": "#777777",
            "axes.labelcolor": "#202020",
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    figures = []
    fig, ax = plt.subplots(figsize=(12.6, 5.9))
    fig.subplots_adjust(left=0.35, right=0.96, bottom=0.25, top=0.76)
    colors = ["#245b78", "#a5493d"]
    ys = [3, 2, 1, 0]
    for (_, p), y in zip(stages, ys):
        ax.plot(p, [y, y], color="#aaaaaa", lw=1.2, zorder=1)
        for j, (value, color) in enumerate(zip(p, colors)):
            ax.scatter(
                value,
                y + (0.035 if j == 0 else -0.035),
                s=65,
                color=color,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )
    ax.set_yticks(ys, [s[0] for s in stages])
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlabel("Completion probability within 10 transitions", labelpad=12)
    ax.grid(axis="x", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)
    fig.text(
        0.04,
        0.94,
        "An intervention exposes a hidden state distinction",
        fontsize=19,
        weight="medium",
    )
    fig.text(
        0.04,
        0.865,
        f"Fixed synthetic example: trust family, 128 states, seed 40000. RBF search, eight queries.\nWitness environment ({selection['witness_point'][0]:.3f}, {selection['witness_point'][1]:.3f}); {len(original)} original blocks, {len(refined)} refined blocks.",
        fontsize=11,
    )
    for x, state, color in zip([0.36, 0.58], [a, b], colors):
        fig.text(x, 0.13, f"● State {state}", color=color, fontsize=11)
    fig.text(
        0.04,
        0.055,
        "The changed kernel is known in this example. Forecast differences isolate abstraction error; they do not test environment learning.",
        fontsize=10,
        color="#555555",
    )
    figures.append(("witness", fig))
    fig, ax = plt.subplots(figsize=(12.6, 6.5))
    fig.subplots_adjust(left=0.36, right=0.96, bottom=0.24, top=0.75)
    labels = [
        "Bundled primary comparison (95%)",
        "Random search",
        "Maximin grid",
        "RBF acquisition",
        "Local search",
    ]
    positions = [5, 3, 2, 1, 0]
    endpoints = []
    for r, y in zip(summary["comparisons"], positions):
        values = np.array([s["paired_mean_difference"] for s in r["seed_effects"]]) * 100
        lo, hi = np.array(r["seed_cluster_bootstrap_interval"]) * 100
        endpoints.extend([lo, hi, *values])
        ax.scatter(
            values, y + np.linspace(-0.13, 0.13, len(values)), s=16, color="#bbbbbb", zorder=2
        )
        ax.hlines(y, lo, hi, color="#242424", lw=1.7, zorder=3)
        ax.scatter(
            r["paired_mean_difference"] * 100, y, marker="D", s=42, color="#242424", zorder=4
        )
    lo, hi = min([0, *endpoints]), max([0, *endpoints])
    padding = max(0.03, (hi - lo) * 0.10)
    ax.set_xlim(lo - padding, hi + padding)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_yticks(positions, labels)
    ax.tick_params(axis="y", length=0, pad=12)
    ax.set_ylim(-0.6, 5.6)
    ax.set_xlabel(
        "Change in held-out maximum probability error (percentage points)\nNegative favors targeted refinement",
        labelpad=13,
    )
    ax.text(
        0,
        4.05,
        "Within each search: same queried kernels and state count; 98.75% intervals",
        transform=ax.get_yaxis_transform(),
        fontsize=9,
        color="#555555",
    )
    fig.text(
        0.04, 0.94, "Refinement versus matched-capacity controls", fontsize=19, weight="medium"
    )
    fig.text(
        0.04,
        0.86,
        "300 response-changing worlds; 16 fixed held-out environments; horizon 10.\nTen shared seed clusters, not individual worlds or predictions, determine uncertainty.",
        fontsize=11,
    )
    fig.text(
        0.04,
        0.105,
        "Gray dots: seed-level effects. Black diamond: paired mean. Lines: seed-cluster bootstrap intervals (5,000 replications).",
        fontsize=9.5,
        color="#555555",
    )
    fig.text(
        0.04,
        0.055,
        "Primary compares RBF/targeted with random-search/random-capacity strategies and can differ in capacity. The four lower rows isolate refinement.",
        fontsize=9.2,
        color="#555555",
    )
    figures.append(("comparisons", fig))
    methods = ["random", "maximin_grid", "rbf_ucb", "local_search"]
    controls = [
        "unchanged",
        "targeted_witness",
        "random_witness_same_data",
        "random_capacity",
        "all_observed",
        "identity",
    ]
    frontier = []
    for method in methods:
        for control in controls:
            for seed in range(40000, 40010):
                rows = [
                    row
                    for result in full_manifest["results"]
                    if result["spec"]["surface"] != "negative_control"
                    and result["spec"]["seed"] == seed
                    for row in result["means"]
                    if row["method"] == method
                    and row["control"] == control
                    and row["horizon"] == 10
                ]
                if len(rows) != 30:
                    raise ValueError("Complete seed-level descriptive frontier required")
                frontier.append(
                    dict(
                        method=method,
                        control=control,
                        seed=seed,
                        world_cells=len(rows),
                        mean_error=float(
                            np.mean([r["mean_total_prediction_max_error"] for r in rows])
                        ),
                        mean_states=float(np.mean([r["states"] for r in rows])),
                    )
                )
    table("full-control-frontier.csv", frontier)
    targeted = {
        m: np.mean(
            [
                r["mean_error"]
                for r in frontier
                if r["method"] == m and r["control"] == "targeted_witness"
            ]
        )
        for m in methods
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.17, top=0.79, wspace=0.50, hspace=0.35)
    control_labels = [
        "Unchanged",
        "Targeted witness",
        "Random witness",
        "Matched random split",
        "All queried environments",
        "Full-state identity",
    ]
    maximum = max(r["mean_error"] for r in frontier) * 1.08
    for ax, method, title in zip(
        axes.flat, methods, ["Random search", "Maximin grid", "RBF acquisition", "Local search"]
    ):
        for i, control in enumerate(controls):
            values = [
                r["mean_error"]
                for r in frontier
                if r["method"] == method and r["control"] == control
            ]
            y = 5 - i
            ax.scatter(values, y + np.linspace(-0.12, 0.12, 10), s=12, color="#bbbbbb")
            ax.scatter(np.mean(values), y, marker="D", s=32, color="#242424", zorder=3)
        ax.set_title(title, loc="left", fontsize=12, pad=10)
        ax.set_yticks(range(6), list(reversed(control_labels)), fontsize=9)
        ax.tick_params(axis="y", length=0)
        ax.set_xlim(0, maximum)
        ax.set_ylim(-0.5, 5.5)
        ax.grid(axis="x", color="#e1e1e1", lw=0.6)
        ax.set_axisbelow(True)
    fig.text(
        0.04, 0.94, "All search methods and all partition controls", fontsize=19, weight="medium"
    )
    fig.text(
        0.04,
        0.865,
        "Descriptive horizon-10 frontier over all 300 response-changing worlds.\nEvery search has eight queries; only targeted and random-split controls within a method have exactly matched capacity.",
        fontsize=11,
    )
    fig.supxlabel("Mean held-out maximum outcome-probability error", y=0.105, fontsize=11)
    fig.text(
        0.04,
        0.045,
        "Gray dots: ten seed-level means. Black diamonds: their mean. Full-state identity isolates response-surface learning error.",
        fontsize=9.5,
        color="#555555",
    )
    figures.append(("full-frontier", fig))
    with PdfPages(out / "figure-05-interventions.pdf") as pdf:
        for name, figure in figures:
            pdf.savefig(figure)
            for suffix in ["png", "svg", "pdf"]:
                figure.savefig(out / (name + "." + suffix), dpi=180)
            plt.close(figure)
    primary = summary["comparisons"][0]
    caption = (
        "# Figure 5. Counterexample-guided refinement and held-out controls\n\n"
        "Panel A uses a fixed trust-family 128-state world (seed 40000, localized response surface). RBF acquisition chooses a witness from eight queries. Both displayed states share an exact training equivalence class. The changed kernel is known here, so merged and refined forecasts isolate state-abstraction error. The states, probabilities and selected environment are measured outputs, not psychological profiles.\n\n"
        "Panel B reports the complete response-changing portion of the frozen 400-world experiment: 300 worlds across three surfaces, with 100 structural negative controls reported separately. Errors compare learned environment-response kernels and their quotients with actual held-out synthetic kernels. The primary strategy comparison can differ in capacity. Four within-method comparisons hold queried kernels and reduced state count fixed. Effects average within each seed over mechanisms, sizes, surfaces and the same sixteen evaluation coordinates. Gray dots show ten shared seed-cluster effects. Bootstrap intervals use 5,000 resamples: 95% for the primary and 98.75% for each member of the four-comparison family. These limited-cluster summaries condition on the generator and fixed catalogue; they are not distribution-free confidence certificates.\n\n"
        f"The primary mean change is {primary['paired_mean_difference']:.8f}, with interval {primary['seed_cluster_bootstrap_interval']}. The maximum mean prediction discrepancy in structural negative controls is {summary['negative_control_max_mean_prediction_error']:.5g}.\n\n"
        f"Panel C shows every search method and partition control descriptively, without additional hypothesis tests. With targeted refinement in both strategies, RBF minus random-search error is {targeted['rbf_ucb'] - targeted['random']:.8f}, and RBF minus maximin-grid error is {targeted['rbf_ucb'] - targeted['maximin_grid']:.8f}. Positive differences mean RBF is worse. The bundled primary comparison therefore does not establish an advantage for adaptive search. State counts can differ slightly across search strategies; the CSV records them.\n\n"
        "Formal certificates apply to estimated full-kernel versus quotient discrepancies. They do not certify environment-response learning or human generalization. Source CSVs retain every displayed probability and seed effect.\n"
    )
    (out / "caption.md").write_text(caption)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir()
        if p.name != "manifest.json"
    }
    manifest = dict(
        status="rendered; visual inspection pending",
        figure=5,
        campaign_id=campaign,
        analysis_source_digest=summary["analysis_source_digest"],
        analysis_sha256=hashlib.sha256(raw).hexdigest(),
        example_report_sha256=hashlib.sha256(example_raw).hexdigest(),
        render_digest=digest,
        artifact_sha256=files,
        root=str(out),
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return manifest


@app.local_entrypoint()
def main(
    campaign: str = "intervention-main-v1-f38425758e38",
    analysis_directory: str = "analysis-773c68323e0f7a90",
):
    import json

    call = render.spawn(campaign, analysis_directory)
    print(
        json.dumps(dict(function_call_id=call.object_id, function_id=render.object_id)), flush=True
    )
    print(json.dumps(call.get()), flush=True)
