"""Measured Figure 3: full synthetic error-versus-bound comparisons."""

from pathlib import Path, PurePosixPath
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-figure03-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research", create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "matplotlib==3.10.3")
    .env({"PYTHONPATH": "/research/src", "OPENBLAS_NUM_THREADS": "1", "MPLBACKEND": "Agg"})
    .add_local_dir(ROOT / "src", remote_path="/research/src")
)


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(4096, 4096),
    timeout=600,
    retries=0,
    volumes={"/artifacts": volume},
)
def render(summary_root: str):
    import csv
    import gzip
    import hashlib
    import json
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from behavioral_equivalence.core import coupling_bound

    path = PurePosixPath(summary_root)
    if path.is_absolute() or any(not p.replace("-", "").isalnum() for p in path.parts):
        raise ValueError("Relative synthetic summary root required")
    volume.reload()
    root = Path("/artifacts") / summary_root
    raw = (root / "summary.json").read_bytes()
    summary = json.loads(raw)
    if (
        summary["status"] != "complete"
        or summary["worlds"] != 1200
        or summary["method_variants"] != 26400
        or summary["verified_integer_certificates"] != 26400
        or summary["floating_bound_violations"] != 0
    ):
        raise ValueError("Full reconciled and audited matrix required")
    for name in ["methods.csv.gz", "properties.csv.gz"]:
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != summary["output_sha256"][name]:
            raise ValueError("Figure source table changed")
    with gzip.open(root / "methods.csv.gz", "rt", newline="") as f:
        methods = list(csv.DictReader(f))
    with gzip.open(root / "properties.csv.gz", "rt", newline="") as f:
        properties = list(csv.DictReader(f))

    def identity(row):
        return tuple(row[n] for n in ["family", "size", "perturbation", "seed", "method_index"])

    radii = {identity(row): float(row["epsilon_exact_normalized_for_display"]) for row in methods}
    if len(radii) != 26400 or len(properties) != 105600:
        raise ValueError("Missing or duplicated plot inputs")
    display = []
    for row in properties:
        h = int(row["horizon"])
        error = float(row["maximum_unconditional_floating_error"])
        epsilon = radii[identity(row)]
        bound = coupling_bound(epsilon, h)
        display.append(
            dict(
                family=row["family"],
                size=int(row["size"]),
                perturbation=float(row["perturbation"]),
                seed=int(row["seed"]),
                method_index=int(row["method_index"]),
                method=row["method"],
                parameters=row["parameters"],
                horizon=h,
                maximum_unconditional_floating_error=error,
                exact_normalized_epsilon_for_display=epsilon,
                coupling_bound_for_display=bound,
                original_floating_bound=float(row["floating_coupling_bound"]),
            )
        )
    digest = hashlib.sha256(Path(__file__).read_bytes() + raw).hexdigest()
    out = root / ("figure03-" + digest[:12])
    out.mkdir(exist_ok=False)
    (out / "render-source.py").write_bytes(Path(__file__).read_bytes())
    with gzip.open(out / "plotted-comparisons.csv.gz", "wt", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(display[0]))
        w.writeheader()
        w.writerows(display)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#a5adb2",
            "axes.linewidth": 0.6,
        }
    )
    style = {
        "random_merging": ("#9ca3a8", "Random merging"),
        "feature_embedding_clustering": ("#bb8058", "Embedding clustering"),
        "predictive_clustering": ("#518f7b", "Predictive clustering"),
        "exact_rational": ("#28343b", "Exact rational partition"),
        "approximate_bisimulation": ("#31769a", "Approximate bisimulation"),
        "feature_initialized_verified_refinement": ("#8d679d", "Verified refinement"),
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 9.0), sharex=True, sharey=True)
    for ax, h in zip(axes.ravel(), [5, 10, 25, 50]):
        selected = [r for r in display if r["horizon"] == h]
        for name, (color, _) in style.items():
            rows = [r for r in selected if r["method"] == name]
            ax.scatter(
                [r["coupling_bound_for_display"] for r in rows],
                [r["maximum_unconditional_floating_error"] for r in rows],
                s=5,
                alpha=0.19 if name != "exact_rational" else 0.65,
                c=color,
                linewidths=0,
                rasterized=True,
            )
        ax.plot([0, 1], [0, 1], color="#4f585e", linestyle="--", linewidth=0.8, zorder=0)
        ax.set_title(f"H = {h} transitions", loc="left", fontsize=12)
        ax.set(
            xlim=(-0.015, 1.015),
            ylim=(-0.015, 1.015),
            xticks=[0, 0.25, 0.5, 0.75, 1],
            yticks=[0, 0.25, 0.5, 0.75, 1],
        )
        ax.text(
            0.03,
            0.94,
            "26,400 quotient comparisons",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            color="#5b646b",
        )
    axes[0, 0].set_ylabel("Maximum unconditional property discrepancy")
    axes[1, 0].set_ylabel("Maximum unconditional property discrepancy")
    axes[1, 0].set_xlabel("Coupling bound, displayed from certified radius")
    axes[1, 1].set_xlabel("Coupling bound, displayed from certified radius")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="none",
            markerfacecolor=c,
            markeredgewidth=0,
            markersize=5,
            label=label,
        )
        for c, label in style.values()
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.075),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=2.4,
    )
    fig.suptitle(
        "Prediction discrepancy and its finite-horizon bound", x=0.07, ha="left", fontsize=17
    )
    fig.text(
        0.07,
        0.925,
        "1,200 synthetic worlds. All 22 method variants. Independent integer audit: 26,400 certificates.",
        fontsize=10,
    )
    fig.text(
        0.07,
        0.043,
        "Each point: maximum completion, disengagement or survival discrepancy over all related states. Dashed line: equality.",
        fontsize=9,
        color="#515a61",
    )
    fig.text(
        0.07,
        0.020,
        "Discrepancies use floating calculations; bounds show certified radii at display precision. Exact numeric scopes and tolerances are in the caption.",
        fontsize=8.5,
        color="#515a61",
    )
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.19, top=0.87, hspace=0.23, wspace=0.15)
    for suffix in ["png", "svg", "pdf"]:
        fig.savefig(out / ("figure-03-preservation." + suffix), dpi=180)
    plt.close(fig)
    caption = (
        "# Figure 3. Full synthetic prediction-preservation benchmark\n\n"
        "The 1,200-world frozen matrix crosses five mechanisms, 32/128/512 states, four perturbations and twenty seeds. "
        "Every panel contains all 26,400 quotient variants, including uncertified clustering and random controls. "
        "The vertical coordinate is the largest floating completion, disengagement or survival discrepancy across all related deterministic origins at that horizon. "
        "The horizontal coordinate displays 1-(1-epsilon)^H using the exact-normalized one-step radius recorded by the integer auditor, rounded for plotting. "
        "The directed-integer certificates remain the numerical evidence; displayed radii and dots do not replace their exact arithmetic. "
        f"The frozen floating benchmark reports no exceedance after its {summary['floating_numerical_tolerance']:g} numerical tolerance. "
        f"{len(summary['strict_requested_budget_exceedances'])} requested strict epsilon budgets are exceeded under exact-normalized interpretation, by at most {summary['maximum_exact_budget_excess_for_display']:.17g}. "
        f"{summary['serialized_exact_quotients_with_nonzero_radius']} serialized exact-partition quotients have nonzero radius, at most {summary['maximum_serialized_exact_radius_for_display']:.17g}. "
        "The certificates use actual serialized radii. These implementation observations do not invalidate their bounds, but they prevent interpreting every requested epsilon as an exact achieved tolerance. "
        "Conditional completion uses a different conditioning bound and remains in the source benchmark table, outside this unconditional-event plot. "
        "Independent PRISM checking concerns completion and disengagement at horizon 50 from state zero, rather than every plotted origin and property. "
        "Repeated seeds and related methods are paired design conditions; point density does not imply independent statistical replication. "
        "Scatter points are rasterized inside otherwise vector PDF/SVG exports to keep publication files manageable. This figure concerns specified synthetic models only.\n"
    )
    (out / "caption.md").write_text(caption)
    result = dict(
        status="rendered_pending_visual_qa",
        render_digest=digest,
        summary_sha256=hashlib.sha256(raw).hexdigest(),
        comparison_rows=len(display),
        output_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.iterdir())
            if p.is_file()
        },
        root=str(out),
    )
    (out / "manifest.json").write_text(json.dumps(result, indent=2))
    volume.commit()
    return result


@app.local_entrypoint()
def main(summary_root: str):
    import json

    call = render.spawn(summary_root)
    print(
        json.dumps(dict(function_call_id=call.object_id, function_id=render.object_id)), flush=True
    )
    print(json.dumps(call.get()), flush=True)
