"""Measured Figure 4: precision frontier and actual robust probability ranges."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-figure04-v1")
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
def render(campaign: str) -> dict:
    import csv
    import gzip
    import hashlib
    import json
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from behavioral_equivalence.core import Chain, quotient, reachability
    from behavioral_equivalence.integer_bounds import integer_kernel, reachability_enclosures

    if not campaign.replace("-", "").isalnum():
        raise ValueError("Invalid campaign")
    root = Path("/artifacts") / campaign
    raw = (root / "summary.json").read_bytes()
    summary = json.loads(raw)
    if (
        summary["status"] != "complete"
        or summary["worlds"] != 200
        or summary["paired_transition_datasets"] != 600
    ):
        raise ValueError("Complete audited uncertainty campaign required")
    for name in ["core.py", "integer_bounds.py"]:
        if (root / "source" / name).read_bytes() != (
            Path("/research/src/behavioral_equivalence") / name
        ).read_bytes():
            raise ValueError("Figure computation differs from frozen numerical source")
    digest = hashlib.sha256(Path(__file__).read_bytes() + raw).hexdigest()
    out = root / ("figure04-" + digest[:12])
    out.mkdir(exist_ok=False)
    (out / "render-source.py").write_bytes(Path(__file__).read_bytes())

    def table(name, rows):
        with (out / name).open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    table("mean-widths.csv", summary["widths"])
    style = {
        "certified_hoeffding": ("#565b61", "o", "Hoeffding, exact tail certificate"),
        "clopper_pearson_float_quantiles": ("#1b6c87", "s", "Clopper-Pearson, floating quantiles"),
    }
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#30353a",
            "text.color": "#242a30",
            "axes.edgecolor": "#b7bdc1",
            "axes.linewidth": 0.6,
            "xtick.color": "#5b6065",
            "ytick.color": "#5b6065",
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 7.1), sharex=True, sharey=True)
    for row, size in enumerate([32, 128]):
        for column, h in enumerate([5, 10, 25, 50]):
            ax = axes[row, column]
            for method, (color, marker, label) in style.items():
                points = sorted(
                    [
                        r
                        for r in summary["widths"]
                        if r["method"] == method
                        and r["size"] == size
                        and r["property"] == "complete"
                        and r["horizon"] == h
                    ],
                    key=lambda r: r["samples_per_row"],
                )
                if len(points) != 3 or any(r["independent_worlds"] != 100 for r in points):
                    raise ValueError("Incomplete plotted mean grid")
                ax.plot(
                    [r["samples_per_row"] for r in points],
                    [r["mean_active_state_width"] for r in points],
                    color=color,
                    marker=marker,
                    markersize=4,
                    linewidth=1.5,
                    label=label,
                )
            ax.set_xscale("log")
            ax.set_xticks([100, 1000, 10000], ["100", "1,000", "10,000"])
            ax.minorticks_off()
            ax.set_ylim(-0.02, 1.04)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
            ax.grid(axis="y", color="#e9ebed", linewidth=0.6)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(f"H = {h} transitions", fontsize=11, pad=11)
            if column == 0:
                ax.set_ylabel(f"{size} concrete states\nMean interval width", labelpad=9)
    fig.suptitle(
        "Verified intervals can remain wide",
        x=0.075,
        y=0.965,
        ha="left",
        fontsize=19,
        fontweight="medium",
    )
    fig.text(
        0.075,
        0.912,
        "Completion reachability: means over 100 synthetic worlds per model size",
        fontsize=11,
        color="#5b6065",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower left",
        bbox_to_anchor=(0.067, 0.075),
        frameon=False,
        ncol=2,
        fontsize=9,
    )
    fig.supxlabel("Independent transitions sampled per concrete state", y=0.165, fontsize=11)
    fig.text(
        0.075,
        0.045,
        "200 worlds; 600 paired transition datasets. Rectangular uncertainty envelope includes state abstraction.",
        fontsize=9,
        color="#5b6065",
    )
    fig.text(
        0.075,
        0.021,
        "Means only; paired world-bootstrap comparisons are in the source report. Synthetic model bounds, not human guarantees.",
        fontsize=9,
        color="#5b6065",
    )
    fig.subplots_adjust(left=0.08, right=0.985, bottom=0.255, top=0.835, hspace=0.29, wspace=0.16)
    fig.savefig(out / "figure-04a-width.png", dpi=200)
    fig.savefig(out / "figure-04a-width.svg")
    fig.savefig(out / "figure-04a-width.pdf")
    first_figure = fig
    spec = dict(family="trust", size=128, seed=20000)
    key = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    world = root / "worlds" / key
    report_bytes = (world / "report.json").read_bytes()
    if hashlib.sha256(report_bytes).hexdigest() != summary["world_report_sha256"][key]:
        raise ValueError("Selected source report changed")
    report = json.loads(report_bytes)
    saved = np.load(world / "world.npz")
    concrete = Chain(saved["p"], tuple(saved["labels"].tolist()), saved["initial"])
    counts = np.load(world / "counts-10000.npz")["counts"]
    truth = reachability_enclosures(
        integer_kernel(concrete), concrete.labels, frozenset({"complete"}), [5, 10, 25, 50]
    )
    cases = []
    for method in style:
        record = next(
            r for r in report["records"] if r["method"] == method and r["samples_per_row"] == 10000
        )
        packed = (world / record["certificate"]).read_bytes()
        if hashlib.sha256(packed).hexdigest() != record["certificate_sha256"]:
            raise ValueError("Selected certificate changed")
        cert = json.loads(gzip.decompress(packed))["evidence"]
        blocks = tuple(tuple(b) for b in cert["uniform_abstraction"]["partition"])
        abstract_origin = next(i for i, b in enumerate(blocks) if 0 in b)
        estimate = Chain(counts / 10000, concrete.labels, concrete.initial)
        nominal, _ = quotient(estimate, blocks)
        for h in [5, 10, 25, 50]:
            values = cert["robust_values"]["complete"]["values"][str(h)]
            d = 1 << cert["robust_values"]["complete"]["bits"]
            cases.append(
                {
                    "method": method,
                    "horizon": h,
                    "lower": values["min_lower"][abstract_origin] / d,
                    "upper": values["max_upper"][abstract_origin] / d,
                    "nominal": float(reachability(nominal, "complete", h)[abstract_origin]),
                    "known_model_lower": truth[h][0][0] / (1 << 48),
                    "known_model_upper": truth[h][1][0] / (1 << 48),
                    "concrete_origin": 0,
                    "samples_per_row": 10000,
                }
            )
    table("range-example.csv", cases)
    fig, ax = plt.subplots(figsize=(12.6, 5.5))
    ys = []
    names = []
    for i, h in enumerate([5, 10, 25, 50]):
        for j, method in enumerate(style):
            case = next(r for r in cases if r["method"] == method and r["horizon"] == h)
            y = 7 - i * 2 - j
            color, _, _ = style[method]
            ax.plot(
                [case["lower"], case["upper"]],
                [y, y],
                color=color,
                linewidth=5,
                solid_capstyle="butt",
                alpha=0.65,
            )
            ax.scatter(case["nominal"], y, marker="D", s=22, color=color, zorder=4)
            ax.scatter(
                (case["known_model_lower"] + case["known_model_upper"]) / 2,
                y,
                marker="|",
                s=130,
                color="#171c20",
                zorder=5,
            )
            ys.append(y)
            names.append(f"H={h}  {'Hoeffding' if j == 0 else 'Clopper-Pearson'}")
    ax.set_yticks(ys, names)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    ax.set_xlabel("Completion probability within H transitions", labelpad=10)
    ax.spines["left"].set_visible(False)
    ax.grid(axis="x", color="#edf0f2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.scatter([], [], marker="D", s=22, color="#65727a", label="Nominal quotient estimate")
    ax.scatter([], [], marker="|", s=130, color="#171c20", label="Known generating model")
    fig.legend(
        *ax.get_legend_handles_labels(),
        loc="upper left",
        bbox_to_anchor=(0.23, 0.855),
        frameon=False,
        fontsize=9,
        ncol=2,
    )
    fig.suptitle(
        "A fixed synthetic example: nominal predictions and verified ranges",
        x=0.035,
        y=0.965,
        ha="left",
        fontsize=16,
    )
    fig.text(
        0.035,
        0.895,
        "Trust-family world, 128 states, seed 20000; 10,000 sampled transitions per row; concrete origin 0.",
        fontsize=10,
        color="#5b6065",
    )
    fig.text(
        0.035,
        0.055,
        "Intervals include hidden-state and transition uncertainty. No additional nominal abstraction allowance is added.",
        fontsize=9,
        color="#5b6065",
    )
    fig.text(
        0.035,
        0.025,
        "Known-model marker uses integer-enclosure midpoints. Clopper-Pearson quantiles remain floating. No human prediction claim.",
        fontsize=9,
        color="#5b6065",
    )
    fig.subplots_adjust(left=0.23, right=0.975, bottom=0.19, top=0.78)
    fig.savefig(out / "figure-04b-ranges.png", dpi=200)
    fig.savefig(out / "figure-04b-ranges.svg")
    fig.savefig(out / "figure-04b-ranges.pdf")
    with PdfPages(out / "figure-04-uncertainty.pdf") as pdf:
        pdf.savefig(first_figure)
        pdf.savefig(fig)
    plt.close("all")
    manifest = {
        "status": "complete",
        "figure": "04",
        "scope": "measured synthetic uncertainty, not final human fidelity",
        "source_campaign": campaign,
        "source_protocol_digest": summary["protocol_digest"],
        "source_summary_sha256": hashlib.sha256(raw).hexdigest(),
        "example_spec": spec,
        "example_selection": "fixed first main-campaign seed in the trust family at size 128; no selection by outcome",
        "render_source_digest": digest,
        "files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(out.iterdir())
            if p.is_file()
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    volume.commit()
    return {"status": "complete", "artifact_root": str(out), "files": list(manifest["files"])}


@app.local_entrypoint()
def main(campaign: str = "uncertainty-main-v1-9550b167f4b9"):
    import json

    call = render.spawn(campaign)
    print(
        json.dumps({"function_call_id": call.object_id, "function_id": render.object_id}),
        flush=True,
    )
    print(json.dumps(call.get(), indent=2), flush=True)
