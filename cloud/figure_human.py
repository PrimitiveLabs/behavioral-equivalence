"""Measured human Figures 2 and 6, requiring complete fixed result matrices."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-human-figures-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6", "matplotlib==3.10.3")
    .env({"MPLBACKEND": "Agg", "OPENBLAS_NUM_THREADS": "1"})
)


@app.function(
    image=image,
    cpu=(1, 1),
    memory=(2048, 2048),
    timeout=300,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def render(data):
    import json
    import hashlib
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    expected = {(p, s) for p in ["known_users", "unseen_users"] for s in range(20261015, 20261020)}
    response_expected = {
        (p, s)
        for p in ["response_all_groups", "response_heldout_groups"]
        for s in range(20261015, 20261020)
    }
    if (
        data["status"] != "complete"
        or {(r["protocol"], r["seed"]) for r in data["frontier"]} != expected
        or {(r["phase"], r["seed"]) for r in data["responses"]} != response_expected
        or len(data["inputs"]) != 20
    ):
        raise ValueError("Complete primary prediction and response matrices required")
    raw = json.dumps(data, sort_keys=True).encode()
    source = Path(__file__).read_bytes()
    digest = hashlib.sha256(source + raw).hexdigest()
    volume.reload()
    out = Path("/artifacts") / ("human-final-figures-" + digest[:16])
    if (out / "manifest.json").exists():
        report = json.loads((out / "manifest.json").read_text())
        if any(
            hashlib.sha256((out / name).read_bytes()).hexdigest() != sha
            for name, sha in report["files"].items()
        ):
            raise ValueError("Stored human figure changed")
        return report
    out.mkdir(exist_ok=False)
    (out / "render-source.py").write_bytes(source)
    (out / "plotted-aggregate-data.json").write_bytes(raw)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.edgecolor": "#aeb4bb",
            "axes.labelcolor": "#30353a",
            "text.color": "#242a30",
            "xtick.color": "#505761",
            "ytick.color": "#505761",
            "savefig.facecolor": "white",
        }
    )
    styles = {
        "strict": ("#276783", "o", "Strict certificates"),
        "finite": ("#b76b2b", "^", "Finite-property certificates"),
        "control": ("#7d848d", "x", "Matched controls"),
        "independent_teacher_quantization": ("#397d58", "s", "Independent quantization"),
        "pooled_count_model": ("#946079", "D", "Pooled count model"),
        "parent": ("#20262c", "+", "256-state parent"),
    }
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.5), sharey=True)
    for ax, protocol, title in zip(
        axes, ["known_users", "unseen_users"], ["Known users", "Unseen users"], strict=True
    ):
        rows = [r for r in data["frontier"] if r["protocol"] == protocol]
        ax.axhline(0, color="#505761", linewidth=0.8)
        ax.axhline(0.01, color="#b76b2b", linewidth=0.8, linestyle=":")
        ax.axvspan(0, 128, color="#eff2f3", zorder=0)
        for family, (color, marker, label) in styles.items():
            selected = [r for r in rows if r["family"] == family]
            if selected:
                ax.scatter(
                    [r["latent_states"] for r in selected],
                    [r["candidate_minus_parent_nats"] for r in selected],
                    s=23 if family != "parent" else 50,
                    alpha=0.6,
                    color=color,
                    marker=marker,
                    linewidths=0.7,
                    label=label,
                    zorder=5 if family == "strict" else 2,
                )
        gru = [r["candidate_minus_parent_nats"] for r in rows if r["method"] == "gru"]
        ax.axhline(float(np.mean(gru)), color="#30353a", linestyle="--", linewidth=0.9)
        ax.set(title=title, xlabel="Discrete latent states", xlim=(0, 272))
        ax.annotate(
            "Strict primary: 256 states\nin all five fits",
            xy=(256, 0),
            xytext=(105, 0.025),
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "#276783", "linewidth": 0.7},
            color="#276783",
        )
        ax.set_xticks([16, 64, 128, 192, 256])
        ax.grid(axis="y", color="#e5e8eb", linewidth=0.5)
    axes[0].set_ylabel("Held-out NLL minus 256-state parent (nats)")
    handles = [
        Line2D([], [], color=color, marker=marker, linestyle="None", label=label)
        for color, marker, label in styles.values()
    ]
    handles.append(
        Line2D([], [], color="#30353a", linestyle="--", label="GRU, mean across fixed fits")
    )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.03),
        fontsize=8,
    )
    fig.suptitle(
        "Figure 2. Measured compression and predictive loss", x=0.07, ha="left", fontsize=13
    )
    fig.subplots_adjust(bottom=0.25, top=0.86, wspace=0.12)
    fig.savefig(out / "figure-02-human-frontier.pdf", bbox_inches="tight")
    fig.savefig(out / "figure-02-human-frontier.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2))
    ax = axes[0, 0]
    methods = ["markov", "context", "gru", "transformer", "parent_k256"]
    colors = {"known_users": "#276783", "unseen_users": "#b76b2b"}
    for protocol, offset in [("known_users", -0.12), ("unseen_users", 0.12)]:
        for i, method in enumerate(methods):
            values = [
                r["nll_equal_user_mean"]
                for r in data["frontier"]
                if r["protocol"] == protocol and r["method"] == method
            ]
            ax.scatter(
                np.full(len(values), i + offset),
                values,
                color=colors[protocol],
                s=24,
                alpha=0.65,
                label=protocol.replace("_", " ") if i == 0 else None,
            )
            ax.plot(
                [i + offset - 0.055, i + offset + 0.055],
                [np.mean(values)] * 2,
                color=colors[protocol],
                linewidth=1.5,
            )
    ax.set_xticks(
        range(5), ["Markov", "No history", "GRU", "Transformer", "256 states"], rotation=15
    )
    ax.set(title="A. Held-out one-step loss", ylabel="Equal-user NLL (nats)")
    ax.legend(frameon=False, fontsize=8)
    ax = axes[0, 1]
    calibration_colors = {"gru": "#276783", "transformer": "#b76b2b", "parent_k256": "#397d58"}
    ax.plot([0, 1], [0, 1], color="#aeb4bb", linestyle="--", linewidth=0.8)
    for method, color in calibration_colors.items():
        for seed in range(20261015, 20261020):
            cells = [
                c
                for c in data["calibration"]
                if c["protocol"] == "known_users"
                and c["method"] == method
                and c["seed"] == seed
                and c["predicted"] is not None
            ]
            ax.plot(
                [c["predicted"] for c in cells],
                [c["observed"] for c in cells],
                color=color,
                alpha=0.35,
                linewidth=0.9,
                marker="o",
                markersize=2,
                label=method.replace("_", " ") if seed == 20261015 else None,
            )
    ax.set(
        title="B. Known-user calibration",
        xlabel="Mean predicted probability",
        ylabel="Observed response rate",
        xlim=(0, 1),
        ylim=(0, 1),
    )
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 0]
    for protocol, color in colors.items():
        xs = []
        means = []
        lower = []
        upper = []
        for k in [16, 32, 64, 128, 256]:
            values = [
                r["mean_interval_width"]
                for r in data["uncertainty"]
                if r["protocol"] == protocol
                and r["latent_states"] == k
                and r["status"] == "evaluated"
            ]
            if len(values) != 5:
                raise ValueError("All five interval diagnostics required")
            xs.append(k)
            means.append(np.mean(values))
            lower.append(min(values))
            upper.append(max(values))
        ax.plot(xs, means, color=color, marker="o", linewidth=1, label=protocol.replace("_", " "))
        ax.fill_between(xs, lower, upper, color=color, alpha=0.12)
    ax.set(
        title="C. Pooled-model uncertainty precision",
        xlabel="Discrete latent states",
        ylabel="Mean interval width on test origins",
        ylim=(0, 1.04),
    )
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256], ["16", "32", "64", "128", "256"])
    ax.legend(frameon=False, fontsize=8)
    ax = axes[1, 1]
    for phase, offset, color, marker, label in [
        ("response_all_groups", -0.12, "#276783", "o", "All-group calibration"),
        ("response_heldout_groups", 0.12, "#b76b2b", "s", "Held-out calibration groups"),
    ]:
        for i, seed in enumerate(range(20261015, 20261020)):
            row = next(r for r in data["responses"] if r["phase"] == phase and r["seed"] == seed)
            y = i + offset
            ax.hlines(y, *row["ci95"], color=color, linewidth=1.2)
            ax.scatter(
                row["candidate_minus_reference_nats"],
                y,
                color=color,
                marker=marker,
                s=25,
                label=label if i == 0 else None,
            )
    ax.axvline(0, color="#7d848d", linewidth=0.8)
    ax.set_yticks(range(5), ["15", "16", "17", "18", "19"])
    ax.invert_yaxis()
    ax.set(
        title="D. Targeted versus matched refinement",
        xlabel="Candidate minus control NLL (nats)",
        ylabel="Fixed seed suffix",
    )
    ax.ticklabel_format(axis="x", style="sci", scilimits=(-3, 3))
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
    for ax in axes.flat:
        ax.grid(axis="y", color="#e5e8eb", linewidth=0.5)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Figure 6. Human predictive fidelity and limited condition transfer",
        x=0.07,
        ha="left",
        fontsize=13,
    )
    fig.text(
        0.07,
        0.015,
        "Shared test users across fitted seeds. Pointwise user-bootstrap intervals; no arbitrary-interface or zero-shot claim.",
        fontsize=8,
        color="#505761",
    )
    fig.tight_layout(rect=(0.02, 0.045, 1, 0.96))
    fig.savefig(out / "figure-06-human-fidelity.pdf", bbox_inches="tight")
    fig.savefig(out / "figure-06-human-fidelity.png", dpi=250, bbox_inches="tight")
    plt.close(fig)
    caption = """# Human figures

Figure 2 shows every fixed discrete candidate across five fits and both protocols. State counts and NLL differences use the same held-out cases within each fit. Shading identifies at-least-twofold compression relative to 256 states; the dotted horizontal line is the 0.01 nat loss margin. The dashed GRU reference averages fitted-seed point differences descriptively. Strict, finite-property and uncertified control/quantization families remain separate. Independent pooled count models are predictive comparators, not quotient certificates for the teacher-emission parent.

Figure 6 uses complete primary 24-hour prediction and both five-seed response matrices. Panel A plots every fitted seed and descriptive mean marks; seeds reuse human observations and are not independent population samples. Panel B plots 20 equal-user calibration bins per fit without confidence bands. Panel C shows equal-user mean interval widths assigned to test origins and the range across fixed fits, not confidence-coverage estimates. These visitation-weighted widths differ from the calibration report's equally weighted averages across state rows. Panel D uses paired original-user 95% bootstrap intervals. Held-out calibration groups were not withheld from base encoder training. Overlapping fold forecasts are scored as losses within original cases; no extra independent samples are created. All-group and held-out-condition studies remain distinct.
"""
    (out / "caption.md").write_text(caption)
    report = dict(
        status="complete",
        figures=[2, 6],
        render_digest=digest,
        analysis_freeze_digest=data["analysis_freeze_digest"],
        artifact_root=str(out),
        input_reports=data["inputs"],
        scope=data["scope"],
        files={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()
        },
    )
    (out / "manifest.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    import json
    import hashlib
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    path = ROOT / "reports/figures/human-final-tables.json"
    if not path.exists():
        raise ValueError("Build complete measured human tables before rendering")
    data = json.loads(path.read_text())
    digest = hashlib.sha256(
        Path(__file__).read_bytes() + json.dumps(data, sort_keys=True).encode()
    ).hexdigest()
    reservation = BudgetLedger(ROOT / ".campaign/nested-human-budget.sqlite", 40).reserve(
        "human-figures-" + digest, 0.03
    )
    state = ROOT / ".campaign" / ("human-figures-" + digest[:16] + ".json")
    if state.exists():
        call = modal.FunctionCall.from_id(json.loads(state.read_text())["function_call_id"])
    else:
        call = render.spawn(data)
        state.write_text(
            json.dumps(
                dict(
                    function_call_id=call.object_id, reservation=reservation, render_digest=digest
                ),
                indent=2,
            )
        )
    print(state.read_text(), flush=True)
    result = call.get()
    (ROOT / "reports/preflight" / ("human-figures-" + digest[:16] + ".json")).write_text(
        json.dumps(result, indent=2)
    )
    print(json.dumps(result), flush=True)
