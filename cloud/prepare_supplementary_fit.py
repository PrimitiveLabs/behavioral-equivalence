"""Pack outcome-blind supplementary train/validation samples on private Modal storage."""

from pathlib import Path
import modal

ROOT = Path(__file__).resolve().parents[1]
app = modal.App("behavioral-equivalence-supplementary-fit-cohort-v1")
volume = modal.Volume.from_name("behavioral-equivalence-research")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy==2.2.6")
    .env({"PYTHONPATH": "/research/tools", "OPENBLAS_NUM_THREADS": "2"})
    .add_local_file(
        ROOT / "tools/supplementary_outcomes.py",
        remote_path="/research/tools/supplementary_outcomes.py",
    )
    .add_local_file(
        ROOT / "configs/human-supplementary-heads-v1.json", remote_path="/research/config.json"
    )
)


def checksum(path):
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while b := f.read(8 * 1024**2):
            h.update(b)
    return h.hexdigest()


def source_identity():
    import hashlib
    import json

    base = Path("/research") if Path("/research/config.json").exists() else ROOT
    config = (
        base / "config.json"
        if base == Path("/research")
        else base / "configs/human-supplementary-heads-v1.json"
    )
    source = {
        "cloud.py": Path(__file__).read_text(),
        "config.json": config.read_text(),
        "tools/supplementary_outcomes.py": (base / "tools/supplementary_outcomes.py").read_text(),
    }
    return source, hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest()


@app.function(
    image=image,
    cpu=(2, 2),
    memory=(8192, 8192),
    timeout=900,
    retries=0,
    max_containers=1,
    volumes={"/artifacts": volume},
)
def prepare():
    import hashlib
    import json
    import time
    import numpy as np
    from supplementary_outcomes import fixed_role_sample, categorical_targets

    volume.reload()
    started = time.monotonic()
    source, digest = source_identity()
    cfg = json.loads(source["config.json"])
    targets = Path("/artifacts") / cfg["target_attachment_root"]
    cases_root = Path("/artifacts") / cfg["case_root"]
    context_root = Path("/artifacts") / cfg["context_root"]
    tr = json.loads((targets / "report.json").read_text())
    if (
        hashlib.sha256(json.dumps(tr, sort_keys=True).encode()).hexdigest()
        != cfg["target_attachment_report_canonical_digest"]
    ):
        raise ValueError("Target attachment report changed")
    if (
        checksum(cases_root / "report.json") != cfg["case_report_sha256"]
        or checksum(context_root / "report.json") != cfg["context_report_sha256"]
    ):
        raise ValueError("Case/context manifest changed")
    cr = json.loads((cases_root / "report.json").read_text())
    ir = json.loads((context_root / "report.json").read_text())
    if any(r["status"] != "complete" or len(r["buckets"]) != 32 for r in [tr, cr, ir]):
        raise ValueError("All32 complete input buckets required")
    groups = {(protocol, role): [] for protocol in cfg["protocols"] for role in [0, 1]}
    for b in range(32):
        path = targets / f"bucket-{b:02d}.npz"
        if checksum(path) != tr["buckets"][b]["output_sha256"]:
            raise ValueError("Target arrays changed")
        with np.load(path, allow_pickle=False) as a:
            for protocol, role in groups:
                mask = a["role_code"] == role
                if protocol == "unseen_users":
                    mask &= a["user_role_code"] == role
                groups[(protocol, role)].append(int(mask.sum()))
    samples = {
        key: fixed_role_sample(
            np.array(counts),
            cfg["sampling"]["maximum_train_cases" if key[1] == 0 else "maximum_validation_cases"],
            cfg["sampling"]["seed"],
        )
        for key, counts in groups.items()
    }
    pieces = {key: [] for key in groups}
    observed_ids = {key: [] for key in groups}
    for b in range(32):
        name = "embargo-24h.npz"
        cp = cases_root / f"bucket-{b:02d}" / name
        ip = context_root / f"bucket-{b:02d}" / name
        if (
            checksum(cp) != cr["buckets"][b]["output_sha256"][name]
            or checksum(ip) != ir["buckets"][b]["output_sha256"][name]
        ):
            raise ValueError("Source cases or context changed")
        with (
            np.load(targets / f"bucket-{b:02d}.npz", allow_pickle=False) as t,
            np.load(cp, allow_pickle=False) as c,
            np.load(ip, allow_pickle=False) as intrinsic,
        ):
            ids = t["case_id"]
            if not np.array_equal(ids, c["case_id"]) or not np.array_equal(
                ids, intrinsic["case_id"]
            ):
                raise ValueError("Complete source identity/order differs")
            for key in groups:
                protocol, role = key
                mask = t["role_code"] == role
                if protocol == "unseen_users":
                    mask &= t["user_role_code"] == role
                eligible = np.flatnonzero(mask)
                ix = eligible[samples[key][b]]
                selected = {
                    name: t[name][ix]
                    for name in [
                        "is_like",
                        "is_follow",
                        "is_comment",
                        "is_forward",
                        "is_hate",
                        *[f"watch_category_{k}" for k in range(4)],
                    ]
                }
                pieces[key].append(
                    dict(
                        case_id=ids[ix],
                        user_id=t["user_id"][ix],
                        tokens=c["tokens"][ix],
                        tab=c["tab"][ix],
                        video_id=c["video_id"][ix],
                        intrinsic_context=intrinsic["context"][ix],
                        targets=categorical_targets(selected),
                    )
                )
                observed_ids[key].append(ids[eligible])
    root = Path("/artifacts") / ("supplementary-fit-cohort-" + digest[:16])
    root.mkdir(exist_ok=True)
    records = []
    for key, parts in pieces.items():
        protocol, role = key
        arrays = {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}
        # Reconstruct global positions independently of the bucket slicing helper.
        full = np.concatenate(observed_ids[key])
        maximum = cfg["sampling"][
            "maximum_train_cases" if role == 0 else "maximum_validation_cases"
        ]
        chosen = np.sort(
            np.random.default_rng(cfg["sampling"]["seed"]).choice(
                len(full), min(maximum, len(full)), replace=False
            )
        )
        if not np.array_equal(arrays["case_id"], full[chosen]):
            raise ValueError("Independent global sample replay disagrees")
        path = root / f"{protocol}-{'train' if role == 0 else 'validation'}.npz"
        np.savez_compressed(path, **arrays)
        records.append(
            dict(
                protocol=protocol,
                role="train" if role == 0 else "validation",
                eligible_cases=len(full),
                sampled_cases=len(chosen),
                users=len(np.unique(arrays["user_id"])),
                path=path.name,
                sha256=checksum(path),
                case_id_sha256=hashlib.sha256(
                    arrays["case_id"].astype("<u8").tobytes()
                ).hexdigest(),
            )
        )
    report = dict(
        status="complete",
        source_digest=digest,
        root=str(root),
        records=records,
        wall_seconds=time.monotonic() - started,
        scope="Outcome-blind train/validation cohort only; no supplementary test sampling or scores. Exact global ordinal sample independently replayed; shared frozen input identity.",
    )
    (root / "source.json").write_text(json.dumps(source, indent=2))
    (root / "report.json").write_text(json.dumps(report, indent=2))
    volume.commit()
    return report


@app.local_entrypoint()
def main():
    import json
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from behavioral_equivalence.budget import BudgetLedger

    source, digest = source_identity()
    state = ROOT / ".campaign" / ("supplementary-fit-cohort-" + digest[:16] + ".json")
    if state.exists():
        saved = json.loads(state.read_text())
        call = modal.FunctionCall.from_id(saved["function_call_id"])
    else:
        print(
            json.dumps(
                BudgetLedger(ROOT / ".campaign/human-case-audit-budget.sqlite", 4).reserve(
                    "supplementary-fit-cohort-" + digest, 0.10
                )
            ),
            flush=True,
        )
        with state.with_suffix(".submission-attempt").open("x") as marker:
            marker.write(digest + "\n")
        call = prepare.spawn()
        saved = dict(source_digest=digest, function_call_id=call.object_id, reservation_usd=0.10)
        state.write_text(json.dumps(saved, indent=2))
    print(json.dumps(saved), flush=True)
    (
        ROOT / "reports/source-snapshots" / ("supplementary-fit-cohort-" + digest[:16] + ".json")
    ).write_text(json.dumps(source, indent=2) + "\n")
    report = call.get()
    (ROOT / "reports/preflight" / ("supplementary-fit-cohort-" + digest[:16] + ".json")).write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report), flush=True)
