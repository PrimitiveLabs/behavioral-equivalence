# Behavioral equivalence under uncertainty

Primitive Labs research workspace for all nine deliverables and four experiments: mathematical specification, synthetic benchmarks, learned abstractions, probabilistic verification, human prediction, intervention refinement and a reproducible research release.

The complete 1,200-world synthetic benchmark, repeated transition-uncertainty study and 400-world intervention study are finished. All 30 human neural fits and ten state-abstraction frontiers are complete. The locked 50-job final human matrix and independent aggregate-output audit are complete. The separately frozen ten-fit joint trajectory study is scored and independently replayed. Research completion is tracked in the separate execution and release evidence manifests.

Start with the [goal architecture](docs/goal-architecture.md), [dependency manifest](configs/program.json), [execution state](reports/execution-state.json) and [deliverable evidence index](reports/deliverable-evidence.json). Per-campaign reservations are internal controls, not actual bills, a user-approved global budget or provider-enforced spending limits. No external language-model inference is used.

## What the completed studies establish

- [Synthetic preservation](reports/synthetic-main-results.md): 26,400 abstraction records, 422,400 property comparisons and 55,200 independent PRISM point checks, with exact-normalized numerical auditing and retained serialization caveats.
- [Repeated uncertainty](reports/uncertainty-repeated-results.md) and [nominal ablation](reports/nominal-ablation-results.md): robust outcome enclosures cover the tested generating probabilities, often at poor precision. The narrow point control frequently misses. Kernel membership and event-probability containment remain separate metrics.
- [Synthetic interventions](reports/intervention-results.md): targeted refinement helps the fixed comparison, while the tested RBF search heuristic loses to random/grid search with refinement held fixed. The [diversity extension](reports/environment-diversity-results.md) reuses existing worlds and is exploratory.
- [Human compression](reports/human-abstraction-results.md): every fixed primary retains 256/256 states and fails the twofold compression target. Secondary variants cannot replace it.
- [Human uncertainty construction](reports/human-cluster-uncertainty-results.md): all 256-state emission intervals are [0,1]. These conditional count-model intervals do not establish human Markovity or predictive coverage.

The [paper](reports/paper-draft.md) and [blog](reports/blog-draft.md) are unpublished interim drafts. All six measured figures are rendered and visually reviewed. The primary, trajectory and supplementary endpoint output audits are complete. The accompanying release manifest records reproduction and packaging status.

## Local checks

```sh
uv sync --group dev
uv run pytest -q -W error
uv run ruff check src tests cloud tools
uv run python -m behavioral_equivalence plan
```

Optional Torch, scikit-learn, Polars and DuckDB checks run in pinned cloud environments. The planner validates dependencies and lists eligible work; it does not submit jobs. Full campaigns use the explicit source-bound controllers described in [reproduction and recovery](docs/reproduction.md); the planner is not a generic cloud DAG submitter.

## Final human execution

The [joint analysis lock](reports/final-analysis-freeze.json) binds source snapshots, models, cases, statistics and ablations before final scoring. This is an internal lock, not external preregistration. The [job plan](reports/final-human-job-plan.json) retains all seeds, baselines and feedback-availability sensitivities. Using the configured Modal environment:

```sh
python tools/run_final_human_jobs.py --workers 2
python tools/summarize_human_primary.py
```

The runner attaches to saved Modal call handles and retains failures for inspection. Do not restart an input solely because observation timed out. Completed aggregate reports are saved under `reports/final-human`; raw interactions, weights and individual forecasts remain private on Modal. The paired-user primary decision requires both twofold compression and an upper 95% bound below 0.01 additional nats for every fixed fit.

`cloud/audit_final_human.py` checks saved output hashes and replays aggregate proper losses and primary user-bootstrap intervals. Its scope excludes neural-forward recomputation and secondary subgroup/calibration replay. `tools/build_human_figure_tables.py` requires complete primary and response matrices before `cloud/figure_human.py` renders measured human figures.

## Mathematics and reproducibility

[Analytical proofs](spec/proofs.md) distinguish observable trace events, finite-property certificates, conditional events and robust uncertainty. [Lean sources](formal/BehavioralBounds.lean) mechanically check recurrence-to-event error and interval expansion. [Finite maximal coupling](formal/FiniteCoupling.lean) and [weighted observable events](formal/WeightedEvents.lean) also have pinned Lean checks. Partition lifting and complete path-law integration remain analytical, while statistical assumptions and the numerical implementation are not one end-to-end mechanization. External peer review remains unperformed.

Build a deterministic source-only archive with:

```sh
python tools/package_source.py
```

The archive includes source, tests, formal proofs, frozen configurations, static rights records and two explicit test fixtures. It excludes human interaction rows, weights, individual predictions, secrets, caches and runtime ledgers. Scientific results and dependency/toolchain environments are separate artifacts. A source archive is not a public release or independent scientific reproduction.

[Rights records](data/license-register.json) govern dataset use and redistribution. Logged randomized recommendation exposures support restricted response analyses, not arbitrary interface counterfactuals. Hash-verified private R2 storage is content addressed, not server-enforced WORM. Respect these boundaries when interpreting or publishing results.

Project code is MIT licensed; see LICENSE and NOTICE. Separately obtained datasets and third-party dependencies retain their own terms. See the data and model cards in docs/.
