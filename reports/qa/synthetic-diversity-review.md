# Synthetic generator and diversity QA, scoped review

Read synthetic.py, environment_diversity.py and diversity_analysis.py in full, including the complete aggregate/replay implementation. Inspected the relevant generator/mechanism tests in test_core.py and complete environment-diversity and diversity-analysis tests. No repository CLAUDE.md exists.

The five families use different transition constructions, with quantized Bayesian posterior updates and dyadic unperturbed masses. The four-copy partition preserves block transition probabilities before perturbation. Seeds vary mechanism parameters as well as internal wiring. Perturbation and intervention RNGs are separate. Terminal labels remain closed. Family names describe generating assumptions, not established human mechanisms.

Diversity keeps eight nonneutral queries for each of three frozen candidate spans and the same held-out catalogue. Estimated-kernel certificates do not bound environment-learning error. Aggregate analysis requires the complete world/seed/surface matrix, source identity, all 48 replay bindings, exact capacity controls and invariant negative controls. Seed-cluster intervals are explicitly descriptive and exploratory. Existing tests cover all family reductions, perturbation stability, a 512-state endpoint, actual Bayesian updates, seed variation, negative controls, persisted replay/tampering and major aggregate failures.

Before remediation, direct assertions remained for invalid family/size/perturbation and intervention endpoints, the minimal and odd base-size mechanism cases, geometry guard, individual per-world summary constraints and remaining aggregate capacity/cluster gates. These are regression checks, not another full matrix or human fidelity evidence.

Verdict before remediation: needs-more-tests-before-merge for the listed gates, not whole-release approval.


## Remediation and scoped closure

`tests/test_synthetic_diversity_boundaries.py` adds 39 cases covering all listed boundaries. Minimal 16-state and 20-state worlds exercise every mechanism, exact known partitions, deterministic seed reconstruction and maximal perturbation while retaining terminal rows. Intervention endpoints preserve the source and expose duplicate identity. A real small invariant diversity report exercises summary rejection paths; the frozen geometry guard rejects a changed candidate pool. Aggregate tests retain complete seed/surface designs and reject unsupported capacity, missing means and absent changing worlds.

No generator or diversity implementation changed. Tests retain conditional/exploratory scope rather than upgrading these checks to a new 400-world study or evidence about humans. Scoped verdict: no identified gap remains in the three reviewed modules. Installed-wheel validation is recorded in reports/preflight/synthetic-diversity-corrected-wheel-validation.json; whole-source QA remains open.
