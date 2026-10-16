# Intervention and search QA, scoped review

Read refinement.py, intervention_space.py, catalogue_search.py and intervention_benchmark.py in full, with test_refinement.py and test_intervention_space.py. No repository CLAUDE.md exists.

The scalar witness search compares merged-state pairs only and rechecks discovered environments separately. Catalogue search evaluates exactly the selected query indices; the RBF surrogate uses observed scores only. Interpolation receives training plus queried kernels, while held-out generating kernels enter scoring separately. The benchmark distinguishes estimated-environment error, abstraction error and total error. Its certificates bound estimated full versus reduced kernels, not prediction of unseen generating environments. Random-capacity splitting matches targeted count within the original partition. Negative controls preserve block-level behavior numerically; they are not a human intervention result.

Existing tests cover all four search methods with identical query budgets, deterministic replay, disjoint search/evaluation catalogues, matching vectorized/scalar JS, neutral and changed environments across all five families, structural negative controls and shared interpolation. Remaining assertions before remediation: invalid JS/distribution/partition boundaries, unsupported witness/catalogue/point/budget settings, evaluator invalid scores and exceptions, interpolation labels/initial laws/bandwidth, exact refinement capacity endpoints, and a complete small benchmark checking retained observed data and decomposed controls. Existing measured 400-world results remain independent evidence.

Verdict before remediation: needs-more-tests-before-merge for the listed boundaries. No inspected evidence supports upgrading the negative active-search or human generalization findings.


## Remediation and scoped closure

`tests/test_intervention_boundaries.py` adds 46 assertions for all listed gates, plus two full 16-state intervention benchmarks (linear and structural negative control, two queries per method, H1). These retain all four search methods and six refinement controls. Assertions check 384 score rows and 64 certificate bindings per world, queried observation identity, exact matched capacity, the triangle error decomposition, identity-model isolation and numerical negative-control invariance. Errors from evaluator callbacks propagate and invalid catalogues make no query. These are small implementation regressions, not replacements for the original 400-world experiment or evidence about humans.

All 46 also pass against the exact cloud-tested installed wheel, with source-path injection disabled and matching numerical dependency versions. `reports/preflight/intervention-boundaries-installed-wheel.json` binds their bytes and output. No implementation or frozen scientific source changed. Scoped verdict: no identified gap remains in the four reviewed modules. Whole-repository ship-ready status remains unassessed.
