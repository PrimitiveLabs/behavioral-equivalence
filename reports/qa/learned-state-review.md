# Learned-state estimation and refinement QA, scoped review

Read latent_estimation.py, learned_abstraction.py, quantization.py, parent_identity.py and property_refinement.py in full with their corresponding tests. No repository CLAUDE.md exists.

Adjacent counts require consecutive source bundles from one user and condition successors on the current outcome. Positive nominal smoothing fills unsupported rows but is not a confidence guarantee. Initial mass is accumulated with equal-user weights and returned separately from user count. Scaling and teacher emissions use supplied training-only arrays; unsupported emissions have an explicit global fallback. Strict latent refinement checks the actual lifted rational radius. Finite-property refinement uses floating screening followed by exact numerical acceptance and only certifies listed pending-state events. Parent cache identity binds all upstream selectors, while the saved original estimator function remains byte-structurally bound by an existing regression test.

Existing tests cover source-pair adjacency, equal-user weighting, empty initial support, identical and distinct emissions, singleton-origin roundoff fallback, stable ties, unsupported emission fallback, all cache selectors, and a future-property counterexample. Missing direct boundaries before remediation include count/index/prior/representation validation, incompatible parent layouts, refinement tolerance/horizons and prediction maps. A small exact example also needs to assert that floating equality does not bypass numerical acceptance, and that singleton witnesses can force successor partition refinement. No inspected evidence upgrades the failed strict human compression result or establishes human Markov sufficiency.

Verdict before remediation: needs-more-tests-before-merge for these explicit gaps, not whole-release approval.


## Remediation and actual implementation finding

`tests/test_learned_state_boundaries.py` adds 65 cases for the listed estimation/quantization/refinement boundaries. It exposed a real integer-fallback defect: passing 0 or 1 to the teacher-emission fallback created an integer output buffer and NumPy division raised UFuncOutputCastingError (two failed tests, 62 passed before correction). The corrected call explicitly supplies float64. An AST comparison proves this is the only semantic edit in quantization.py. The original source is retained as a hash-bound project-source fixture and matches the cloud-tested archive and four frozen human source maps.

Both recorded training callers supply Python floats: quantize_calendar accumulates a Python-float teacher total and divides by n; fit_history_states explicitly casts teacher.mean(). Sixty deterministic randomized fixtures confirm exact output equality for original float64 fallback behavior. This establishes compatibility of the corrected operation for those caller types; it is not a rerun of human inference. Integer endpoints now succeed.

The new tests also cover a genuine numerical distinction: at H3, floating screening reports zero error but the exact certificate requires refinement; at H5, a singleton witness forces refinement of a nontrivial successor block. Final numeric acceptance and certificate replay pass. Cache selector and frozen estimator checks remain in the existing suite.

The locally built corrected wheel passes the entire available suite: 1018 tests, eight optional-neural/data skips, source-path injection disabled. `reports/preflight/quantization-dtype-fixed-wheel-validation.json` binds that distinct wheel, correction, tests and output. Ruff passes. The historical zero-skips cloud result does not automatically apply to the corrected library. Frozen controllers require the preserved original module in an isolated tree, as documented in docs/reproduction.md.

Scoped verdict: the identified five-module gaps are closed, including the dtype correction. Whole-repository ship-ready status remains unassessed. Historical scientific artifacts and success/failure decisions are unchanged.
