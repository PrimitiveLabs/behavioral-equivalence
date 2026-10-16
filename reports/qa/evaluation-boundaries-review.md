# Evaluation and split QA, scoped review

Read in full: data.py, calendar_splits.py, statistics.py, streaming_scores.py and their data/calendar/statistics/prediction/streaming test files. Also inspected confirmatory preparation tests. No repository CLAUDE.md exists. This is a scoped review, not whole-release approval.

The calendar suite checks cutoff equality, availability embargoes, unseen-user identity/time intersection, scalar/batch equivalence and int64 overflow. Data tests reject future/cross-user history, duplicate case IDs, ambiguous timestamp ties and disjoint-user violations. Statistical tests distinguish equal-user from equal-case aggregation; streamed metrics agree with materialized metrics, reject split users and retain original-case counts for replicated forecasts. Shared bootstrap contrasts agree with explicit resampling and preserve paired signs. Proper losses are averaged across replicas before users, not computed from averaged probabilities. Probability clipping is declared explicitly. None of these checks establishes a human Markov assumption, calendar independence or causal intervention identification.

Before merge, add direct regression assertions for remaining constructor and protocol boundaries: invalid binary identities and histories, short/unsupported split cases, scalar calendar settings, scoring alignment/targets/probabilities, paired-bootstrap settings and minimum cluster support, and streaming/bootstrap identifier/count/model/contrast boundaries. Existing successful fixtures do not assert these rejection paths. No inspected evidence contradicts the completed scientific result matrices. Legacy within-user fractional split helpers are not substituted for the confirmatory shared-calendar pipeline.

Verdict before remediation: needs-more-tests-before-merge for the listed evaluation boundaries.

## Remediation and evidence

`tests/test_evaluation_boundaries.py` adds 78 direct boundary assertions covering the gaps listed above. All pass locally and against the exact installed wheel from the 626-test cloud run, with source-path injection disabled. `reports/preflight/evaluation-boundaries-installed-wheel.json` binds exact test bytes, wheel hash, matching dependency versions and output. The full local suite passes 748 tests with 11 optional-dependency skips; Ruff passes. No implementation or frozen controller source changed.

The four-module boundary slice has no remaining identified gap from this review. This is scoped closure, not a whole-repository ship-ready verdict.

## Additional modules inspected, separate pending scope

Also read `confirmatory_preparation.py`, `calendar_support.py`, `tests/test_confirmatory_preparation.py` and `tests/test_calendar_support.py` in full. Existing preparation assertions demonstrate outcome-blind selection, causal histories, per-user caps, duplicate IDs and missing mappings. Support diagnostics agree with independent user/timestamp enumeration and preserve excluded counts. Remaining direct assertions are needed for invalid/missing role caps and duplicate timestamp bundles in preparation, and shape/type/noncausal availability/negative embargo branches in support diagnostics. Optional Polars preparation tests require the pinned data environment; do not claim this local NumPy-only amendment executed them.


Preparation/support remediation: `tests/test_history_support_boundaries.py` adds 27 assertions for input shape/type, embargo, sortedness, duplicate times, exposure support, empty cohorts, and feature history timing/count/cap boundaries. `tests/test_preparation_boundaries.py` adds nine real-Polars cases for missing/extra/invalid caps, duplicate timestamp bundles and empty selected events. `features.py` was read in full with `tests/test_features.py`; its delayed-feedback and strict-boundary semantics remain unchanged. All 36 added preparation/support/feature tests pass against the exact installed cloud-tested wheel with pinned Polars 1.31.0, on small synthetic tables. No raw human data, new provider compute or implementation changes are involved. This closes the listed preparation/support gaps; a skipped local module is not used as evidence of their execution. The source-bound supplemental report is `reports/preflight/preparation-support-installed-wheel.json`.
