# Result audits, human figure tables and state sufficiency scoped review

Read final_result_audit.py, supplementary_result_audit.py,
build_human_figure_tables.py and state_sufficiency.py in full with their four
existing test modules and new test_result_audit_boundaries.py. Generic QA defaults
apply; no root CLAUDE.md exists. Whole-repository readiness remains unproven.

Two defects corrected. State-sufficiency grouping previously promoted mixed
signed/uint64 cell-user keys through float, merging distinct large identities.
Structured state/category/user fields now preserve exact integer identities,
including states above signed 64-bit range. Recorded signed-domain results are
identical in a seeded compatibility test. Calibration replay now rejects nonfinite
reported/recomputed values before tolerance comparison; NaN could previously
pass the comparison. Both original sources match the historical cloud archive
and corresponding retained frozen source maps. No original results were edited.

Binary result audits average replica losses before user aggregation, preserving
the actual replicated-forecast estimand. Sorted-user arithmetic is distinct from
production bincount grouping. Categorical replay separately rebuilds proper
losses and per-category/top-label calibration. Bootstrap resamples original users,
not cases or replica copies. Shared arithmetic/runtime primitives limit claims
of software independence. User-disjoint buckets remain a caller contract where
not explicitly checked. State/category heterogeneity remains descriptive, with
unsupported cell rates withheld; it is not a proof of Markov sufficiency.

Figure tables require the full locked ten-fit primary prediction and ten response
report matrices, verify analysis/source/partition identities, preserve statistical
bootstrap seeds separately from fit seeds, and retain state counts and paired
intervals. Regenerated the actual retained table into an isolated temporary tree:
485 frontier rows, 1200 calibration cells and ten response comparisons match
original bytes exactly. Original table was not overwritten.

Added 23 tests: exact integer identities, nonfinite calibration values in both
groups/all three numeric fields, signed compatibility and empty/missing-support
mismatches. Focused suite: 38 passed with warnings-as-errors. Full suite: 1471
passed, three Torch-module skips and three retained horizon backend warnings.
Ruff passes. Evidence: reports/preflight/result-audit-corrections.json,
result-audit-local-validation.json and human-figure-table-replay.json.

No provider jobs, human inference or new empirical claims. Remaining source QA,
strict final environment validation and final release acceptance remain open.
