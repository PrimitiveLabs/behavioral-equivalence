# Library entry points, preparation and horizon heads scoped review

Read __init__.py, __main__.py, human_preparation.py and horizon_heads.py in full
with test_planner.py, test_human_preparation.py, test_horizon_heads.py and the
new test_entrypoint_horizon_boundaries.py. No library source changes. No root
CLAUDE.md exists; generic QA defaults apply. Whole-source acceptance remains open.

The planner checks duplicate IDs, missing dependencies, cycles and deliverable
mappings; ready tasks exclude running/completed tasks and require completed
parents. It cannot submit cloud work. Its configuration is a planning artifact,
not the current empirical-results source. The execution manifest is separate.
Preparation keeps tied timestamps in one bundle, retains maximum and summed
feedback-time proxies and separates user assignments from chronological splits.
Those proxies do not establish exact human response availability.

Horizon heads use fitted origin-feature scaling, inverse-user-frequency weights,
missing-target exclusion and explicit structural zeros. H1 reachability reuses
or complements the occupancy head with identical observed support. Single-class
fallback uses its declared smoothed constant. Predictions validate frozen-array
alignment, positive scales and finite parameters. No horizon head establishes a
joint trajectory distribution merely by predicting multiple marginal events.

Added 21 tests for planner/CLI gates, invalid frozen parameters, empty/extreme
predictions and inconsistent training labels/features/users. Pinned scikit-learn
1.7.1 and two small dependencies were installed into an ephemeral local target,
allowing four previously skipped fitting cases to run without cloud charges.
Full suite: 1415 passed, 3 Torch-module skips, 3 numerical-backend warnings.
Ruff passes. Warnings-as-errors fails on a local matrix multiplication diagnostic.
Bounded operands (maximum standardized feature 3.71, coefficient 4.81) yield finite
probabilities matching an independent unoptimized einsum within 2.23e-16.
No numerical source change or warning suppression was introduced. This does not
replace the remaining pinned-cloud strict validation requirement.

Evidence: reports/preflight/horizon-expanded-local-validation.json and
reports/preflight/horizon-local-numerical-diagnostic.json. Whole-source QA,
strict final environment validation and release acceptance remain open.
