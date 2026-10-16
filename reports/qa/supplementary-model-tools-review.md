# Supplementary outcomes, models, scores and intervals scoped review

Read supplementary_outcomes.py, supplementary_counts.py, supplementary_heads.py,
supplementary_metrics.py, supplementary_abstraction.py and
supplementary_intervals.py in full, with their six test modules. No root
CLAUDE.md exists; generic QA defaults apply. Whole-repository readiness unproven.

Corrected one defect: accepted uint64 state keys mixed with signed labels promoted
cell indices to float before bincount. Both bounded terms now use int64. Original
source is preserved, matches the cloud-tested archive, and exactly reconstructs
both frozen supplementary calibration/final source maps when combined with the
eight original library modules. Existing signed-state results are unchanged.

Outcome tools preserve invalid/missing responses as -1, use recorded watch-time
cutoffs and bind targets to authenticated physical source ordinals rather than
ambiguous event-key joins. Counts enforce the per-user cap and endpoint-specific
missingness. Smoothed emissions are point models. Categorical heads scale only
observed training cases, weight users equally and retain absent classes through
explicit pseudo-observations. They do not retrain encoders.

Scores floor/renormalize probabilities for proper losses while calibration uses
original predictions. User-disjoint merging rejects repeated users across buckets.
Four-class Brier sums category errors; binary categorical Brier is twice the
positive-class convention. One-outcome endpoint abstractions do not describe
subsequent trajectories or joint endpoint dependence. Uniform H1 envelopes use
exact rational interval-box averages and outward rounding. Feasible simplex
membership is required; coordinate-dependence relaxation is conservative. Human
statistical membership/transfer remains a separate claim.

Added 15 boundary tests for integer widths, invalid partitions, nondyadic outward
averaging, role sampling and categorical merge contracts. Focused supplementary
suite: 68 passed with warnings-as-errors, including fitting tests. Full local
suite: 1448 passed, three Torch-module skips and three retained horizon backend
warnings. Ruff passes. No new cloud jobs or original human outputs changed.

Evidence: reports/preflight/supplementary-integer-states-correction.json and
reports/preflight/supplementary-boundary-local-validation.json. Strict final
environment validation, remaining controller/tool QA and release acceptance open.
