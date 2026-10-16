# Human cohort construction controller review

Read prepare_adjacent_calibration.py, prepare_randomized_cases.py,
prepare_multistep_test.py and prepare_multistep_fit.py in full. This is a scoped
source and retained aggregate evidence review, not whole-release acceptance.
No implementation or test source changed, and no cloud entrypoint was invoked.

All four controllers authenticate the prepared report, independent audit, physical
bucket parquet files and intrinsic metadata before construction. Physical case IDs
are assigned before filtering and preserve source-row identity. Bundle joins are
many-to-one and fail on unmatched exposures. Cached private NPZ files are rehashed.
Source snapshots name the relevant helpers and configuration; historical snapshots
must remain distinct from corrected current helper versions.

Adjacent calibration uses outcome-independent representatives within complete
same-user timestamp bundles, actual consecutive eligible calibration bundles and
complete feedback-available histories. Queries include both members of each pair;
calibration-role assertions precede output. Sampled cases are not mistaken for
adjacent original interactions. This estimates a point kernel, not a confidence
certificate for human behavior.

Randomized preparation retains every physical randomized exposure, including
concurrent exposures, and preserves unavailable-role rows for audit. Its 75 codes
combine recorded scenario with four duration bins plus an unknown-duration bin.
The randomization flag selects the cohort and is not a fitted propensity or
predictor. Recorded exposures alone do not identify arbitrary product changes or
remove all causal identification assumptions.

Multistep test preparation chooses outcome-blind test origins before evaluating
future support. Horizons include the origin and require consecutive original
bundles belonging to the same user and role. Observed masks and censored counts
remain explicit. Event labels are private eventual feedback, not predictor inputs.
Training preparation restricts origins and future support to actual training or
validation calendar roles, records separate unseen-user roles, and produces no
final test predictions. Fitting data and test data are separate constructions.
These conditional observed-horizon estimands do not establish unconditional
survival, intentional abandonment or human Markov sufficiency.

The retained report reconciliation checks 128 bucket records across four full
preparations, 2464 aggregate support/environment/censoring checks and matching
physical input hashes for all 32 buckets. Evidence is retained in
reports/preflight/cohort-construction-controller-reconciliation.json. This does
not rehash private NPZ bytes live or independently reconstruct individual rows;
separate retained physical audits cover those earlier executions.

Legacy launchers reserve finite resource ceilings and save function handles but
lack later lifetime-attempt/concurrency controls. They were not relaunched. No
provider transport, concurrent writer or fault-recovery behavior was exercised by
this review. No test rerun was needed for report-only changes. The last full local
suite remains 1627 passed, three Torch collection skips and three retained
numerical warnings; strict final environment validation remains open.
