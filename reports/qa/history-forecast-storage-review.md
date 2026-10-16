# History sampling and forecast storage scoped review

Read history_state.py, horizon_archive.py and pooled_forecasts.py in full,
with all three corresponding test modules. No source changes were necessary.

History sampling selects sorted physical-row positions from nonrandom exposure
IDs with a seeded generator; it does not use outcomes to choose the sample.
Initial predictive bins use stable ties and cover every supplied parent state.
Horizon storage reconstructs direct, iid-origin and finite-table forecasts with
shape, finite-probability and state-support checks. It rejects duplicate names
across storage classes. Private arrays remain the caller's cloud responsibility.
Pooled forecasts sum integer counts as Python objects, avoiding uint64 reduction
overflow, and form rational nominal emissions before float conversion. Missing
support yields the declared neutral 0.5 point value, without narrowing supplied
uncertainty intervals. These forecasts have the calibration-count estimand, not
the training-teacher estimand.

Validation: 11 existing tests passed against the installed corrected wheel with
warnings treated as errors and pytest source-path injection disabled. Additional
local probes verified uint64 counts whose combined total exceeds 2**64 and a
zero-row direct-forecast archive. No human data, provider inference, cloud jobs,
model changes or original scientific-output changes were involved.

This is a scoped review, not whole-repository ship-ready acceptance. Full source
QA, final candidate assembly and publication remain open.
