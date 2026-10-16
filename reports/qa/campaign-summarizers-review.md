# Full synthetic campaign summarizers scoped review

Read summarize_nominal.py, summarize_repeated.py, summarize_synthetic.py,
summarize_uncertainty.py and summarize_interventions.py in full. No source edits.
Generic QA defaults apply; no root CLAUDE.md exists. Whole-source readiness open.

Nominal summary binds the original manifest and paired ablation, checks complete
4000-dataset/8000-support-evaluation design, and retains analysis/source hashes.
Repeated summary structurally validates all paired records before replaying every
retained chunk. Cached replays bind report and replay-source identities. Complete
summaries compare each manifest chunk with its retained report and verified counts.
Successful partial replay sidecars persist after a failed aggregate job; a failure
is not an instruction to rerun calibration or replace a live handle.

Synthetic summary requires the complete 1200-world matrix and 26400 integer
certificates, per-world source/report/audit hashes, matching methods/properties
and the specified PRISM state-zero checks. It retains tiny requested-budget
exceedances and serialized nominal-exact nonzero radii. Floating display values
and empirical runtime/RSS summaries do not replace integer certificates or
isolated hardware performance measurements. Coordinator recovery is explicit.

Uncertainty summary preserves paired datasets, widths/vacuity, kernel membership
and event-enclosure counts as different quantities. It reports floating CP-tail,
rectangular-semantic and external-checker limitations. Its cached world sidecars
are source-namespaced, not a fresh arithmetic replay on each cache hit. It assumes
the retained campaign tree remains fixed; mutable source artifacts would require
sidecar invalidation/rehashing before reuse. Intervention summary binds all
retained artifact/model/certificate identities and reconstructs detailed means,
then uses seed-cluster analysis. That structural audit does not independently
replay integer bounds or certify environment-model error/human transfer.

Legacy direct summary entrypoints lack some later launcher-lock/attempt guards;
no launch performed. Reviewed protocol loaders require exact protocol, source
capsule and matching arithmetic. Eighteen new function-body tests exercise those
bindings and invalid SHA identities without Modal imports/submission. Full local
suite: 1530 passed, three Torch skips, three retained horizon warnings. Ruff passes.

Separately audited all 37 retained source capsules: 2125 embedded Python entries
parse successfully, all 11 original correction fixtures hash-verify, and current,
fixture-preserved and older embedded versions remain explicitly classified.
This is availability/identity evidence, not experimental recomputation.
Evidence: reports/preflight/source-capsule-availability-review.json and
reports/preflight/campaign-summary-local-validation.json. Remaining controller
QA, strict final environment validation and final release acceptance remain open.
