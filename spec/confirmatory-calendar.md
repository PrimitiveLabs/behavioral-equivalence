# Prospective human evaluation calendar

Status: candidate design, not a final confirmatory freeze. No final 27K predictive scores have been inspected.

The earlier 1K engineering experiments use per-user percentage splits and user-held-out splits over the recorded period. Per-user chronological ordering does not establish that every training event from every other user precedes an evaluated prediction. Treat those fits as exploratory retrospective benchmarks. Do not describe them as a prospective deployment backtest.

The confirmatory candidate uses shared UTC cutoffs based only on timestamps:

| Role | Exposure timestamps |
| --- | --- |
| Training | Before 2022-04-29 00:00 UTC |
| Validation | 2022-04-29 through 2022-05-01 UTC |
| Calibration | 2022-05-02 through 2022-05-04 UTC |
| Test | From 2022-05-05 00:00 UTC to the recorded period end |

These are epoch-based UTC instants, not an assumed interpretation of the source's local `date` or `hourmin` fields. The dataset preparation's percentage-split columns remain diagnostic metadata. Confirmatory case selection applies this calendar separately and records its own immutable manifest.

For known users, apply the calendar to all eligible nonpilot users. For unseen users, require both the outcome-independent user role and the calendar role to agree. Training-user events after the training cutoff cannot train the model. Earlier held-out-user history can supply observed prefix context at prediction time but cannot update fitted model weights, normalizers, clusters or hyperparameters.

For training, validation and calibration cases, require the declared outcome-availability proxy plus the selected embargo to lie at or before that role's end. This prevents using feedback that becomes available only after the corresponding fitting or selection cutoff. Test outcomes can arrive after the prediction because they serve as evaluation targets, not inputs. Every history token still requires its own strictly earlier, available observation time.

Candidate primary timing policy: summed nonnegative feedback-duration proxy per timestamp bundle plus a 24-hour embargo. Sensitivity policies use 0 and 48 hours. These remain assumptions about logged feedback availability. The official release does not supply complete observation-time provenance, so the final paper must not claim unconditional online observability. Before the final freeze, check case/user support and training/validation diagnostics for these policies; preserve the same declared timing rules for all compared methods.

The final freeze must record the selected calendar, embargo, supported user population, sampling rules, fitted architectures, seed list, error tolerances and noninferiority margin before final test scoring. The 0.01-nat prediction margin and 2x compression target remain unchanged. Report any failure of those criteria.

## Constructed and independently audited cases

The frozen case-construction rules are in `configs/human-case-construction-v1.json`. Every user contributes at most 256 training, 64 validation, 128 calibration and 128 test cases, sampled without replacement by seeded hash priority of frozen physical source row IDs. Unseen-user cases are the identical case IDs restricted to matching user/calendar roles. Target values enter the selected case artifact only after selection. Per-user prediction means therefore remain the primary estimand; case-weighted metrics are secondary. Training uses the capped case sample, and its exact loss weighting is separately recorded.

Source digest `d05a64f50a264877b4acd1e3cf61dfa80738fb667cb4f469330fc70c8e2e5585` constructs all three embargo policies. Independent audit digest `69ea08c4b9ccfe9cc60ef1e2fa99496c6111851abf6bdd6cf27ca483fb03a965` passes on every case and history token: 14,690,640 cases at zero hours, 14,534,926 at 24 hours and 14,064,461 at 48 hours. Audit reconstruction uses PyArrow physical rows, separate NumPy calendar/selection logic and a bounded heap of available prior bundles. Only the protocol's pinned Polars hash primitive is shared with selection. These counts overlap across timing policies and are not independent samples.

Fit-cutoff eligibility depends on the declared maturity proxy and can select against long-duration feedback near a boundary. Report that truncation rather than assuming a random maturity pattern. Final test cases are not filtered by their own outcome duration. Full-period release-cohort selection remains a separate generalizability limitation, discussed in the intervention contract.

Training/validation model artifacts will be staged separately from calibration/test artifacts. The locked training configuration uses three seeds and two continuous sequence architectures, with a current-item vocabulary fitted solely on training exposures. It does not finalize the abstraction operating-point or statistical analysis preregistration. Those must be locked before final test scoring.
