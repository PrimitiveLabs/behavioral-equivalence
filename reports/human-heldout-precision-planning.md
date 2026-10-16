# Held-out response precision and eligibility

Status: completed validation-only planning and outcome-free future eligibility. Final human test predictions and metrics remain ungenerated.

All five fixed fits use 375,340 known-duration future cases from 23,284 users, with ordered case SHA256 `4831e001139d26eaffeb406dab8dbae560866981a65d40a8c10bde084ba860f6`. Each case occurs in three held-out folds. The aggregate averages paired losses across those three forecasts per case and then within user. Fold copies do not create additional independent observations.

| Seed | Validation user-loss SD | Exact held-out forecast identity |
|---|---:|---|
| 20261015 | 0.002579 | False |
| 20261016 | 0.001788 | False |
| 20261017 | 0.004162 | False |
| 20261018 | 0.004415 | False |
| 20261019 | 0.000000 | True |

Eighteen of thirty individual seed/fold pairs are structurally identical, including every fold for seed20261019. Its strict-improvement power is therefore reported as impossible, not as a two-user experiment. All fixed folds and seeds remain in the evaluation.

Normal sample-size calculations for nonidentical aggregate forecasts are conditional variance proxies at assumed effects, not actual effect estimates or finite-sample power guarantees. The small resulting normal-approximation counts are not reliable evidence that a tiny study would suffice. No observed cohort is reduced on that basis. Checkpoint selection already used validation; post-calibration forecasts at earlier validation dates are retrospective. Human dependence, future variance shift, uncertain observation timing and calendar-level common shocks remain limitations.

No validation mean loss or final-test outcome score was exported. The first per-fold audit is preserved separately; the aggregate report adds the correct user-level unit for overlapping folds. Full paired user-bootstrap final evaluation, comparator coverage, multistep analysis and the joint immutable analysis freeze remain open.
