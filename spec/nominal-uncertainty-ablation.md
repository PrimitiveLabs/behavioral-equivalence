# Removing statistical transition uncertainty

Status: full paired campaign, summary and numerical-failure classification complete. Independent arithmetic replay of the new nominal evidence remains open. This is a secondary ablation planned after the uncertainty results were inspected. It must not be described as a preregistered primary result.

## Scientific comparison

Apply an empirical point-kernel control to the exact same 4,000 retained calibration count matrices used by the repeated uncertainty experiment. Compare with the recorded Hoeffding and Clopper-Pearson results within each dataset, support condition, property, and horizon. Use the same ground-truth world, label-preserving partition, uniform hidden-state envelope, integer precision, and event definitions. This changes statistical transition uncertainty only. It does not simultaneously remove abstraction conservatism.

For each sampled row, estimate transition probabilities by exact count ratios. For a zero-sample row, use a declared uniform imputation over allowed successors. Under unknown terminal support, every successor is allowed. Under the specified closed-terminal-label control, terminal rows may transition only to states sharing their terminal label. Counts contradicting that assumption are rejected. These imputation choices are model assumptions and receive no finite-sample confidence guarantee. Report zero-sample-row counts and sensitivity separately from sampled rows.

## Numerical semantics

A rational count ratio such as 1/3 is not exactly representable as a binary float. Setting both interval endpoints to its nearest float can exclude the intended point kernel or violate exact simplex feasibility. `nominal_ablation.empirical_point_intervals` computes each ratio with rational arithmetic and rounds the lower endpoint downward and upper endpoint upward. Dyadic ratios remain exact. The interval family may contain nearby kernels in addition to the empirical point kernel, but this is numerical allowance rather than sampling uncertainty.

The resulting rows pass the same exact-normalized interval and robust-enclosure machinery as the statistical alternatives. Ground-truth model membership and event-probability containment are measured, not assumed. A test fixture demonstrates proven event-probability noncoverage for a plausible sampled count row. The control is expected to be narrower but need not contain generating probabilities.

## Required execution and analysis

1. Bind each count artifact to the frozen repeated campaign manifest, its compressed-file SHA256, source world, and the independent replay record. Reject changed or missing files. Do not resample counts.
2. Evaluate both support conditions, all three existing event properties and all four horizons. Preserve exact-directed arithmetic and use the calibration dataset as the repeated unit.
3. Pair each control record to the two existing statistical methods. Report containment, proven noncoverage, unresolved numerical cases, interval width, and paired width changes. Include active-state and initial-weighted summaries, not only all-state averages dominated by terminal rows.
4. Report fixed-world repeated variation separately from variation across worlds. State that the 4,000 datasets are not 4,000 independent worlds, and that properties, horizons, methods and support controls on one dataset are dependent.
5. Retain negative and vacuous results. Confidence intervals for the Monte Carlo summary require the same simultaneous analysis rules as the existing repeated study. Do not interpret the point-kernel control as a confidence procedure.

The full cloud execution and paired aggregate analysis are now complete. The retained nominal certificates have not yet received a separate full arithmetic replay. Measured results and the exact verification scope are in `reports/nominal-ablation-results.md`. The source-bound smoke completed 20 paired point evaluations. The full 4,000-dataset campaign was subsequently submitted under a provisional $2 CPU-only campaign allowance, with a $1.991194 timeout/startup reservation. This is an internal planning allowance, not a user-specified global cap or a provider-enforced lifetime spending limit. The numerical source is frozen at a0450320cc941c72da068fb7873cd294c60d4b1d3bb6c8c0f2ee58e7b006960d.
