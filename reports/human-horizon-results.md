# Complete human horizon comparisons

All ten fixed protocol/seed reports are complete. The linked JSON table preserves every event, contrast and reported pointwise interval, with input hashes. Independent full-output replay remains in progress. This summary does not pool fitted seeds into independent human samples.

The following range is the minimum and maximum fitted-seed point difference. It is not a confidence interval. Negative candidate-minus-parent NLL favors the candidate.

| Protocol | Click occupancy decisions | Candidate | Delta NLL range, nats | Negative pointwise intervals / five |
| --- | --- | --- | --- | --- |
| known_users | 1 | context.direct-vs-parent | 0.035964 to 0.037596 | 0/5 |
| known_users | 1 | gru.direct-vs-parent | -0.006001 to -0.004397 | 5/5 |
| known_users | 1 | transformer.direct-vs-parent | -0.005730 to -0.003795 | 5/5 |
| known_users | 5 | context.direct-vs-parent | 0.032541 to 0.034970 | 0/5 |
| known_users | 5 | gru.direct-vs-parent | -0.010068 to -0.008592 | 5/5 |
| known_users | 5 | transformer.direct-vs-parent | -0.011547 to -0.009353 | 5/5 |
| known_users | 10 | context.direct-vs-parent | 0.023868 to 0.027454 | 0/5 |
| known_users | 10 | gru.direct-vs-parent | -0.018450 to -0.015581 | 5/5 |
| known_users | 10 | transformer.direct-vs-parent | -0.019941 to -0.016598 | 5/5 |
| known_users | 25 | context.direct-vs-parent | 0.005758 to 0.008964 | 0/5 |
| known_users | 25 | gru.direct-vs-parent | -0.035914 to -0.033372 | 5/5 |
| known_users | 25 | transformer.direct-vs-parent | -0.037282 to -0.034556 | 5/5 |
| known_users | 50 | context.direct-vs-parent | -0.001972 to -0.001004 | 5/5 |
| known_users | 50 | gru.direct-vs-parent | -0.042906 to -0.041584 | 5/5 |
| known_users | 50 | transformer.direct-vs-parent | -0.044224 to -0.043309 | 5/5 |
| unseen_users | 1 | context.direct-vs-parent | 0.037040 to 0.038117 | 0/5 |
| unseen_users | 1 | gru.direct-vs-parent | -0.004363 to -0.003449 | 5/5 |
| unseen_users | 1 | transformer.direct-vs-parent | -0.004311 to -0.003212 | 5/5 |
| unseen_users | 5 | context.direct-vs-parent | 0.033732 to 0.037639 | 0/5 |
| unseen_users | 5 | gru.direct-vs-parent | -0.010850 to -0.006434 | 5/5 |
| unseen_users | 5 | transformer.direct-vs-parent | -0.011411 to -0.007369 | 5/5 |
| unseen_users | 10 | context.direct-vs-parent | 0.023838 to 0.029464 | 0/5 |
| unseen_users | 10 | gru.direct-vs-parent | -0.018870 to -0.013385 | 5/5 |
| unseen_users | 10 | transformer.direct-vs-parent | -0.019756 to -0.013841 | 5/5 |
| unseen_users | 25 | context.direct-vs-parent | 0.006149 to 0.011779 | 0/5 |
| unseen_users | 25 | gru.direct-vs-parent | -0.036446 to -0.030243 | 5/5 |
| unseen_users | 25 | transformer.direct-vs-parent | -0.037413 to -0.031035 | 5/5 |
| unseen_users | 50 | context.direct-vs-parent | -0.001741 to 0.000564 | 2/5 |
| unseen_users | 50 | gru.direct-vs-parent | -0.042631 to -0.039757 | 5/5 |
| unseen_users | 50 | transformer.direct-vs-parent | -0.043704 to -0.040512 | 5/5 |

All 25 event targets remain in [the full comparison table](human-horizon-comparison-table.json), including cumulative-count and reachability outcomes. Horizon availability defines conditional observed cohorts. Missing future observations are censored, not negative labels. Direct heads use only origin-available features. Frozen-origin iid extrapolation and finite-state dynamics are separate comparators.

The original plan also names multistep trajectory prediction error. These marginal event forecasts do not establish joint trajectory log likelihood. That requirement remains open in the completion audit; it cannot be relabeled as an unavailable-data limitation while recorded sequences exist.
