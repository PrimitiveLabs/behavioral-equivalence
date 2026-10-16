# Recorded trajectory likelihood results

All ten fixed protocol/seed fits are scored on the same 4,096 outcome-blind sampled origins per protocol. Origin selection preceded horizon-support filtering. Each model forecasts the complete recorded binary click sequence from the origin state, marginalizing unobserved future latent states. Future item contexts and future encoder states are not supplied.

All 450 metric cells and 400 paired original-user bootstrap contrasts pass independent backward replay. Maximum saved forward versus independent backward NLL difference is 2.8421709430404007e-13; maximum metric difference is 7.105427357601002e-15; maximum bootstrap difference is 3.0531133177191805e-16. All 30 stored output hashes match. The replay verifies common sampled IDs across fits; it does not independently repeat the raw representative-selection procedure.

The table reports ranges of fitted-seed point differences in NLL per observed decision. These ranges are descriptive, not confidence intervals. Each pointwise interval remains in the complete JSON table. All fits share human test observations.

| Protocol | Decisions | Contrast | Point difference range, nats/decision | Negative pointwise intervals / five |
| --- | --- | --- | --- | --- |
| known_users | 1 | finite-k64-b0p01-vs-parent | -0.0000602 to 0.0000202 | 0/5 |
| known_users | 1 | gru.frozen_origin_iid-vs-parent | -0.0066091 to -0.0040391 | 5/5 |
| known_users | 5 | finite-k64-b0p01-vs-parent | -0.0000351 to 0.0000345 | 0/5 |
| known_users | 5 | gru.frozen_origin_iid-vs-parent | 0.0133187 to 0.0169131 | 0/5 |
| known_users | 10 | finite-k64-b0p01-vs-parent | -0.0000624 to 0.0000229 | 1/5 |
| known_users | 10 | gru.frozen_origin_iid-vs-parent | 0.0257328 to 0.0292259 | 0/5 |
| known_users | 25 | finite-k64-b0p01-vs-parent | -0.0001694 to 0.0000534 | 3/5 |
| known_users | 25 | gru.frozen_origin_iid-vs-parent | 0.0323843 to 0.0356298 | 0/5 |
| known_users | 50 | finite-k64-b0p01-vs-parent | -0.0002449 to 0.0000550 | 3/5 |
| known_users | 50 | gru.frozen_origin_iid-vs-parent | 0.0394322 to 0.0431430 | 0/5 |
| unseen_users | 1 | finite-k64-b0p01-vs-parent | -0.0001437 to 0.0001335 | 1/5 |
| unseen_users | 1 | gru.frozen_origin_iid-vs-parent | -0.0053176 to -0.0024124 | 4/5 |
| unseen_users | 5 | finite-k64-b0p01-vs-parent | -0.0000197 to 0.0000441 | 0/5 |
| unseen_users | 5 | gru.frozen_origin_iid-vs-parent | 0.0157116 to 0.0203341 | 0/5 |
| unseen_users | 10 | finite-k64-b0p01-vs-parent | -0.0000295 to 0.0000238 | 0/5 |
| unseen_users | 10 | gru.frozen_origin_iid-vs-parent | 0.0271512 to 0.0324746 | 0/5 |
| unseen_users | 25 | finite-k64-b0p01-vs-parent | -0.0000603 to 0.0000296 | 3/5 |
| unseen_users | 25 | gru.frozen_origin_iid-vs-parent | 0.0348397 to 0.0413171 | 0/5 |
| unseen_users | 50 | finite-k64-b0p01-vs-parent | -0.0000687 to 0.0000290 | 3/5 |
| unseen_users | 50 | gru.frozen_origin_iid-vs-parent | 0.0416084 to 0.0488008 | 0/5 |

The strict identity abstraction matches the parent, while the fixed finite-property variant retains 242 to 250 states and does not meet twofold compression. All capacity-matched random, embedding and predictive controls remain in [the complete comparison table](human-trajectory-comparison-table.json). Context, GRU and Transformer frozen-origin iid comparators are explicitly weak joint models; direct marginal heads do not define joint distributions and are not multiplied into trajectory forecasts.

Missing horizons are censored, not negative outcomes. The 4,096-origin subset controls compute and limits population precision. This supplementary design was frozen after primary scoring began, so the results are exploratory. No new primary noninferiority, human Markov sufficiency, intentional-abandonment, cognitive-state or interface-transfer claim follows. Joint sequence accuracy is measured here separately from the primary marginal horizon analysis.
