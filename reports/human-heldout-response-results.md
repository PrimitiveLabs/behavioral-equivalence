# Held-out calibration-condition response models

Status: model construction and persisted-certificate replay complete. Final human test predictions remain unscored.

All five fixed known-user GRU fits contribute six folds. Each fold queries two of four known-duration groups and withholds random-calibration responses in the other two. Encoders already saw these contexts during earlier training, so this is not zero-shot context generalization. Teacher priors use standard-policy training predictions and do not assert a randomized-pool response law.

Each fold retains 20 candidate models. All 600 models have one-step certificates for 75 contexts, yielding 45,000 persisted arithmetic replays. All 600 neutral certificates were separately replayed. Streamed hashes of every reported model, partition and certificate artifact match the completion reports. The same arithmetic engine performs replay; this is not independent theorem review.

| Seed | Initial neutral states | Targeted budget-8 states across six folds | Directed witnesses across six folds |
|---|---:|---|---|
| 20261015 | 21 | 21, 21, 22, 21, 22, 22 | 0, 0, 1, 0, 1, 1 |
| 20261016 | 22 | 22, 22, 23, 22, 23, 23 | 0, 0, 1, 0, 1, 1 |
| 20261017 | 20 | 20, 20, 21, 20, 21, 21 | 0, 0, 1, 0, 1, 1 |
| 20261018 | 22 | 22, 22, 24, 22, 24, 24 | 0, 0, 2, 0, 2, 2 |
| 20261019 | 21 | 21, 21, 21, 21, 21, 21 | 0, 0, 0, 0, 0, 0 |

Fold order: q01, q02, q03, q12, q13, q23. Initial neutral-model radius is certified at most 0.01. That tolerance does not apply uniformly to the changed-context models: targeted budget-8 held-out-context radii range from approximately 0.102 to 0.201. These are point-model probability discrepancies, not measured human prediction errors.

Zero-witness folds and the fifth zero-witness fitted seed remain in the fixed evaluation. Structural forecast identity must be checked before power planning: strictly improving an identical forecast is impossible. Neutral-repair splits and capacity-fill splits are recorded separately from directed witnesses. No label-derived final-test selection or favorable seed replacement is permitted.

Remaining work: verify the fixed future-test case eligibility, calculate paired variance proxies with structural identity checks, lock the full human analysis contract, and score held-out observations. This construction does not establish human transfer, causal effects of arbitrary software changes, psychological states, or sequential prediction preservation.

The separate validation-only precision audit confirms exact forecast identity in 18 of 30 targeted-versus-matched-control seed/fold pairs, including all six folds for seed 20261019. These statements compare every coarse state and held-out context, rather than merely checking equality on sampled validation rows. Positive improvement power is not reported for identical forecasts. The aggregate analysis preserves the original user clusters and all three held-out forecasts per eligible case.
