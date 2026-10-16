# Supplementary explicit-feedback and watch-time results

All ten fitted-model evaluations are complete across five explicit feedback endpoints and one four-category absolute watch-time endpoint. This study was separately specified after primary click scoring began and is exploratory. The independent full-matrix audit passed all ten fits, 340 output hashes, 7,200 proper-loss cells, all reported class/top calibration bins and 1,860 paired contrasts.

The table uses the fixed start64, epsilon0.01 refinement for every fit. Delta ranges summarize fitted-seed point differences, not confidence intervals. Counts below the 0.01-nat margin are descriptive checks of individual pointwise upper bounds, not newly preregistered findings.

| Protocol | Endpoint | Latent state range | Delta NLL range, nats | At least 2x compression / five | Upper loss bound below .01 / five | Both / five |
| --- | --- | --- | --- | --- | --- | --- |
| known_users | is_like | 90 to 108 | 0.0002875 to 0.0004521 | 5/5 | 5/5 | 5/5 |
| known_users | is_follow | 66 to 71 | -0.0000043 to 0.0000713 | 5/5 | 5/5 | 5/5 |
| known_users | is_comment | 65 to 69 | 0.0000675 to 0.0001349 | 5/5 | 5/5 | 5/5 |
| known_users | is_forward | 65 to 69 | -0.0000294 to 0.0000646 | 5/5 | 5/5 | 5/5 |
| known_users | is_hate | 65 to 68 | -0.0000095 to 0.0000762 | 5/5 | 5/5 | 5/5 |
| known_users | watch_category | 241 to 249 | 0.0000130 to 0.0000428 | 0/5 | 5/5 | 0/5 |
| unseen_users | is_like | 125 to 137 | -0.0005159 to 0.0003337 | 1/5 | 5/5 | 1/5 |
| unseen_users | is_follow | 86 to 93 | -0.0003692 to -0.0000699 | 5/5 | 5/5 | 5/5 |
| unseen_users | is_comment | 89 to 94 | -0.0002066 to -0.0000989 | 5/5 | 5/5 | 5/5 |
| unseen_users | is_forward | 86 to 93 | -0.0001987 to 0.0001557 | 5/5 | 5/5 | 5/5 |
| unseen_users | is_hate | 86 to 94 | 0.0000673 to 0.0002926 | 5/5 | 5/5 | 5/5 |
| unseen_users | watch_category | 248 to 252 | -0.0000684 to 0.0000533 | 0/5 | 5/5 | 0/5 |

All 60 selected endpoint/fit rows, paired parent/random/feature contrasts and source hashes are retained in [the complete result table](supplementary-endpoint-results.json). The full stored outputs retain every other fixed state budget and model baseline. Context, GRU and Transformer heads use frozen encoders; head regularization was selected on validation data. No architecture or endpoint was selected on final-test performance.

The nominal abstraction certificate concerns one categorical emission, not joint feedback trajectories. Latent state counts exclude a common endpoint-specific absorbing-label overhead. Binary categorical Brier uses the sum over both classes, twice the usual positive-class Brier; watch-time Brier sums four classes. Missing endpoint labels are excluded under an explicit unknown-label rule.

Confidence-family uniform uncertainty allowances and sparse calibration cells can make adjusted bounds vacuous. These intervals remain conditional on stated sampling/model assumptions; binary-outcome containment is not confidence coverage. Actual feedback observation times are unavailable and calendar dependence is not removed by resampling users. The primary click compression failure and unsupported all-five response-improvement findings remain unchanged.

The fixed 0.01-nat absolute margin can be permissive for rare endpoints. Compression plus this descriptive margin alone does not establish practical utility or superiority over a trivial predictor; the complete constant, Markov, recurrent and Transformer comparisons and class calibration must remain visible.
