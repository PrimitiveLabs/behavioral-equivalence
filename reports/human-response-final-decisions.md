# Complete human response-study decisions

Status: both fixed five-seed scoring matrices complete. Independent all-case/aggregate loss and fixed primary-bootstrap replay passed for the full 50-job program. Neural forward recomputation and secondary subgroup/calibration replay are outside that audit. Neither study meets its frozen all-five-seed strict-improvement criterion.

Each fit evaluates the same 375,340 original cases from 23,284 users. Seeds are fitted-model replicates on these same observations, not independent human samples. Reported intervals use paired original-user resampling, conditional on the fixed fitted models and declared calendar assumptions.

| Study | Seed | Candidate minus fixed matched control, nats | User-bootstrap95% interval |
| --- | --- | --- | --- |
| A: all calibration groups | 20261015 | -0.0005398119 | [-0.0007402078, -0.0003327988] |
| A: all calibration groups | 20261016 | -0.0004247621 | [-0.0005951152, -0.0002538074] |
| A: all calibration groups | 20261017 | -0.0000234320 | [-0.0002105559, 0.0001692495] |
| A: all calibration groups | 20261018 | -0.0002852797 | [-0.0005514986, -0.0000460238] |
| A: all calibration groups | 20261019 | -0.0003108229 | [-0.0005024559, -0.0001339395] |
| B: withheld calibration groups | 20261015 | -0.0000138494 | [-0.0000361959, 0.0000082662] |
| B: withheld calibration groups | 20261016 | 0.0000367976 | [0.0000198909, 0.0000534659] |
| B: withheld calibration groups | 20261017 | 0.0000333030 | [-0.0000067829, 0.0000761792] |
| B: withheld calibration groups | 20261018 | 0.0000067465 | [-0.0000301283, 0.0000435883] |
| B: withheld calibration groups | 20261019 | 0.0000000000 | [0.0000000000, 0.0000000000] |

Study A has negative intervals in four seeds, but seed 20261017 crosses zero, so improvement across every fixed seed is unsupported. This does not establish that refinement never helps. Study B has no strictly negative interval, seed 20261016 has a strictly positive interval and seed 20261019 has exactly identical targeted/control forecasts. No seed or fold is dropped.

Study B tests withholding response-calibration conditions, with all six complementary duration-group pairs and three eligible forecasts per original case. Losses are averaged before user aggregation. Base encoders have seen those groups during ordinary training, so this is not zero-shot prediction of unseen interfaces or a population causal treatment-effect estimate. The complete controls and baselines remain in the underlying final reports.

Exact values and report hashes: `final-human-primary-decisions.json`. These findings are separate from nominal synthetic counterexample recovery, synthetic search efficiency and supplementary endpoint calibration.
