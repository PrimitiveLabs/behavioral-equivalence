# Comparative inference validation

Status: both fixed validation probes complete. Final human predictive test scores remain ungenerated.

| Protocol | Cases | Candidate models | Verified output files | Largest neural reproduction error |
|---|---:|---:|---:|---:|
| known_users | 1,594,880 | 42 | 33 | 1.19e-07 |
| unseen_users | 159,059 | 42 | 33 | 1.79e-07 |

Each probe includes Markov, context-only, GRU and Transformer predictors, all five independently fitted quantizers, the exact-normalized calibrated parent and all32nested candidates and matched controls. Every candidate uses the same immutable case IDs within its protocol. The encoder probabilities reproduce saved validation probabilities within1e-6; every quantizer state assignment and teacher-emission forecast matches its saved artifact exactly. All stored forecast-bucket and private user-loss file hashes were independently read and checked.

For both primary strict models, the candidate and discrete parent are the same point forecast: paired mean difference and percentile interval are exactly zero. This is an identity regression check, not evidence of useful human compression, because both primary outputs retain256states. It cannot replace the ten-fit final efficiency/prediction decision.

Metric aggregation uses complete users within source buckets, equal-user primary weighting and shared original-user bootstrap draws. Unsupported history/tab/context cells retain support and descriptive summaries without inferential claims. Current reports are retrospective engineering validation: neural checkpoint selection already used validation, and calibration-based model construction postdates these observations. Do not present these losses as prospective or confirmatory human fidelity.

Source digest: `22d587bbc968fded84ac0649bbe755787c5e80d4abc601ce7de38e80e5801b1e`. Frozen source and an idempotent restored controller are retained. Raw forecasts and user-level losses remain private in the research Modal volume; the reports contain aggregate metrics and identities only.
