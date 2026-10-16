# Supplementary calibration results

Status: all ten fixed calibration fits complete; supplementary human test scoring remains in progress.

The complete matrix contains 300 endpoint/state-count emission interval models and 1,200 one-outcome certificate replays. Each fit verified the fixed 64 origin cap, disjoint user buckets, source/target identities, all interval tail inequalities and every requested refined radius. No new neural calibration inference was performed.

| Endpoint | Refined state range, starting64 at epsilon .01 | Mean coordinate width at 256 states |
| --- | --- | --- |
| is_like | 90 to 137 of 256 | 0.999642 to 1.000000 |
| is_follow | 66 to 93 of 256 | 0.999557 to 1.000000 |
| is_comment | 65 to 94 of 256 | 0.999565 to 1.000000 |
| is_forward | 65 to 93 of 256 | 0.999558 to 1.000000 |
| is_hate | 65 to 94 of 256 | 0.999555 to 1.000000 |
| watch_category | 241 to 252 of 256 | 1.000000 to 1.000000 |

Compression is endpoint-specific: most feedback endpoints retain fewer states than watch-time. These are next-observation distributions in a specified model, not a new sequential human-state theorem. Some like-model fits also exceed 128 states, so the full fixed matrix cannot meet an all-fit twofold compression condition for that endpoint. Watch-time retains241 to 252 states and does not meet twofold compression. Rare feedback rates alone can make one-outcome distributions similar; held-out accuracy and calibration must be evaluated before claiming usefulness.

The256-state parameter intervals are nearly or completely vacuous. This is an unfavorable precision result. It must not be interpreted as reliable human outcome prediction merely because broad intervals contain many plausible probabilities. Confidence membership remains conditional on the stated user, representation and population assumptions.

Source: `preflight/supplementary-calibration-summary.json`, which binds all ten report byte hashes. Exact endpoint values, count-model certificates, state controls and per-fit evidence remain available in those reports and private Modal artifacts. Final score and independent forecast/statistical replay are outstanding.
