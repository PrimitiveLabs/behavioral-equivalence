# Complete human model comparison

Thirty neural fits cover context-only, GRU and Transformer models, both protocols and all five fixed seeds. Markov and learned discrete predictors use the same final cases. The 50-job primary matrix retains the 0/24/48-hour feedback-availability sensitivities, direct horizon comparisons and both response studies. The full independent audit checks 1,660 output hashes, 13,320 aggregate proper-loss cells and 290 primary bootstrap contrasts.

| Protocol | Model | Equal-user NLL range | Binary Brier range | Descriptive ECE20 range |
| --- | --- | --- | --- | --- |
| known_users | markov | 0.633816 to 0.633816 | 0.222930 to 0.222930 | 0.008378 to 0.008378 |
| known_users | context | 0.642026 to 0.642697 | 0.226958 to 0.227304 | 0.020239 to 0.022641 |
| known_users | gru | 0.599495 to 0.600561 | 0.207104 to 0.207611 | 0.018041 to 0.024796 |
| known_users | transformer | 0.600004 to 0.600909 | 0.207330 to 0.207717 | 0.020570 to 0.027748 |
| known_users | parent_k256 | 0.603714 to 0.605439 | 0.208776 to 0.209703 | 0.015807 to 0.024795 |
| unseen_users | markov | 0.634668 to 0.634668 | 0.223193 to 0.223193 | 0.009009 to 0.009009 |
| unseen_users | context | 0.642834 to 0.643535 | 0.227173 to 0.227454 | 0.013505 to 0.021398 |
| unseen_users | gru | 0.600528 to 0.601660 | 0.207525 to 0.207958 | 0.008679 to 0.017798 |
| unseen_users | transformer | 0.600838 to 0.602820 | 0.207642 to 0.208438 | 0.007097 to 0.025375 |
| unseen_users | parent_k256 | 0.605015 to 0.605881 | 0.209343 to 0.209741 | 0.007211 to 0.017330 |

Ranges describe five fitted models on shared human observations, not confidence intervals. Every per-fit value and source hash is in human-baseline-results.json; paired per-user comparisons remain in the original complete reports. Context, GRU and Transformer parameter counts are 1,053,425, 1,171,185 and 1,341,169 respectively. The shared underlying representation and exact-capacity controls are used for quotient comparisons; cross-architecture counts are reported rather than asserted identical.

The strict primary retains 256 states in every fit. Its additional NLL versus its identical parent is zero, but all ten fits fail the required twofold compression, so predictive noninferiority alone does not establish success. The complete 485-point discrete frontier is in figures/human-final-tables.json and measured Figure2. No favorable seed, tolerance or unverified quantization replaces the primary.

The ten-fit horizon comparison retains 11,875 contrasts over 25 prespecified event/horizon targets. The separate sampled joint-trajectory study retains 450 metric cells and 400 independently replayed per-decision NLL contrasts. Direct marginal heads and coherent joint decision dynamics remain distinct. Right censoring is not a negative outcome.

All ten supplementary explicit-feedback/watch-time evaluations pass independent output replay, including all class/top-label calibration bins and 1,860 contrasts. The fixed nominal one-step refinement meets both descriptive engineering checks in 46 of 60 endpoint/fit cases; watch-time fails twofold compression in all ten. This exploratory extension cannot rescue the primary click result. An absolute 0.01-nat margin can be permissive for rare labels, so constant/Markov baselines and full class calibration remain essential.

Primary next-click NLL uses the frozen probability floor; trajectory likelihood is separately reported without clipping and with explicit zero-support accounting. All intervals are conditional on original-user resampling assumptions and fitted models, not independent population samples across seeds. Known-user learning, shared calendar effects, unavailable feedback timestamps and model misspecification limit transport and coverage. No unique psychological-state, arbitrary interface or population causal-effect claim follows.

Detailed results: human-response-final-decisions.md, human-horizon-results.md, human-trajectory-results.md, supplementary-endpoint-results.md, human-cluster-uncertainty-results.md.

All 30 one-step report identities, all-case model results at 0/24/48-hour availability assumptions and every fixed 24-hour history-cohort model result are retained in human-availability-history-results.json. These are support-aware sensitivities with frozen models, not retrained memory-width experiments or causal estimates.
