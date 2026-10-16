# Within-state response heterogeneity

Post-hoc descriptive diagnostics use all ten fixed 24-hour fits and existing saved origin-state assignments. Case, user, forecast, source-feature and model identities are checked. Original users are weighted equally within each state/stratum cell. Published cell rates require at least 50 users and 2,000 cases. Seven focused tests pass in the pinned cloud environment.

| Protocol | Axis | States with two supported strata, range across five fits | Largest observed rate span |
| --- | --- | --- | --- |
| known_users | history_length | 0 to 0 | unsupported |
| known_users | context_missingness | 13 to 43 | 0.112549 |
| known_users | tab | 9 to 24 | 0.424169 |
| unseen_users | history_length | 0 to 0 | unsupported |
| unseen_users | context_missingness | 0 to 1 | 0.003206 |
| unseen_users | tab | 0 to 0 | unsupported |

History strata are 0, 1-8, 9-16 and 17-32 available feedback-rate tokens. No state has two supported history strata, so this analysis does not establish conditional history independence. Unseen-user tab comparisons are also unsupported. Unsupported cells are retained with rates suppressed, rather than dropped or represented as zero differences.

Observed spans are selected descriptive maxima, not confidence intervals, multiplicity-adjusted discoveries or causal contrasts. Known-user within-state response rates can differ across supported context/tab strata. This cautions against treating a finite learned state as an established sufficient description of human behavior. User composition, sampling variation, logging policy and calendar effects remain possible contributors. The diagnostic cannot prove or refute general human Markovity.

The complete aggregate cell data, support counts and exact input/source bindings are in preflight/state-sufficiency-d1d51a45cdd9c084.json. No new encoder training, neural inference, primary threshold change or final-test model selection was performed.
