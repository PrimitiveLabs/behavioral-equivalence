# Confirmatory human comparison and decision rules

Status: candidate analysis contract, not the final immutable analysis freeze. No final human predictive scores have been inspected. Final artifact hashes, abstraction operating-point rules and supported multistep cohorts must be frozen before test scoring. Positive scientific outcomes are not required for completing the experiment.

## Population, observations and comparators

Use the independently audited nonpilot 27K cohort and shared UTC calendar. The primary feedback timing policy is summed bundle-duration availability plus 24 hours. Zero- and 48-hour policies are sensitivity analyses on their own recorded case sets. They are overlapping observations, not independent replications. Actual feedback observation provenance remains unavailable.

Cases are the immutable per-user capped samples. All comparators receive identical case IDs within a protocol and timing policy. Known-user and unseen-user protocols are reported separately. Unseen-user fitting requires the declared user/calendar roles to agree. Model inputs exclude user identity, current feedback, unavailable historical feedback, random-exposure flag and full-period response statistics.

Compare the finite-order Markov baseline, a current-context-only neural predictor, GRU and Transformer predictors, the full discrete parent, independently fitted quantizations and nested parent-state reductions. Context-only, GRU and Transformer models share training-only current-item vocabularies and intrinsic duration context. Missing/too-recent metadata remains explicit. The context-only comparator omits behavioral history. Model capacities and exact parameter counts are reported; shared large item tables do not make different encoders exactly capacity matched.

Use five fixed fitted seeds, 20261015 through 20261019. The initial immutable matrix contains the first three; a separate source-bound supplement contains the last two. This supplement preserves the architecture document's five-seed robustness target and adds no selected favorable seeds. All five enter every final decision. It is declared while initial training/validation runs are underway and before any final predictive test scoring. Checkpoint selection uses the lowest equal-user validation NLL among three fixed training epochs, with earliest-epoch tie breaking. Hardware and numeric precision must be recorded before fitting. A hardware change cannot silently change case counts, architectures, epochs or seed lists.

## Learned parent and abstraction scope

The primary parent uses the GRU predictor's 128-dimensional penultimate MLP activation. Fit its state quantizer and representation normalizers on training data only. The K256 parent is the reference discrete model. Evaluate K in 16,32,64,128,256 for independent quantization baselines. Report parent-versus-continuous predictive loss so a weak discretization cannot conceal degradation relative to the strong sequence comparators.

Nested methods merge parent states and retain their explicit maps. Exact/approximate bisimulation and direct finite-property certificates have different claim scopes. All point-model certificates refer to the actual estimated, normalized finite kernel. Human transition samples are dependent and the representation is learned; no fixed-iid-row statistical coverage claim is imported from the synthetic experiment.

Transition estimation must use adjacent, exposure-indexed timestamp bundles, with one outcome-independent random representative exposure per bundle. Randomly sampled nonadjacent prediction cases cannot stand in for consecutive transitions. The current bundle's eventual response can enter a joint transition factorization, but cannot enter the next encoder until the declared feedback availability permits it. This retrospective factorization is not a causal behavioral policy.

Calibration can estimate the point-model transition kernel after encoders and quantizers freeze. A point-model property check remains valid after selecting a partition, because it checks the actual selected kernel. That fact does not establish a statistical confidence set for the unknown human process. Any stronger uncertainty claim requires its own valid sampling/selection argument or fresh data.

## Primary noninferiority estimand

For each user, calculate the mean paired difference in next-response NLL on that user's sampled test cases. Average those means equally across supported users. Clip both probability vectors to [1e-6,1-1e-6] when calculating NLL, with the same rule for all models. Case-weighted NLL and Brier scores are secondary descriptions.

For each fixed fitted seed and protocol, use a paired user-cluster bootstrap with 2,000 replications and prespecified seed 20261015. The candidate is noninferior only if the upper endpoint of its two-sided 95% interval is strictly below 0.01 additional nats per prediction. Report the complete interval and point difference, including statistically detectable loss increases that remain within the margin.

The compression target is at least 2x relative to the same K256 parent, or at most 128 reduced latent states. The primary success declaration requires both compression and noninferiority for every predeclared fitted seed in both evaluation protocols. A noninferior identity partition fails compression. A smaller separately refitted quantizer is not a successful nested certified quotient.

This is an intersection-union decision over the fixed fitted models: all required conditions must pass. A favorable seed is insufficient. Report between-seed variation and paired averages descriptively. These conditional intervals do not estimate the population of all possible training seeds. Comparisons against the continuous models and other methods retain their own reference labels and are not relabeled as the primary test.

## Operating-point selection and secondary analyses

The fixed nested-construction contract is `configs/human-nested-abstraction-v1.json`. Every seed and protocol uses the same primary rule: cluster the saved standardized K256 training centroids into 64 nonempty blocks with seed 20261015, then refine until the lifted point model meets a uniform one-step error budget of 0.01. Retain that output even if refinement exceeds 128 latent states. Such an output fails the primary compression target. No alternative grid point replaces it in the primary decision.

The complete descriptive grid starts from 64 and 128 blocks. Strict budgets are 0, 0.005, 0.01 and 0.03. Separately labeled finite-property budgets are 0.005, 0.01 and 0.03. The finite suite covers positive occupancy, positive and negative reachability, and at least two and four positive outcomes, at decision horizons 1, 5, 10, 25 and 50. All pending parent states are prediction origins. Direct finite-property certificates do not establish bisimulation. Controls use every distinct output capacity from this frozen grid, the same parent kernel, partition seed and 100-iteration clustering budget.

This shared rule is fixed before final test scoring, after the excluded 1K pilots and the first three training/validation-only GRU quantizations. It is not a blind choice made before those diagnostics. Validation selects the neural checkpoint before transition calibration. Post-validation calibration then estimates the pre-test point kernel and the fixed refinement rule constructs its partition. No post-calibration validation forecast is presented as a prospective prediction at that earlier validation date. The pre-test construction and its provenance must freeze before scoring the final test period. Report the entire grid even when the fixed primary fails.

Match random, embedding and predictive-distribution controls to the same parent and reduced state counts. Controls receive the same additional calibration/refinement observations and tuning budgets. Also report independent quantizations as a distinct baseline class. Finite-property certification is not a substitute for a failed strict-bisimulation result.

Report calibration in 20 fixed probability bins with equal-user weights, alongside support counts. Cohort comparisons by tab or history length require at least 50 users and 2,000 prediction cases; unsupported cells are reported without inferential claims. Retain rare-state frequency and missing-context summaries. Disclose common calendar/environment dependence beyond user clustering.

For multistep outcomes, define a prespecified observable event, origin rule and decision horizon before scoring. Report observation support and right-censoring at each horizon. Restricting to users with H future recorded bundles creates a conditional observed-horizon cohort; it is not an unconditional survival or intentional-abandonment estimand. Future-label events are indexed by exposure decisions and eventual recorded feedback, not necessarily by when that feedback becomes available online.

Human uncertainty evaluation must compare the appropriate cohort event-probability estimand with aggregated observations and their sampling uncertainty. Containment of one binary response in a probability interval is not a confidence-coverage test. Model-level uncertainty, empirical calibration and observational selection are reported separately.

Supported intervention prediction uses a separate frozen design and actual randomized-exposure observations, as specified in [the intervention contract](intervention-estimand.md). It cannot establish arbitrary-interface or multi-step counterfactual transfer. No human transfer claim follows from synthetic witnesses alone.

## Remaining final-freeze requirements

The confirmatory analysis is not ready to score final tests until the exact source/model/case hashes, quantizer construction, transition cohort, primary abstraction selection rule, complete baseline and ablation grid, supported multistep origins, statistical resampling configuration, intervention separation and hardware/numeric policy are recorded immutably. Any later deviation receives a dated sidecar and an exploratory label.

## Candidate observed-horizon construction, recorded before scoring

`configs/human-multistep-cohort-v1.json` now supplies a concrete origin and target construction. It remains part of the unfinished final freeze. `multistep_cohort.py` selects at most 64 origins per user from test-role timestamp bundles without observing subsequent support or outcomes. For each H, a selected origin supports a target only when its next H consecutive original bundles, including itself, remain within that user's test role. The support denominator retains selected origins that do not meet this requirement. An excluded intervening bundle cannot be skipped even when both endpoints are test-role observations.

Each timestamp bundle uses the same outcome-independent representative-exposure rule as adjacent calibration. Event targets concern its recorded binary response, not an aggregate any-click-across-all-concurrent-exposures outcome. This matches the one-exposure emission in the calibrated point model. Positive occupancy uses the Hth response; positive and negative reachability and threshold counts use the H responses beginning at the origin. Unknown targets have sentinel -1 and never become observed negative outcomes. The decision lift consumes 2H-1 graph transitions for these events.

Cohort construction has tests for user boundaries, excluded-role gaps, end-of-record censoring, reproducible within-user origin sampling, empty support, and target alignment. The full human multistep input cohort has been constructed and independently audited; model prediction scoring has not run. Calendar-level dependence and support-conditioned selection still limit the interpretation of any future user-clustered intervals.

Full source-bound multistep preparation and independent reconstruction are complete across all 32 buckets. Selected origins number 1,539,530 for known users and 153,863 for unseen users. The independent audit report SHA256 is `9f0a885467acd10e56c143bff1c0ce4c81a165e0ec46ca5c5cf94efd4b7dfa25`; the cohort report SHA256 is `200b8c2f603467ac63a807e8ae4e294286c671f35d377f49eb5fe7d6e95b198f`. See `reports/human-multistep-inputs.md` for horizon-specific support and censoring. These inputs contain private targets and outcome-blind support counts, not model prediction scores. The final analysis freeze and human predictive evaluation remain open. This secondary cohort does not replace the immutable one-step case set used by the primary noninferiority test.

## Forecast alignment and history-length convention

The forecast bridge selects the declared validation or test calendar role and, for unseen users, the matching fixed user role. Intrinsic-context sidecars must have identical immutable case IDs in identical order before row selection. User identity is retained for paired analysis and does not enter model inputs.

History-length cohorts count only available feedback-rate tokens 1 through 16. Token 17 is the empty-history start marker, so it contributes zero observed bundles; padding 0 also contributes zero. Counting all nonpadding tokens would incorrectly label cold-start rows as having a history. The tested bridge rejects noncanonical padding/marker layouts. No human metrics were scored before fixing this convention.

For full discrete parents, binary emissions are converted to the same exact-normalized-row forecast convention as saved nested next-positive probabilities, then rounded to the nearest float for proper-loss evaluation. The identity partition therefore reproduces the parent forecast convention. Independently fitted quantizers retain their labeled teacher-emission baseline convention. These conventions do not create human confidence guarantees.
