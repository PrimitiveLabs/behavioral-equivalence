# Supplementary watch-time and explicit-feedback evaluation

Status: specified candidate design; not fitted, scored, or represented as preregistered primary evidence. This closes an implementation gap only when its complete evaluation and audits exist. The locked click campaign and its primary decisions remain unchanged. No supplementary outcome distributions or results have been inspected to choose these thresholds.

## Targets

Five separate binary endpoints use recorded is_like, is_follow, is_comment, is_forward and is_hate. A recorded zero is the absence of that feedback indicator, not a claim about intent. Values outside {0,1}, including missing/nonfinite values, are unknown and excluded for that endpoint only, with exclusions counted. A non-click must never be relabeled as explicit dislike.

Absolute recorded play time defines one four-class endpoint: [0,1000), [1000,5000), [5000,30000), and [30000,infinity) milliseconds. These engineering bins distinguish subsecond, short, intermediate and longer recorded viewing without estimating cut points from the test set. Zero is observed zero play time; negative/nonfinite times are unknown. This is not a duration-normalized completion rate, attentiveness measure or intentional abandonment measure. Long viewing may include replay. Outcome category membership is never an input feature.

`tools/supplementary_outcomes.py` defines the candidate target rules and six tests cover boundaries, missingness, independent feedback fields, alignment, rank and empty inputs. Target construction is outside the locked primary source library to avoid changing source identities during its active campaign.

## Reuse and data integrity

Reuse the existing audited calendar roles, embargo proxy, both known/unseen-user protocols, five seeds, preprocessing and frozen context/GRU/Transformer encoders. Additional training consists of outcome-specific predictive heads, not retraining the encoders. Describe this accurately as frozen-representation predictive evaluation; it does not demonstrate an end-to-end model optimized for every outcome.

Before fitting, audit a lossless join from every canonical case to its exact source observation and target values. Candidate (user, video, time) identifiers are not documented unique event IDs. Reproduce the existing deterministic representative selection and complete-record tie resolution; do not join ambiguously or use target values to choose a representative. Store row identity digests and role-specific missingness/support counts in private cloud storage. An independent reconstruction must validate the join. All original-case/user counts and endpoint exclusions must be reported. This join audit is still outstanding.

## Comparators and abstraction

Fit five binary logistic heads and one multinomial watch-time head for each frozen context, GRU and Transformer representation. Preserve all 30 encoder fits. Use the same audited 100,000 training-origin sample per protocol as the direct-horizon fitting design if those origin identities map exactly; otherwise freeze a new deterministic train-only sample and document why. Validation selects head regularization from a prespecified finite grid before supplementary test scoring. No test-set tuning is permitted. Include constant/context and finite-order Markov baselines.

Reuse the fitted K=16/32/64/128/256 training-derived partitions as candidate state mappings. Estimate endpoint-specific binary or categorical emissions from calibration data. A new target labeling/emission model requires a new certificate: click emission bounds do not certify these endpoints. Keep shared transition uncertainty assumptions explicit. Build the label-augmented outcome model for the joint target alphabet, or certify endpoint-specific graphs with an explicit multiplicity policy. Include unverified predictive clustering and matched random-state controls. Do not silently count five separately refitted emission models as one joint behavioral-state guarantee.

For watch-time, predict a coherent probability simplex using the multinomial head or calibrated category counts, not unrelated one-vs-rest scores treated as a categorical distribution. Report categorical NLL and multiclass Brier; feedback uses binary NLL/Brier. Use training/calibration-only smoothing and probability floors fixed before test inspection. Preserve rare feedback outcomes, with uncertainty and support counts, rather than dropping endpoints after observing unfavorable results.

## Statistics and claims

Evaluate the complete eligible final cases under both protocols, paired within original users, for every fixed seed. Use the existing user-clustered bootstrap procedure and declared equal-user primary estimand. Report all endpoints and states, full candidate-reference deltas and confidence intervals. Supplementary contrasts are descriptive unless a separate multiplicity-adjusted decision rule is frozen before scoring. They cannot rescue the failed twofold-compression click primary or substitute for the original five-seed intervention decision.

## Execution gates still required

1. Complete exact-row target join audit and source identity manifest.
2. Freeze endpoint configuration, head grid, samples, missingness rules, statistical comparisons and certificate event scope.
3. Quote and reserve bounded compute within the existing research allowances before any Modal submission. Do not raise a ceiling automatically.
4. Fit heads, estimate endpoint emissions, construct/replay new certificates, score all fixed cases and independently audit outputs.
5. Add measured outcome results and limitations to the unified benchmark/publication and reproducible release.

No supplementary cloud job has been launched by writing this specification. No model-provider inference is needed.

## Physical-row attachment implementation checkpoint

Canonical IDs encode `(bucket << 32) + physical_row` before source rows are joined or sorted (see cloud/confirmatory_cases_27k.py and the independent case audit). `source_bound_targets` now decodes only unsigned 64-bit, distinct IDs in the requested bucket and verifies user/time/video/tab/current-click identity against the exact prepared physical rows. It never joins on candidate event keys. Tests retain two tied records with identical event keys/click labels but different likes and viewing durations, proving the selected physical row determines the target. Source-shape and namespace/identity corruptions are rejected. This function assumes its caller verifies the entire prepared file SHA first. Unit tests are not evidence that all 32 cloud buckets have been attached or independently audited.

Budget checkpoint: the existing human-case-audit ledger has ceiling $4.00, reservations $2.895938 and $1.104062 unreserved. The existing human-cases ledger has $0.205684 unreserved; final human scoring/release has $0.537632 unreserved. These are existing internal allowances, not interchangeable provider-enforced caps. No new ceiling or reservation was created at this checkpoint. A full join-audit quote must fit its assigned ledger; further supplementary fitting/evaluation requires its own complete quote, including head fitting, emissions/certificates, scoring and independent replay. Do not use remaining audit allowance as authorization to run an unbounded supplementary campaign.

## Fitting-cohort completion and head freeze

The packed cohort completed at source identity 287d52d00557d282, with 100,000 train and 100,000 validation cases under each protocol. Known-user samples contain 25,445 training users and 24,932 validation users; unseen-user samples contain 18,115 training users and 2,554 validation users. The sample uses canonical prediction cases rather than the horizon benchmark's one-per-bundle origin sample, because those are different case populations. Selection is fixed before supplementary outcome support or fitting is inspected and shared across all architectures and seeds. All IDs are replayed against an independently reconstructed global ordinal selection.

The supplemental head configuration now has status execution_frozen. There are 30 exact frozen-parent fits, each with five binary and one four-category head, and the same three-C validation grid. Missing training labels keep an explicit unavailable endpoint; missing validation labels select the lowest C and flag unavailable validation support. These rules do not inspect the supplementary test set. The cloud controller binds all parent reports, original encoder/preprocessing code and weights, vocabulary/scaling artifacts and the packed-cohort digest. No current-label field is passed into the encoder.

The complete 30-fit reservation is $4.50 in the existing $30 human-training-calendar-v2 ledger. The ceiling is unchanged. This supplements and does not rerun or erase earlier v2 attempts. First submit one known-user context fit as a counted member of this fixed matrix; inspect it before the remaining submissions. Actual expenditure is not inferred from reservation totals. Full final scoring, endpoint emission/certificate analysis and independent replay remain separately required and unquoted.

## Supplementary one-outcome certificate scope

`tools/supplementary_abstraction.py` constructs K pending predictive states plus C distinct absorbing outcome labels. It certifies each endpoint's next-observation categorical distribution separately, using exact-normalized integer kernels, the actual quotient radius and a horizon of one graph step. It refines proposed parent-state partitions until that exact radius meets the fixed tolerance, then replays the existing independent certificate checker. This graph is not a new sequential outcome-conditioned transition estimator. Absorbing labels must not be described as evidence about subsequent human decisions. Separate marginal endpoint certificates do not establish the joint distribution of likes, dislikes and watch-time. Calibration emissions and their uncertainty must still be estimated and audited before these routines support a human result.

Head-validation reports use the sum-of-category-squared-errors Brier convention for both binary and four-category endpoints. For binary feedback this equals twice the commonly reported single-positive-class Brier score; label the convention explicitly when combining these reports with primary click metrics. NLL determines the frozen regularization selection and is unaffected by this display convention. Do not silently compare differently scaled Brier values.

## Complete model inventory and supplementary test freeze

All 30 frozen-encoder fits completed, selecting 180 endpoint heads from 540 fitted regularization candidates. All ten calibration fits completed, producing 300 endpoint/state-count emission interval models and replaying 1,200 one-outcome abstraction certificates. See the measured training/calibration summary artifacts. These results are not held-out predictive performance.

The supplementary final configuration is frozen at SHA a4b6c454cfd265e2bfd93f834f1db2b8af59e6aee95e6c8ed0904c4a0b407df0. It includes both protocols, all five seeds, every fitted head and calibration model, all eligible test cases, the parent/frontier controls, fixed statistical procedures and original input identities. No partial model inventory may pass the freeze builder. The scorer retains all model families with endpoint-specific identical supports and pointwise original-user bootstrap intervals. This supplementary analysis is exploratory because its detailed design was specified after primary click scoring began. It cannot rescue the original click/intervention decision.

Neural forecasts are saved directly. Discrete predictions are stored losslessly as per-case state IDs plus bound shared emission/quotient tables. Markov state IDs and baseline tables are retained too. Raw identifiers, individual forecasts, fitted parameters and private loss arrays stay on Modal; only permitted aggregate reports are retrieved locally. An independent numerical replay of the resulting forecasts and statistics remains required before release.

Ten scoring calls reserve $3.00 within the existing $40 calendar-quantization allowance. The ceiling is unchanged. A single known-user seed is submitted first to validate complete-case scoring before submitting the rest. No outcome/user/grid reduction or frontier model API is used.
