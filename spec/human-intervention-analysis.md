# Observed random-response refinement analysis

Status: candidate contract recorded before any final predictive scoring. Input/model output hashes and source, eligibility support, conservative precision planning, all final baselines and the joint human-analysis freeze remain open. This document does not authorize treating unfinished analysis as confirmatory.

## Design and scope

The independently audited randomized cohort supplies observed response prediction cases. The target is recorded click on an exposed item, conditioned on the released user cohort and observed scenario/item context. There is no population causal effect, arbitrary-interface effect or counterfactual trajectory claim. Release selection, unknown list-insertion probability and prior encoder training on mixed logs remain limitations.

History-only GRU states use the five fixed known-user fits. Standard-training teacher-rank coarsenings were examined after the K256 sparsity finding, without reading response values. All five known-user models have jointly supported pairs at K32. This post-pilot design uses K32 with eight initial classes of four adjacent rank states. Original K256 failures remain reported. Unseen-user pair inference lacks support at all five checked resolutions and receives an explicit data-support limitation, rather than a claimed transfer result.

## Fitted candidates and controls

`configs/human-response-frontier-v1.json` fixes split budgets 2, 4 and 8. Budget 8 is the primary operating point; smaller budgets are secondary, with no substitution after final results. Discovery users select splits. Estimation outcomes are excluded from selection, while outcome-free estimation support may gate eligible cells. Both subsets must provide 50 users and 2,000 exposures per selected state/environment. Initial group definitions and state maps use training data only.

The targeted selector ranks supported empirical Bernoulli JS differences. Controls select random environments, fixed-grid environments, estimated-variance proxies or random witnesses, plus a pure random-capacity control. Every control receives the same observed table and fresh estimation data. Controls match the targeted achieved state count. When a selector exhausts supported witnesses, its remaining capacity is filled with recorded random refinement. Unrefined and full-parent models are additional baselines. Report achieved splits, random fills and all capacities.

These are offline selectors over already observed data. Candidate enumeration is recorded. This design does not test active acquisition of human responses or justify a computational search-efficiency advantage. The synthetic experiment remains the separate evidence for environment-search comparisons.

Final point emissions use only estimation outcomes, with the recorded Jeffreys smoothing, environment-wide fallback and convex aggregation weights. Displayed floating point-model discrepancies are not exact arithmetic certificates or human confidence intervals. Numerical certificate checks must be added and scoped separately.

## Primary comparison and interpretation

The primary candidate is `targeted-b8`, compared with `random_capacity-b8` at its matched achieved capacity. Evaluate the complete known-duration random test cohort in the known-user protocol using identical case IDs for all models. Unknown-duration observations remain a separately reported descriptive cohort. Cases with empty histories remain included. Report exclusions, case/user counts and all supported scenario/item-group strata without outcome-based filtering.

For each fitted seed, average paired binary NLL differences within user and then equally across users. Both probabilities use the common [1e-6,1-1e-6] clipping rule. Use a paired user-cluster percentile bootstrap with 2,000 replications and seed 20261015, reporting two-sided 95% intervals. The improvement condition is an upper endpoint strictly below zero additional nats, rather than the D6 noninferiority margin. Primary support requires improvement for every fixed known-user seed 20261015 through 20261019. This intersection-union declaration concerns these fitted models; shared users and common calendar shocks are not independent seed replications.

The unrefined, full-parent and continuous GRU/Transformer predictors, all other selectors and smaller budgets receive labeled secondary comparisons. Do not promote the best secondary result into the primary declaration. Report calibration and Brier scores, unsupported cells, sparse states, uncertainty from finite users and limitations of the user-clustering assumption. Human outcome confidence coverage is a separate estimand, not binary response containment in a model interval.

Final candidate source hashes, permitted baselines, model manifests, inference implementation, outcome-free eligible-case counts and precision/power assumptions must be locked before any final human predictive scores are opened. No prospective ability to detect a small loss change has yet been established. A null result cannot establish absence of behavioral distinctions.

## Separate held-out-condition requirement

The current all-group calibration/frontier construction is an observed future-response comparison. Because it uses every known-duration group during discovery and estimation, it does not establish transfer to conditions withheld from refinement. The original environmental-generalization requirement remains open and must not be replaced by this auxiliary result.

Before any final human scores, construct a separate six-fold calibration-group holdout design: each pair of the four known-duration groups is the query/calibration family, and the complementary pair is withheld from response calibration and refinement. Use all six pairs and all five fixed known-user models, with no favorable-fold selection. Tab remains a conditioning stratum, not a randomized intervention. Discovery and estimation users remain disjoint. The unchanged support thresholds may yield no refinement in some query families; retain that outcome and score the unchanged candidate rather than inventing a witness.

Held-out group emissions cannot use their random calibration outcomes. A candidate implementation uses frozen teacher response priors estimated from standard-policy training exposures, identical across all controls, updating only queried group emissions from the designated estimation users. Teacher-prior construction, exact item-mixture interpretation, role availability, model/partition hashes, statistical decisions and eligibility must freeze before final scoring. Existing base encoders have already seen these groups in ordinary/mixed training data. Any resulting claim is transfer to groups withheld from calibration/refinement, not zero-shot exposure to an entirely novel context or a new software interface.

This design has now been implemented and its five six-fold model builds completed, with all stored output hashes checked. Final human test evaluation remains unrun. Arbitrary-interface interventions remain unavailable in this dataset. These are separate limitations: unfinished implementable held-out-condition work is not classified as unavailable data. The all-group future-response comparison remains labeled separately and does not complete the held-out-condition experiment.

The initial eight history classes are predictive training-rank clusters, not certified neutral-environment bisimulation classes. Their later response splits must not be described as breaking a previously valid equivalence certificate. The separate D6 strict-bisimulation primary retains its own construction, failed compression outcomes and claim scope.


## Held-out study B construction and evaluation unit

The all-group study A above retains its original eight uncertified rank clusters and `random_capacity-b8` control. Study B is separate: `configs/human-heldout-frontier-v1.json` starts from training-only neutral response partitions, certifies each actual quotient at one-step epsilon 0.01, and uses exact discovery count-law gaps exceeding 0.02 to select witnesses. Refined partitions are repaired and reverified against the neutral kernel. `random_neutral_capacity-b8` is B's primary matched-capacity reference. It samples valid ordered cuts, not uniformly random partitions. Capacity fills and neutral repairs are not counted as directed witnesses.

All six two-of-four known-duration query sets are retained. For B's per-seed aggregate contrast, each future known-duration case contributes exactly three held-out-context forecasts, from the three folds that withheld its group. Average the three paired NLL differences for that case, then average within user and equally across users. Equivalently pool the three equally replicated case losses and retain the original user clusters. Do not average six fold estimates as independent observations, treat repeated cases as new samples, or average probabilities before computing losses. Per-fold contrasts are descriptive secondary analyses. Use the existing five fixed seeds, common clipping and user bootstrap settings; no unsupported or identical fold is dropped.

This aggregate construction is specified before test scoring, after inspection of model structure and validation-only variance proxies. Eighteen of the 30 fitted seed/fold comparisons have structurally identical targeted and matched-control forecasts over all declared coarse states and held-out contexts. All six folds for seed 20261019 are identical. Strict improvement for that seed is therefore impossible for these frozen forecasts. Any all-five-seed strict-improvement criterion cannot pass; this fact does not warrant selecting a replacement seed or changing the operating point. Other seed losses, calibration and secondary comparisons remain empirical questions.

The model construction and one-step arithmetic replay are complete, not the joint human-analysis freeze. Source/model/case identities, B aggregate variance planning, full comparator inference, empirical uncertainty diagnostics and final statistical execution still require completion. Neither these certificates nor the prospective use of final observations supplies a confidence guarantee for the unknown human response law.
