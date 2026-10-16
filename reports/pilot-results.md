# Engineering pilot findings

Status: measured development evidence, not the preregistered final experiments. The full research goal remains active.

## Human data and predictors

KuaiRand-1K contains 11,756,073 source log rows. Full-record deduplication across all 19 official columns retains 11,661,848 interactions and 2,032,217 unordered user/timestamp bundles across 1,000 users. The v4 independent audit reconciles all aggregates and both split protocols. Feedback-availability semantics, randomized-exposure identification and 1K-to-27K user isolation remain open. Timestamp ties stay together; current-bundle outcomes do not enter their own prediction history.

A single-seed GPU pilot fits Markov, GRU64 and two-layer Transformer64 predictors under both within-user chronological and user-disjoint protocols. Every fit uses 100,000 sampled training cases and 20,000 matched pilot test cases. Neural fits use two epochs, and all models share the same history and current-scenario inputs. Test interaction-weighted NLL is 0.575662/0.546891/0.550990 for Markov/GRU/Transformer in the chronological protocol and 0.582291/0.554702/0.557089 in the user-disjoint protocol. These are deliberately small engineering baselines, not tuned final model comparisons.

Paired equal-user-weighted GRU-minus-Markov NLL differences are -0.028626 nats (conditional 95% bootstrap interval [-0.035237,-0.021937], 979 observed user clusters) and -0.037760 ([-0.049453,-0.027557], 100 user clusters), respectively. The intervals condition on these fitted models and do not include training-seed variation. They establish a pilot baseline comparison, not certified compression, causal transfer, or a confirmatory population result.

Sources: `preflight/human-preparation-v4.json`, `preflight/human-independent-audit-v4.json`, `preflight/human-training-pilot.json`, and `preflight/human-paired-analysis.json`.

## Uncertainty: coverage and useful precision

The paired synthetic comparison uses five fixed worlds, three fixed row-sample sizes and 20 repetitions per combination: 300 independently sampled transition datasets. Each is reused for two confidence-set methods and four outcome/horizon combinations. Thus 2,400 outcome checks are not 2,400 independent datasets.

All tested intervals contain the known generating probabilities within the documented floating-point tolerance. Average widths for Hoeffding versus simultaneous Clopper-Pearson intervals are 0.906 versus 0.744 at 100 samples per row, 0.558 versus 0.263 at 1,000, and 0.227 versus 0.074 at 10,000. The finite Monte Carlo suite does not prove coverage. The conditional fixed-iid-row statistical argument is separate, and beta quantile computation still lacks directed-rounding certification. No human confidence guarantee follows from this design.

Source: `preflight/uncertainty-comparison-pilot.json`.

## Counterexamples and refinement

Fifteen constructed 32-state worlds compare random, grid and adaptive search with eight actual environment evaluations each. Three withheld scalar intervention strengths produce 135 method/world/environment comparisons. Mean maximum completion-probability discrepancy falls from 0.1061 to numerical roundoff after refinement. However, state counts grow from 3 or 8 to 26, and adaptive search does not outperform the endpoint-containing fixed grid. This is a narrow, monotone synthetic intervention family. Matched-capacity, extra-data and human-intervention controls remain unrun.

The raw pilot report retains one stale limitation string from the earlier uncached implementation. `preflight/refinement-pilot-interpretation.json` records the correction without rewriting measured evidence. No per-method runtime advantage is claimed.

Sources: `preflight/refinement-pilot.json` and its interpretation sidecar.

## Certification and current failures

The earlier synthetic PRISM pilot completes 80 nominal comparisons with maximum dynamic-programming disagreement 4.44e-16 and no discrepancies beyond the calculated bound plus numerical tolerance. Small rational certificates additionally verify exact arithmetic. These do not replace the full 1,200-world benchmark, independent proof review or general numerical soundness.

Adversarial tests identify and fix an exact-partition bug caused by rounded floating aggregation hiding a tiny transition difference. Exact refinement now uses rationally normalized signatures consistent with the exact certificate checker. Approximate floating checks retain their stated numerical status.

The original GPU quantization attempt failed teacher reproduction and supplies no accepted compression result. Its pending retry was cancelled after confirming zero runners. An independent CPU reproduction matches all 40,000 saved prediction cases, with maximum probability difference 1.19e-7. CPU quantization then completed with a strict 1e-6 reproduction threshold.

Five independently fitted train-only partitions of frozen GRU representations use K=16,32,64,128,256. Relative to K=256, K=32 has additional user-weighted NLL 0.00537 (95% user-bootstrap CI 0.00243 to 0.00831) chronologically and 0.00601 (0.00291 to 0.00917) on unseen users. Both upper bounds are below the exploratory 0.01-nat margin. K=16 fails on unseen users, with upper bound 0.01388. These are unverified, separately fitted quantizations, not nested certified abstractions, and do not establish confirmatory success.

The full 1,200-world v3 synthetic matrix is running after 32-state and 512-state smoke checks. Its source and protocol are frozen, and all 22 method variants per world receive independent PRISM checks. The small earlier pilots use their recorded generator revisions. Do not pool them as one confirmatory sample.

KuaiRand-27K download, checksum/license verification, conversion and candidate preparation are complete on Modal. All 1,000 pilot users match uniquely by outcome-free timestamp/tab fingerprints and are excluded from the final candidate cohort. The independent full-data audit passes.

Sources: `preflight/human-quantization-pilot.json`, `preflight/cpu-reproduction-diagnosis.json`, `preflight/pilot-identity-report.json`, `cloud-jobs.json` and `execution-state.json`.

## Expanded uncertainty campaign

The frozen v3-generator campaign evaluates 200 worlds, two sizes (32 and 128), five families, 20 world seeds, and three independent transition-sample sizes per world. Two methods share each count dataset: 600 paired transition datasets and 1,200 method-dataset evaluations. A source-keyed parallel audit checks the complete world/property matrix and every certificate artifact hash. The earlier serial audit times out and contributes no aggregate result.

Every method-dataset's tested outcome-probability enclosures contain the known probabilities, with no unresolved comparisons. Kernel confidence-set inclusion differs: certified Hoeffding contains 600/600 generating kernels; floating-quantile Clopper-Pearson contains 594/600. The six kernel exclusions do not produce outcome-property exclusions. These are dependent state/property checks within each dataset, not millions of independent coverage trials. A conservative finite-Monte-Carlo sidecar now reports simultaneous intervals; further fixed-world repetitions remain open.

Precision is a material limitation. For completion at horizon 10 with 10,000 samples per concrete row, mean interval widths over active states are 0.770 (Hoeffding) versus 0.289 (Clopper-Pearson) at 32 states, and 0.832 versus 0.420 at 128 states. Hoeffding intervals at 100 samples per row are effectively vacuous. The exact numeric and tail certificates do not make wide bounds informative. Clopper-Pearson's floating beta quantiles retain their stated certification limitation. No human coverage claim follows.

Source: `preflight/uncertainty-main-summary.json`, with the frozen protocol and full manifest.

## Nested certified human-state compression

A separate pilot constructs outcome-conditioned transition estimates from calibration-only adjacent timestamp bundles, drawing one exposure uniformly per bundle without using its feedback in the sampling draw. It uses the frozen train-only K256 GRU quantizer, reproduces saved teacher predictions and K256 forecasts, and evaluates identical held-out cases. Transitions use one total uniform Dirichlet pseudocount per state/outcome row. Certificates apply to this estimated finite point model, not to the unknown human process.

Two candidate latent partitions (64 and 128 blocks), three tolerances (0.005, 0.03 and 0.1), and two evaluation protocols yield twelve settings. Eleven retain all 256 latent states. One chronological setting at epsilon 0.1 retains 255, a 1.004x reduction. Every setting therefore fails the proposed 2x compression criterion. This is a measured negative result for this representation, transition estimator and strict equivalence criterion. It does not establish that human histories lack compressible predictive structure. Further analysis must distinguish transition-estimation noise, conservative bisimulation requirements and insufficient learned representations.

Source: `preflight/human-learned-certified-pilot.json`.

## Finite-property compression pilot

A second certified analysis uses the same learned K256 parents and held-out cases, but preserves only 25 enumerated label properties from pending origins. Direct integer enclosures cover occupancy, reachability and count thresholds at decision horizons 1, 5, 10, 25 and 50. These are narrower finite-suite guarantees, not bisimulation or human confidence intervals.

Twelve settings vary initial state counts 64/128, probability-error budgets .005/.01/.03 and two evaluation protocols. The largest reductions retain 159 states chronologically (1.610x) and 171 for unseen users (1.497x). All settings still fail the proposed 2x reduction target. At those largest reductions, paired additional NLL is 0.000246 nats (95% CI 0.000010 to 0.000485) chronologically and -0.000142 (-0.000373 to 0.000071) for unseen users. These exploratory margins pass while compression fails. Matched-capacity clustering controls remain open.

Source: `preflight/human-property-compression-pilot.json`.

## Human-data readiness and prospective timing

The independent full 27K candidate audit checks all 323,464,444 converted raw records against 309,105,777 retained interactions, 53,720,429 timestamp bundles and 26,285 nonpilot users. It reconstructs full-record deduplication, bundle aggregates, cohort fingerprints and split metadata using a separate DuckDB implementation. It does not constitute a second full CSV parse or establish actual real-time feedback availability.

A methodological correction separates the exploratory 1K fits from the final design: per-user percentage splits do not ensure every other user's training observations precede an evaluated case. The candidate final benchmark instead uses shared UTC cutoffs and requires fitting/selection feedback to be available under the declared timing proxy before each role closes. Earlier held-out-user prefixes can supply available context, but cannot fit model weights or normalizers.

Outcome-free support checks cover both max-duration and summed-duration proxies at 0-, 24- and 48-hour embargoes. The primary summed-duration/24-hour policy leaves 25,796 known test users and 2,582 unseen test users. Fixed per-user case sampling then yields 6,623,154 known-user training cases and 3,220,197 known-user test cases, with 321,623 test cases for unseen users. The same immutable cases are shared across architectures. All three embargo artifacts are constructed privately on Modal; independent case-by-case metadata, sampling and history reconstruction passes for every constructed case. No final predictive scores have been inspected.

Sources: `preflight/kuairand-27k-independent-audit.json`, `preflight/kuairand-27k-calendar-support.json`, `preflight/kuairand-27k-confirmatory-cases.json`, and `../configs/human-case-construction-v1.json`.

## Finite Monte Carlo and measured figure

A Hoeffding/Bonferroni sidecar reports 28 simultaneous coverage intervals for the fixed synthetic world/sample-count designs. For each method, 600/600 all-event containment results yield a conservative 95% familywise interval [0.9235,1] for mean containment over those designs. Methods share samples and are not pooled as 1,200 independent trials. This finite experiment does not establish worst-case 95% coverage. Fixed-world repeats and visitation imbalance are now running in the separately frozen extension described below.

Figure 04 is rendered from the completed uncertainty campaign, with vector PDF/SVG, PNG previews, source CSVs, rendering code and hashes. The range-panel legend is corrected in version `figure04-da567fa770d4`; both panels have been visually checked. Other core figures, final human evaluation, intervention transfer, paper and release remain incomplete.

Sources: `preflight/uncertainty-monte-carlo-coverage.json` and `figures/index.json`.

## Historical model-input diagnostic, superseded by the content-aware matrix

The sampled-case audit passes on every case and history token at all three embargoes: 14,690,640 cases at 0 hours, 14,534,926 at 24 hours and 14,064,461 at 48 hours. These policies overlap and are not independent datasets. The audit rebuilds physical-row bindings, exact seeded-priority samples, user roles, shared-calendar eligibility and available histories using separate array and bounded-heap algorithms. Only the pinned hash primitive is shared with the preparation algorithm.

Training/validation-only v1 artifacts are staged separately. A top-65,535 training-item vocabulary leaves 88.79% of known-user validation items and 88.79% of unseen-user validation items unknown. No final test scores are involved in this diagnostic. This motivates an intrinsic item-context audit before the full neural comparison, rather than accepting a weak item-ID-only baseline. The current three-seed controller and neural components are development artifacts until that input specification is resolved. CPU component tests pass; GPU throughput preflight is queued. No full final neural fit has started.

The continuous integer audit stops after 4,224 certificates when reused workers cannot see newly committed source-world files. A corrected controller refreshes the volume on each reused input, checks that all certificate modules match the original audit source exactly, and resumes cached certificates in the same namespace. The failed controller source is retained. The main synthetic campaign also survives a platform preemption and replays its source-bound cache. Neither infrastructure event establishes a mathematical violation.

Sources: `preflight/kuairand-27k-confirmatory-case-audit.json`, `preflight/human-training-feature-review.json`, `preflight/neural-exposure-component-tests.json`, `cloud-jobs.json`.

## Intrinsic item metadata and randomized-pool audit

The checksum-bound archive yields all 32,038,725 reindexed basic item records with no missing IDs. Duration is invalid or absent for 6,758,072 items, and upload dates are unavailable for 579. The observed full random source contains the documented 7,583 items. Of those, 239 lack a usable duration; the four positive-duration bins contain 42, 997, 1,775 and 4,530 items. These are treatment-pool metadata counts, not response results.

Among 1,149,139 random-source rows with positive logged and metadata durations, every duration matches exactly and no duration group disagrees. No random-source row with a valid upload date places that upload after its UTC exposure date. These checks support consistency of intrinsic duration in the randomized pool. They do not prove availability provenance or consistency across all standard recommendation cases. The next input audit must check the actual sampled standard cases before revising the neural model specification. The raw random-source counts precede full-record deduplication and are not independent units for statistical inference.

The insertion probability remains withheld and the released-user cohort is selected by full-period random-exposure count. Metadata agreement does not remove those causal identification limits. No human intervention response comparison or final human test score has been computed.

Source: `preflight/kuairand-27k-item-metadata-audit.json`.

## Content-aware full matrix and independent calibration repeats

The sampled standard/random case audit is complete. Intrinsic duration context requires a valid upload date strictly before the exposure UTC day. About 72% of primary cases have usable context. Five known-user training cases, three validation cases and seven test cases have mismatched positive metadata/log durations; they remain retained and visible. The context feature never uses a current response or chooses availability using realized logged duration. Agreement does not establish online provenance.

The full 18-fit content-aware matrix runs on Modal CPU float32 with all frozen cases, three seeds, three epochs, hidden dimensions and current-item vocabulary unchanged. Three known-user context-only fits complete; GRU and Transformer training continues. No calibration/test cases enter fitting, and no final predictive test scores are inspected. The CPU backend avoids indefinite GPU allocation waits. Throughput checks reject bfloat16 autocast because it is slower on the tested hardware. Reservations are conservative execution ceilings, not measured billing.

The new repeated uncertainty extension runs 4,000 independent fixed-design calibration datasets and 16,000 paired method/support evaluations. It includes imbalanced allocation with unsupported active rows and a known closed-terminal-label control. These are synthetic specifications, not assumptions about human outcomes. The two-dataset smoke and separate eight-evaluation evidence replay pass. Tests also reject tampered counts, metrics and certificates after artifact hashes are recomputed. The full repeated campaign and its full replay remain pending.

The expanded intervention smoke passes on localized and structural-negative-control surfaces, with 128 estimated-kernel abstraction certificates. The full 400-world multidimensional campaign is submitted. It compares four search methods and six partition controls on independent evaluation coordinates. Its environment-response learner uses queried kernels only; formal abstraction certificates do not cover that learner's generalization error. No full-campaign aggregate advantage or human transfer result is claimed.

Sources: `preflight/kuairand-27k-case-context-audit.json`, `../configs/human-training-calendar-v3.json`, `../configs/uncertainty-repeated-v1.json`, `../configs/intervention-main-v1.json`, and `../spec/synthetic-intervention-analysis.md`.

The initial architecture's provisional five-seed target is retained. The original three-seed source is unchanged, and a separate immutable two-seed supplement adds twelve fits with the same cases, architectures, epochs and numeric policy. Its cloud preflight passes. Final decisions require the compression and noninferiority criteria for every seed in both protocols, rather than selecting favorable seeds. All final predictive test scores remain unopened.

## Complete intervention benchmark and calibrated-input pipeline

The [full synthetic intervention results](intervention-results.md) replace the scalar pilot as the primary synthetic intervention evidence. All 400 worlds complete, with 25,600 model/certificate bindings checked. Matched-capacity refinement improves prediction relative to random splits, but RBF acquisition underperforms random/grid acquisition when all use targeted refinement. Maximum-disagreement witness selection adds little over selecting a random observed witness. Figure 05 includes all controls and the state-growth limitation.

The repeated uncertainty extension completes all 4,000 datasets, and a separate source-bound execution replays all 16,000 paired evaluations. Its full cell-level coverage/width interpretation is still pending review; the replay uses the original certified arithmetic rather than an independent numerical implementation.

The context-only quantization integration processes all 6,623,154 training cases and reproduces its frozen validation forecasts with zero maximum discrepancy. K256 has 254 occupied training states. Full adjacent calibration inputs cover 1,393,257 known-user transitions from 25,522 users and 139,378 unseen-user transitions from 2,542 users. The context-only calibrated point models complete at K16 through K256, with 13 unsupported state/outcome rows at K256. These are engineering/calibration results, not final human compression, confidence coverage or noninferiority findings.
