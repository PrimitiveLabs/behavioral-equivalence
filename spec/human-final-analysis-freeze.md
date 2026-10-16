# Final human analysis lock

Status: contract implemented, joint lock not finalized and no final test predictions generated.

The final human evaluation has four executable phases: one-step comparison, direct-horizon comparison, all-group response refinement and held-out-group response refinement. One joint manifest binds every phase's source snapshot and configuration hash before any phase reads final prediction cases. The two prediction phases retain both user protocols and all five seeds. The response phases retain all five known-user seeds; unsupported unseen-user response contrasts remain documented rather than silently relaxing support thresholds.

The manifest also binds evidence for one-step comparators, direct-horizon comparators, both response designs, transition uncertainty, statistical analysis, case definitions and ablations. It includes the full ten-fit prediction matrix. Artifact paths must be private relative paths with matching content hashes. Model reports transitively bind weights and case arrays, and each executable independently checks those identities before using them.

This is an internal analysis lock and integrity check. It does not establish independent scientific review, external preregistration or correct modeling assumptions. The lock's `test_results_inspected=false` declaration must reflect actual execution history. A passing schema cannot make that declaration true.

The primary one-step endpoint remains the joint compression and noninferiority test: at least twofold state reduction and an upper paired user-clustered 95% confidence limit below 0.01 additional nats. All ten protocol/seed combinations are required. Other state capacities, horizons and cohorts are secondary and must not replace a failed primary result.

The direct-horizon comparison uses the same audited origin cohorts for every method, excludes censored targets separately by horizon, and uses only origin-available model inputs. Direct marginal forecasts, finite-state dynamics and frozen-origin iid extrapolations have separate names. Finite-model event tables, state assignments, direct forecast arrays and iid origin probabilities provide lossless reconstruction without duplicating the full model-by-event forecast tensor on disk. Float screening probabilities and exact formal certificates have separate scopes.

If an implementation defect is discovered after scoring, retain the old source, result and lock. Record the defect and issue a new version with unchanged scientific thresholds and case definitions where possible. Any change informed by observed test performance must be labeled exploratory. The final publication must describe that history accurately.

## Final evaluator implementation

`cloud/evaluate_human_responses.py` implements both response phases, retaining every stored selector and split budget. Both compare against Markov, context-only, GRU and Transformer forecasts on the same eligible randomized-exposure origins. The all-group phase scores one forecast per case. The six-fold phase supplies exactly three held-out forecasts per case to `metric_block`; losses are averaged across those forecasts before user means. Calibration gives each forecast copy one third of its original case's mass. Public metrics retain original case and user counts and report the number of forecast copies separately. Per-fold intervals are descriptive and overlapping folds are never independent sampling units.

The one-step evaluator includes feedback-availability sensitivities at 0, 24 and 48 hours. All use the same 24-hour-trained weights, quantizers and calibration models. The audited case/history cohorts can differ across those settings, so this is an availability and cohort sensitivity, not three independent replications or three refitted models. The primary endpoint uses 24 hours.

The complete planned final compute matrix is 30 one-step jobs, 10 horizon jobs and 10 response jobs. Its conservative internal timeout/startup reservation quote is $21.56144 within the existing nested-human campaign allowance. This quote is not actual spend or a provider-enforced cap. Worker limits are 8 CPU cores and 12 GiB, with 1,800 seconds for one-step jobs and 3,600 seconds for horizon/response jobs. All data, seeds and model grids are retained. The complete known-user validation comparison took 371 seconds for 1,594,880 cases; timeout planning allows for the larger final one-step cohort. Failed attempts and provider preemptions must remain in billing reconciliation.
