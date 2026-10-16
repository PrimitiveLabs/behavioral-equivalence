# Multistep target and cohort QA, scoped review

Read horizon_targets.py, multistep_cohort.py, nested_design.py and training_horizons.py in full with their corresponding tests. No repository CLAUDE.md exists.

Origin selection uses user/time/role and seeded per-user caps before future support or outcomes. Observed windows include the origin decision, never cross users, and reject intervening excluded roles even when endpoints match. Missing horizons retain -1 for every event. Training/validation remapping excludes all other roles. The five-event suite uses decision-indexed graph horizons and keeps finite-property scope explicit. The origin-iid reference holds a prediction-time probability fixed; it is not a learned rollout. Existing tests compare that reference to a single-state decision chain, preserve missing support, check role/user gaps, outcome-blind sampling and fixed parent-capacity controls/cache bindings.

Before remediation, direct tests remained for bundle/origin/horizon/cap and recorded-label alignment boundaries, mask/count/last-label consistency, iid-vector validity, parent-feature/fit-role settings and exhaustive short binary target semantics. Generated cohort invariants are supplied by observed_horizon_cohorts; arbitrary fabricated internal cohort dictionaries are not a new public ingestion interface.

Verdict before remediation: needs-more-tests-before-merge for the listed gates, not whole-release approval. Censoring does not identify intentional abandonment, and observed-window event prediction is not unconditional human survival.


## Remediation and scoped closure

`tests/test_multistep_target_boundaries.py` adds 45 cases covering the listed domains. Exhaustive enumeration of all 16 four-decision binary trajectories independently checks occupancy, any-positive, any-negative, two-positive and four-positive targets at H1/2/3/4; censored rows remain -1 for all events. Additional cases assert no invented support for empty cohorts and direct rejection of malformed times, roles, masks, counts, label support and reference probabilities. Fitting targets retain the existing tests showing held-out labels cannot affect within-role outcomes or origin selection.

All added tests and the full available suite pass against the unchanged corrected installed wheel with source-path injection disabled. reports/preflight/multistep-targets-corrected-wheel-validation.json binds the exact commands and test bytes. No implementation or frozen output changed. Scoped verdict: no identified gap remains in these four modules. Censoring and arbitrary-interface counterfactual limitations remain unchanged; whole-source QA remains open.
