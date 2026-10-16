# Response calibration, refinement and certification controller review

Read calibrate_random_response.py, build_response_frontier.py,
build_heldout_response_frontiers.py and certify_response_frontier.py in full.
No source changes, private data download or provider execution occurred.

Calibration authenticates the frozen history encoder, its parent GRU source and
weights, vocabulary report and independently audited randomized cohort. Only
calibration-role rows are encoded. Discovery and estimation are split by a stable
user hash, with an assertion that users do not recur across physical buckets.
Sparse cells remain NaN in descriptive probability tables and unsupported cells
are excluded from witness selection. Unknown-duration cells are not admissible
active witnesses. The retained fine-state tables yield zero supported discovery
witnesses for all five seeds, which remains a reported limitation.

Observed-response frontier construction coarsens those tables using separately
counted distinct-user support. Discovery chooses refinements and disjoint
estimation fits probabilities. Frozen seeds, minimum support, initial blocks,
split budgets and control variants are configuration-bound. Final test outcomes
are not read. Point-model prediction differences do not establish unknown human
transition confidence or intervention identification.

Held-out construction authenticates history, calibration, coarsening and
training-prior reports, requires all six two-of-four duration folds and checks
matched control state counts at each positive split budget. Neutral training
probabilities determine the initial certified partition. Discovery witnesses
only use queried groups. The point model uses a training prior and permitted
estimation groups, rather than held-out randomized calibration outcomes.
Every saved context point certificate is replayed, as are neutral certificates
and exact fitted-count witnesses. Post-construction certificates across held-out
contexts describe the fitted point model; they do not show empirical transfer.
The current neutral-response and integer-identity helpers include separately
reviewed corrections, so historical cloud success is not fresh execution of
current source.

The separate observed-response certificate controller constructs absorbing
binary-response chains for twenty models and seventy-five contexts, writes
1500 exact-normalized one-step certificates and replays each with the same
integer verifier. This is arithmetic implementation validation, not independent
peer review or new human-confidence evidence. Fitted point radii and empirical
held-out predictive comparisons remain separate evidence streams.

Local retained reconciliation covers all twenty reports across five seeds:
100 observed frontier models, 600 held-out models, 52500 recorded point
certificates and 600 neutral certificates. It checks upstream report hashes,
complete fold grids, matched positive-budget capacities and exact rational radius
displays. See reports/preflight/response-frontier-controller-reconciliation.json.
No private NPZ rehash or fresh certificate calculation was performed locally.

Legacy entrypoints reserve finite time/resource ceilings and persist call handles,
but lack later lifetime submission locks. No relaunch was attempted. Provider
transport, concurrent writers and fresh container images are outside this review.
No tests were rerun for report-only changes. Whole-release acceptance and strict
final environment validation remain open.
