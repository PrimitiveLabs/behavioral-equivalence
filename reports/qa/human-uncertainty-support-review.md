# Human uncertainty and support QA, scoped review

Read cohort_uncertainty.py, cluster_ratio.py, resolution_support.py and precision_planning.py in full with their corresponding tests. No repository CLAUDE.md exists.

Cohort diagnostics compare equal-user mean responses with mean endpoints and return no coverage estimate for an unknown human kernel. Their paired bootstrap family is limited to the two endpoints in one declared cohort. Count-ratio intervals include zero-visit users, permit within-user dependence and require independent user tables, a frozen representation and an outcome-blind cap. Aggregate totals cannot verify those upstream assumptions. Pooled count emissions differ from teacher emissions and split the simultaneous alpha across emission/transition sets. Precision planning is validation-only and approximate; identical forecasts cannot acquire positive improvement power through larger samples.

Existing tests cover perfect within-user dependence, cap normalization, vacuous/unsampled states, interval compilation, cohort weighting/compatibility/unsupported support and structural-identity planning. Boundary review found a real dtype defect in cell_support: accepted uint64 environment or user arrays promote signed combined keys to float64, causing bincount failure. Recorded preparation fixes user IDs to signed int64 and environments to narrow integer codes; the unsigned correction must preserve those existing outputs and frozen source. Additional direct input/domain regressions and exact mixed-dtype key checks are required before scoped closure.

Verdict before remediation: needs-more-tests-before-merge, including the confirmed unsigned-key defect. No new human coverage or transfer claim follows.


## Remediation and confirmed dtype correction

`tests/test_human_uncertainty_support_boundaries.py` adds 41 cases. The unsigned defect is corrected by explicitly retaining bounded environment codes as integer cell keys and grouping cell/user identities in separate structured fields. No floating conversion or narrowing of uint64 user identities occurs. Values near 2**64 remain distinct. The original module is preserved in a source-only fixture and verified against the historical cloud archive and all four final-human source maps.

Signed int64 user/int16 environment fixtures at four cohort sizes match the original function exactly and agree with independent Python-set cell counts. The recorded ingestion encodes nonbinary user IDs as signed int64; the environment constructor uses int16 scenario codes. This is compatibility of the count operation for those caller types, not a rerun of actual support-table inference. reports/preflight/support-integer-identities-correction.json retains the defect and scope.

Additional cases cover cohort-domain/support settings, aggregate count cap/overflow/alpha requirements, pooled confidence parameters and structural-identity precision planning. No human confidence-coverage guarantee is asserted. The new two-correction wheel passes the full available suite locally, with eight optional skips; reports/preflight/human-uncertainty-support-corrected-wheel-validation.json binds its distinct identity and confirms that only quantization.py and resolution_support.py differ from the historical cloud wheel. Both original modules reconstruct the four frozen human source maps exactly. Historical results and decisions remain unchanged.

Scoped verdict: the identified four-module gaps, including the unsigned-key bug, are closed. Whole-source QA remains open; the historical zero-skips cloud report must not be attributed to the corrected wheel.
