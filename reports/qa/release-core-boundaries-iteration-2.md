## QA Audit: working tree, core boundaries, iteration 2

### Summary

Read-only review covers core.py, budget.py, contracts.py, analysis_freeze.py and cloud/synthetic_campaign.py in full, with their relevant tests. No repository CLAUDE.md exists; generic QA rules apply. This is a scoped review, not a whole-repository ship-ready verdict.

Verdict: needs-more-tests-before-merge.

### Permutation matrix

| Permutation | Expected behavior | Existing evidence | Status |
| --- | --- | --- | --- |
| Exact/approximate quotient, all mechanisms, nontransitivity | Preserve declared labels and actual-row radius | tests/test_core.py:20 and later named fixtures | covered unit and measured campaigns |
| Invalid chain dimensions/nonfinite/negative/initial sum; defensive copies | Reject invalid input, retain immutable probability arrays | tests/test_core.py covers invalid row sum only | missing explicit boundary assertions |
| Invalid partition and empty/mismatched environment family | Reject invalid state relations | tests/test_core.py covers incomplete partition and mixed initial labels | missing several branches |
| Horizon and coupling limits | H0, epsilon0/1, eta, invalid horizons | tests/test_core.py:86 and invalid-input fixture | partially covered |
| Budget idempotency, changed amount, overrun and changed ceiling | One atomic reservation | tests/test_budget.py:6 | covered integration |
| Competing different reservations | Never exceed ceiling | tests/test_budget.py:18 | covered threaded SQLite integration |
| Resource quote arithmetic and nonfinite rates/workloads | Correct conservative calculation or reject | no resource_reservation tests found | missing, money-sensitive |
| Synthetic resource declarations differ from decorated limits | Reject before submission | validate_protocol enforces equality, no direct boundary test | missing, money-sensitive |
| Two synthetic local launchers see absent state | One original provider call | synthetic_campaign.py main has no local lifetime lock/exclusive submission marker | missing operational protection |
| Synthetic observation timeout | Preserve original saved call, no replacement | existing state branch attaches timeout0; no entrypoint simulation | missing regression |
| RunManifest complete/running/failed without evidence | Reject inconsistent operational envelope | tests/test_contracts.py:6 | covered unit for listed states |
| Submitted/blocked/successful terminal and unsupported scientific outcomes | Preserve separation of execution and scientific support | same test covers only a subset | missing explicit cases |
| ModelCertificate interval/partition/numeric boundaries | Reject reversed intervals, empty/overlapping partitions and nonfinite bounds | exported schema comparison only | missing model-instance assertions |
| Freeze source/artifact/config changes, fit removal and role removal | Fail closed | tests/test_analysis_freeze.py:45,80,97 | covered unit for listed mutations |
| All four phase fit matrices | Retain full fixed matrix | tests/test_analysis_freeze.py:112 | covered unit |
| Freeze wrong schema/unknown phase/empty artifacts/duplicate fits/absolute paths | Fail closed | some related checks covered, not these exact mutations | missing explicit cases |

### Required test additions

1. [critical] cloud/synthetic_campaign.py local entrypoint: concurrent absent-state submission. Provide a source-preserving guarded invocation path with a real lifetime lock and persistent ambiguous-attempt protection, then test simultaneous invocations and observation failure. Do not edit historical source identities or claim old in-memory calls gained new protection.
2. [critical] budget.resource_reservation and synthetic validate_protocol: assert exact quote arithmetic, rejected invalid rates/workloads and rejected declared resources inconsistent with executable limits. Existing campaign validation already blocks negative resource declarations; there is no evidence that completed runs used an invalid quote.
3. [important] Chain/partition/coupling boundaries: add assertions for each listed failure and probability-array defensive copies.
4. [important] ModelCertificate and RunManifest: exercise model validators with valid/invalid instances, not only schema export.
5. [important] analysis-freeze guard: add the listed independent mutations while keeping all valid controller phases.

### Non-coverage findings

The small legacy rational_reachability helper evaluates raw binary-float weights; the production rational and integer certificate layers normalize exactly. Its current callers use exact stochastic fixtures. Preserve that scope instead of implying it is the universal normalized-kernel oracle.

Historical source-bound controllers and configurations must remain reproducible. A new outer guard can add safe invocation without rewriting executable source covered by earlier experiment digests. A generic cold-start DAG submitter is not part of the current release and must not be implied by a planner or CLI exit status.

### Next step for the orchestrator

Iterate: add the required regression cases and guarded invocation, review remaining source areas, then rerun the relevant checks and refresh the source/release candidate. No commit or push is approved by this scoped report.

## Orchestrator remediation evidence

The scoped findings above have regression coverage: `tests/test_release_core_boundaries.py` (38 tests), `tests/test_release_resource_boundaries.py` (26), and `tests/test_release_freeze_boundaries.py` (13). `tools/guarded_modal.py` adds the source-preserving invocation guard. Its 34 tests include actual subprocess overlap, independent-controller concurrency, literal argv, preserved call identity and ambiguous outcomes. The final-human/source-reproduction runner regression group passes all 62 tests. No provider calls occur in these tests.

The complete local suite passes **606 tests with 11 optional-dependency skips**, using `pytest -q -W error`. These skips require the separate pinned cloud environment before a zero-skips release claim. The historical 513-test cloud report does not cover these new additions.

The current implementation does not match the original full synthetic source digest. Recovered original files in `reports/reproduction/frozen-synthetic-main-v3/` match that digest exactly; their manifest hashes all 91 retained files. This resolves source availability for that campaign without changing its frozen protocol. Other campaign source preservation and the final package remain subject to the broader release review. Verdict for the whole repository remains unassessed; this scoped remediation is not authorization to bypass the required final QA gate.
