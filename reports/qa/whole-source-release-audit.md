## QA Audit: research/behavioral-equivalence research release

### Summary
- Executable source files reviewed: 210, all matched to their scoped review hashes.
- Full pinned installed-wheel execution: 1663 tests, zero skips, warnings as errors.
- Additional local launcher/release/freeze replay: 92 cases, not additional scientific samples.
- Permutation inventory: retained per-module matrices and sixteen release-checker cases, plus twelve actual-evaluator lock cases. Test counts are not asserted to equal unique branch counts.
- Unresolved concrete test gaps identified in the reviewed research execution paths: 0.
- Verdict: ship-ready.

The verdict concerns the specified research source and reproducible release, not
independent theorem peer review, correctness of provider SDK internals, production
service availability or human-model validity. No root/nested CLAUDE.md was found;
generic QA rules and the supplied global AGENTS instructions apply. This report
consolidates the complete source reviews; it does not replace their findings or
promote artifact hashes to proof of scientific correctness.

### Permutation matrix
| Reviewed behavior | Unit / filesystem or subprocess assertion anchor | Independent or integrated evidence | Status |
| --- | --- | --- | --- |
| Invalid probability objects, immutable model inputs, partitions and environment families | tests/test_release_core_boundaries.py:32 | Full installed-wheel suite | Covered |
| Bounded error, invalid certificates and serialized arithmetic | tests/test_release_core_boundaries.py:127; tests/test_integer_bounds.py | Complete synthetic certificate audit and fresh ten-world subset | Covered |
| Data split/availability, physical identities, unsupported states and numeric boundaries | Linked preparation, identifier, uncertainty and response scoped reviews in source-audit-inventory.json | Independent full bucket audits and frozen fifty-job scoring/output audits | Covered |
| Every analysis-lock identity and phase boundary | tests/test_release_freeze_boundaries.py:33 | tests/test_final_controller_freeze_gates.py:15 exercises actual evaluators before private reads | Covered |
| Budget rounding, invalid quotes, failed reservations | tests/test_release_resource_boundaries.py:52 | tests/test_source_reproduction_runner.py:119 checks no submission on rejection | Covered |
| Cross-process locks and independent concurrent jobs | tests/test_guarded_modal.py:270; tests/test_final_human_runner.py:97 | Real local processes and filesystem locks | Covered |
| Partial handle writes, unknown submissions, failed observation/download, sibling failures | tests/test_final_human_runner.py:131; tests/test_guarded_modal.py:114 | Controlled provider boundary, real retained local state | Covered |
| Changed or missing original handles never authorize replacement | tests/test_source_reproduction_runner.py:81; tests/test_source_reproduction_runner.py:101 | Persistent attempt-marker and attachment assertions | Covered |
| Archive path/type/symlink restrictions and deterministic contents | tests/test_source_package.py:15; tests/test_source_package.py:44 | Exact candidate member readback and secret/data-file scans | Covered |
| Wheel failure/cardinality, install/import failure, test failure/skips, scientific rerun failure | tests/test_release_checker_boundaries.py:83 | Fresh current-source successful build/install/test/scientific run | Covered |
| Comparative loss, calibration, user pairing and uncertainty estimands | Evaluation and supplementary scoped review records | Frozen original-user proper-loss/bootstrap replays, separate model-level certificates | Covered |
| Negative controls, counterexamples and held-out environment scope | Synthetic/diversity and response refinement scoped review records | Full synthetic intervention matrix; limited human exposure study with negative findings retained | Covered |
| Figures, evidence indices and source-version continuity | Figure/tool scoped reviews and manifest checks | Selected measured figure visual QA and exact content hash closure | Covered |

All 210 per-file reviews remain available through reports/qa/source-audit-inventory.json.
The source roster was compared with the actual src/cloud/tools Python paths;
no executable source was omitted based only on its staged Git state. Each scoped
remediation and its evidence is retained rather than replacing historical failures.

### Required test additions
None remain from the concrete findings identified in the complete review and
subsequent release/freeze/launcher gate audit. External SDK/server internals are
not treated as code owned by this repository. Provider boundary outcomes are
controlled in local failure tests; real campaign executions and the fresh pinned
reproduction supply separate successful integration evidence.

### Non-coverage findings
Legacy direct entrypoints are preserved for source-bound historical reproduction.
Unattended resumable operation must use the documented guarded launcher; local
locks are not distributed exactly-once execution. This release does not certify
arbitrary external SDK failures or adversarial mutable provider storage. These
limits remain in reproduction instructions and the scoped controller reviews.

Formal results are specified-model guarantees. The primary human compression
criterion fails; human response improvement across every seed is unsupported;
several uncertainty intervals are vacuous. These are completed negative findings,
not software test gaps or grounds to relabel empirical success. Full human
retraining, an end-to-end Lean mechanization and external peer review are not
claimed. The fresh scientific reproduction is a ten-world subset, not another
1200-world campaign.

### Next step for the orchestrator
Proceed to lint/build/test. Use the exact current-source installed-wheel build,
1663-test warnings-as-errors report and scientific rerun, then confirm Ruff and
candidate identity remain current. If source changes, invalidate this acceptance.
After those checks, create the QA marker and publish only the isolated permitted
release tree on a feature branch. Public repository creation/publication is a
separate step and has not occurred at this verdict.
