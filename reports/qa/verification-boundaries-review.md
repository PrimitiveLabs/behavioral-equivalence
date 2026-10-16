# Verification interface and artifact audit QA, scoped review

Read prism.py, certificate_audit.py and evidence_registry.py in full with their corresponding test files. No repository CLAUDE.md exists.

PRISM runs through an argv list, not a shell. Labels/properties are identifier-restricted, return codes are checked, and missing scalar output raises. Completed subprocess stdout/stderr are saved before result parsing. Timeout and missing-executable exceptions propagate; the adapter does not promise a completed log for a process that never returns. The certificate audit recomputes the concrete digest, quotient model text and exact integer certificate. It reports requested-tolerance excess separately rather than treating a tiny excess as a soundness violation. Registry verification establishes file identity and lineage only; its return value explicitly leaves scientific completion unassessed.

Existing assertions cover interval export, label injection and initial upper bound; source/model tampering and certificate replay; registry digest/size changes, external symlink escape, missing files/parents, duplicate IDs and self cycles. Before remediation, direct assertions were missing for PRISM transport success/extremum/property argv, failed subprocess with retained logs, missing scalar and timeout propagation, noncompleted world audit, absent/exceeded approximation budget, shared-parent lineage and empty inventory scope. These are implementation boundaries, not substitutes for the measured external PRISM campaigns.

Verdict before remediation: needs-more-tests-before-merge for those explicit gaps. This is not whole-release approval.


## Remediation and scoped closure

`tests/test_verification_boundaries.py` adds 18 passing cases. Success and failure transports use actual Python subprocesses as explicit solver stand-ins; no stand-in is presented as a live PRISM result. Tests assert all three property operators, literal argv, both retained streams, nonzero exit before scalar acceptance, absent/NaN scalar rejection, pre-launch property gates and propagated missing-executable errors. A simulated timeout asserts the actual configured 120-second subprocess deadline and does not fabricate a successful log/result. Artifact cases exercise unfinished worlds, requested budgets absent/equal/exceeded, stored compressed certificate hashes, source-report hashes and shared/cyclic evidence lineage.

All 18 also pass against the exact installed wheel from the completed cloud reproduction, with source-path injection disabled and matching dependencies. `reports/preflight/verification-boundaries-installed-wheel.json` binds the tests, wheel and output. No implementation, frozen protocol or original campaign result changed. Scoped verdict: no identified gap remains in these three reviewed modules. This is not whole-repository ship-ready status.
