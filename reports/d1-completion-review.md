# D1 mathematical specification completion review

D1 is complete as an implemented mathematical specification, analytical proof and verification-criteria deliverable. This is an implementation-author assessment, not independent peer review, and does not close D2 through D9 or establish human validity.

| Required output | Inspected evidence | Scope |
| --- | --- | --- |
| Exact and approximate behavioral equivalence | spec/model.md, spec/proofs.md, spec/learned-state-model.md | Label-preserving finite partitions; actual quotient-row TV radius; declared fixed policy and environment family |
| Finite-horizon theorem and assumptions | spec/coupling-integration.md and reports/mathematical-soundness-review.md | Analytical block lift, correct path marginals, uniform conditional survival, initial mismatch and common observable path events |
| Checked proof components | Three pinned Lean cloud reports | Current BehavioralBounds, FiniteCoupling and WeightedEvents source hashes match successful Lean 4.19.0 checks; latter two reject retained invalid-proof controls |
| Uncertainty and numeric semantics | spec/proofs.md, spec/user-cluster-transition-uncertainty.md, spec/supplementary-uncertainty-envelope.md | Kernel-family quantifiers, rectangular versus fixed-kernel uncertainty, exact-normalized floats and directed enclosures; sampling premises separate |
| Completion, disengagement and transition properties | spec/model.md, native property and integer certificates | Synthetic labels are environmental outcomes; human click/non-click labels are not completion or intentional abandonment |
| Verification criteria and interfaces | spec/certificate-interfaces.md, executable verifiers, exported schema regression tests | Native certificate scopes retained; all five exported JSON schemas match their executable contracts |
| Existing theory and novelty limits | spec/theory-comparison.md, spec/literature.md | Related-work sources and assumptions inspected; no new bisimulation theorem, predictive minimality or exhaustive novelty claim |

The 14 focused contract, registry, integer/property and uncertainty tests pass. Five exported-schema checks also pass. The exported EvidenceRegistry schema was stale and is now corrected. Current Lean source hashes were compared directly to each successful cloud report's proof_sha256 field.

Limitations retained: the complete kernel-to-path theorem is not one end-to-end Lean formalization; the numerical and statistical layers have separate analytical premises and checks; independent mathematical peer review is unperformed. A formal model certificate does not establish that human cognition or unseen software responses satisfy those premises. These limitations constrain the claims, rather than representing missing D1 artifacts.
