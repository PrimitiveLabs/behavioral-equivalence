# Repeated calibration and empirical-point replay scoped review

Read repeated_analysis.py, repeated_evidence.py, nominal_replay.py,
nominal_chunk.py and nominal_ablation.py in full with their five corresponding
test modules. No source changes. No root CLAUDE.md exists; generic QA conventions
apply. This report does not declare whole-repository ship-ready acceptance.

Repeated summaries require a complete paired method/support/repetition design.
One calibration dataset remains one independent trial despite four paired
method/support evaluations. These structural summaries consume typed campaign
records; source-bound arithmetic replay is a separate requirement.

Repeated evidence replay checks report/spec identity, frozen design bounds,
generated world, row allocation, artifact names and hashes, seeded multinomial
draws, complete retained certificates and all saved metrics/widths. It recomputes
with the original arithmetic implementation. This is separately executed replay,
not an independent mathematical implementation or second theorem proof.

Nominal ablation reuses retained calibration counts without resampling. Numeric
point envelopes use Python-integer totals and outward rational-to-float rounding.
Zero rows are explicitly imputed uniformly over allowed successors. Known
terminal closure rejects conflicting counts. These intervals are numeric point
estimate enclosures and explicitly carry no statistical coverage guarantee.
Nominal replay binds both reports and compressed evidence, recomputes canonical
numerical evidence, and verifies agreement between report and saved evidence.

Added 22 tests in tests/test_repeated_replay_boundaries.py. They cover mismatched
status/source/protocol/spec/world identities, incomplete or duplicate designs,
invalid widths, altered confidence certificates and exact uint64 counts whose
row total exceeds 2**64. No original campaign outputs were changed.
Focused tests: 43 passed. Full current installed-wheel suite: 1353 passed,
7 optional Torch/sklearn skips. Ruff passes. Evidence:
reports/preflight/repeated-replay-installed-wheel-validation.json.

These small constructed replay tests supplement the retained full-campaign
audits. They do not establish real human-kernel coverage, new empirical results,
or a fresh cloud zero-skips run. Full source QA and release acceptance remain open.
