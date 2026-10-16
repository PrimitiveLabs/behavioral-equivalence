# Physical-row audits and response frontiers scoped review

Read adjacent_audit.py, adjacent_calibration.py, case_audit.py,
randomized_audit.py, multistep_audit.py, heldout_frontier.py and
response_frontier.py in full with their seven corresponding test modules.
No source changes. No root CLAUDE.md exists; generic QA conventions apply.
This is not whole-repository ship-ready acceptance.

Audit implementations reconstruct physical source rows and timestamp bundles,
calendar/availability exclusions, outcome-independent representative or priority
draws, user assignments and bounded-heap histories. Shared specified PRNG/hash
primitives do not make them wholly independent software stacks. Adjacency means
the next actual bundle, not the next selected row. Randomized audits retain all
random exposure rows, including concurrent exposures. Multistep audits reconstruct
future slices and censoring conditional on an explicitly declared calendar role.
Timing proxies and released-cohort assumptions remain; these audits cannot prove
human Markov sufficiency, unconditional survival or arbitrary causal transfer.

Refinement selectors use discovery responses and estimation support counts, not
estimation outcomes. Held-out query-group masking precedes witness search.
Policies cover targeted, random-environment, fixed-grid, uncertainty and random
witness selection. Exact count gating is distinct from floating prediction gaps.
Capacity controls fill partitions to the comparator state count; held-out neutral
repair retains its fitted neutral-model certificate. None of these operations
creates independent human evidence or upgrades failed empirical criteria.

Added 37 tests in tests/test_physical_audit_boundaries.py, covering physical
ordering/binding/availability failures across three audits, invalid target roles,
representative bundle domains, per-user caps, every zero-step selector and missing
success-count gating. Focused suite: 74 passed, including existing real Polars
selection-to-audit integration. Full installed-wheel suite: 1390 passed,
7 optional Torch/sklearn skips. Ruff passes. Evidence:
reports/preflight/physical-audit-installed-wheel-validation.json.

Small fabricated tables supplement the original full-campaign audits. Cloud
controllers still require their own source review. No provider compute, training,
human inference or original result modification occurred. Full-source QA and
release acceptance remain open.
