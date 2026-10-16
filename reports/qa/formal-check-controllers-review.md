# Formal, neural component and interval crosscheck controller review

Read check_formal_bounds.py, check_finite_coupling.py, check_weighted_events.py,
check_neural_models.py and verify_interval_vertices.py in full. No source changes.
Generic QA defaults apply; no root CLAUDE.md exists. Whole-source acceptance open.

Lean controllers bind proof/config/controller bytes, reject admitted/new-axiom
syntax, require the pinned mathlib revision, and retain compiler status/output and
proof hashes. Finite-coupling and weighted-event controllers additionally require
an invalid False proof to fail compilation. Their three current snapshots exactly
match retained successful report identities. This is source/result identity
verification, not new Lean execution or a single mechanized end-to-end theorem.
The original source-bound Lean component checks support only their stated scope.
Version-pinned download URLs are not archive checksum pins; mathlib commit is
explicitly verified. These controllers use retained handle files but lack later
lifetime-launcher/attempt locking. Do not invoke concurrent launchers or treat an
observation failure as job termination. Reservations bound configured function
compute only; image-build and storage accounting remain separate.

Neural component controller runs two test modules with warnings-as-errors and
measures synthetic CPU throughput. This cannot establish human fidelity, final
scores or current complete test-suite coverage. Its legacy direct entrypoint
lacks the later campaign budget/handle guards and was not invoked here.
Interval crosscheck pins PRISM archive SHA, generates four small dyadic interval
fixtures, compares exact rational vertex extrema with directed bounds and 144
PRISM MDP queries, and retains model/log identities. This supports small-fixture
rectangular interval semantics, not every large uncertainty benchmark or a fixed
unknown-kernel interpretation. Existing crosscheck evidence remains historical.

Added twenty controller guard tests with simulated subprocess results and actual
function bodies stripped of provider decorators. These exercise admission checks,
revision mismatch, compiler failure/sorryAx rejection and invalid-proof control
acceptance failure. They do not compile proofs, build images, invoke PRISM or test
provider transport. Combined cloud-summary/guard tests: 30 passed with warnings-as-
errors. Full local suite: 1512 passed, three Torch skips, three retained horizon
warnings. Ruff passes. No new provider work or original result changes.

Evidence: reports/preflight/formal-controller-identity-replay.json and
reports/preflight/formal-controller-local-validation.json. Remaining controller
QA, strict final environment validation and release acceptance remain open.
