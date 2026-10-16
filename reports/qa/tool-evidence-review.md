# Supplementary launcher and reproduction tools scoped review

Read tools/run_supplementary_fits.py, freeze_synthetic.py, check_evidence.py,
scientific_reproduction.py and interval_vertices.py in full. Read existing
scientific reproduction and interval vertex tests plus new tool-boundary tests.
No source changes. Generic QA defaults apply; no root CLAUDE.md exists.

The supplementary runner validates its full frozen source map before attaching
or launching. It holds a lifetime launcher lock and uses at most two attachments.
Existing handles are retained; an exception stops further queued work and records
needs_inspection, not an inferred remote termination or automatic replacement.
Its per-job controller still owns budget reservation, which this scoped review
does not independently accept. Existing reports must pass endpoint/grid/selection
validation and remote artifact hashes before completion is recorded.

Replayed all 30 retained supplementary report validators locally. Exact report
hashes and identities are retained in supplementary-fit-validator-replay.json.
This rechecks selection rules, not remote weight bytes or predictive scores.
The freeze tool seals a specified config against code and time; it writes that
path and must not be invoked to retroactively relabel an existing campaign.
The evidence checker checks artifact identities only, not scientific success.

Scientific reproduction creates a fresh destination, all mechanism/perturbation
cases, six abstraction families, integer certificate replay and PRISM point-query
comparisons. Its unit backend is explicitly a fixture, not independent PRISM.
Historical real cloud execution remains separate evidence. Vertex enumeration
uses exact binary-float rational coordinates and bounded dimension <=8, with
row-simplex feasibility checks. Its MDP export is numerical and small-fixture
scope; it does not replace general interval-model verification.

Added 18 tests for frozen fit report identities, endpoint grids, validation ties,
missing metrics, unavailable endpoints, and vertex input boundaries. Focused
suite: 27 passed with warnings-as-errors. Full local installed-wheel suite:
1433 passed, three Torch skips and three retained horizon backend warnings.
Ruff passes. Strict final environment validation and broader controller/tool QA
remain open. No provider jobs, model calls or original outputs changed.

Evidence: reports/preflight/tool-evidence-local-validation.json and
reports/preflight/supplementary-fit-validator-replay.json. Whole-repository
ship-ready acceptance has not been established.
