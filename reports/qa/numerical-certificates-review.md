# Numerical certificate QA, scoped review

Read in full: `src/behavioral_equivalence/rational.py`, `integer_bounds.py`, `uncertainty.py`, and the rational, integer, uncertainty-abstraction, property-bounds and interval-vertex tests. No repository CLAUDE.md exists. This is a scoped audit, not whole-repository approval.

## Covered scientific paths

Exact rational normalized-kernel reduction and tamper rejection have assertions in `tests/test_rational.py`. Integer abstraction containment is compared against exact rational probabilities, including conditional-event containment and undefined conditioning, in `tests/test_integer_bounds.py`. The same tests distinguish exact interval feasibility from floating row tolerance and bound directed-arithmetic width. `tests/test_property_bounds.py` independently enumerates finite paths across H0/1/3/5, occupancy, reachability and count thresholds. It also demonstrates that first-step equality does not certify unlisted future properties. `tests/test_uncertainty_abstraction.py` exercises simplex projection, outward rounding, heterogeneous concrete origins, tampering and exact infeasibility. `tests/test_interval_vertices.py` compares extrema against SciPy linear programming and tests supported zero-lower-edge export. The small external PRISM vertex study remains separate from full uncertainty-campaign checking.

## Gaps before merge

Verdict: needs-more-tests-before-merge for these numerical boundaries.

1. Rational `certify`: explicitly assert rejection of negative, Boolean and fractional horizons, and certify H0 with exact zero discrepancy. Core reachability horizon tests do not directly assert this separate gate.
2. Integer `reachability_enclosures` and `robust_enclosures`: explicitly assert empty/negative/Boolean/fractional horizon rejection and precision boundaries. H0 robust endpoint assertions should cover all four directed vectors. Valid scientific results do not establish these input-boundary assertions.
3. IntervalChain: explicitly assert dimension/nonfinite/reversed/out-of-range boundaries and defensive copies/read-only storage. Existing infeasible-row tests cover only part of its constructor boundary.
4. Statistical interval constructors: directly test negative/nonfinite/wrong-shape counts and alpha endpoints, retaining unsampled-row [0,1] and zero/all-positive counts. These are validation tests, not an empirical proof of statistical coverage.

No evidence inspected here contradicts the completed numerical campaign results. These gaps concern regression assertions and do not turn model-level containment into a human-validity guarantee. No implementation changes or additional provider calls occurred in this review.

## Orchestrator remediation

All four identified boundary gaps now have direct assertions in `tests/test_numerical_boundaries.py` (62 parameterized tests). The test groups exercise rational invalid horizons/H0, both integer enclosure interfaces' invalid horizons and precision endpoints, all four H0 robust vectors, interval constructor boundaries and protected input copies, and both statistical constructors' invalid counts/alpha and unsampled/all-positive/zero-count edges.

All 62 pass locally and against the exact installed wheel from the completed 626-test cloud check, with source-path injection disabled. `reports/preflight/numerical-boundaries-installed-wheel.json` binds the test bytes, wheel hash, dependencies, command and output. The full local suite passes 670 tests with 11 optional-dependency skips, and Ruff passes. No implementation or frozen study source changed. One local environment failure before collection is retained in the supplemental report.

Scoped verdict: the four identified numerical-boundary gaps are closed. This does not assert whole-repository ship-ready status or attribute the new tests to the historical cloud execution. The current source archive's continuity report must prove that only this test file differs from the cloud-tested archive before combining these records as release evidence.
