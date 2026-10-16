# Transition confidence and coverage QA, scoped review

Read confidence.py, robust_benchmark.py and coverage_analysis.py in full with their corresponding tests. No repository CLAUDE.md exists.

The certified Hoeffding constructor uses floating arithmetic only to propose a radius; an exact rational exponential-series lower bound decides sufficiency. Outward-rounded endpoints and unsampled whole-simplex rows are retained. The statistical guarantee requires fixed iid row samples under a frozen representation. Robust assessment checks exact normalized-kernel membership separately from event-envelope containment, noncoverage and numerical uncertainty. Monte Carlo intervals count independent transition datasets, not dependent states/properties/horizons; simultaneous comparisons include both per-cell and per-method intervals. None provides a human coverage guarantee.

Existing tests independently check rational tail evidence, outward endpoints, unsampled/tiny rows, certificate corruption and finite-binomial coverage. Before remediation, direct assertions were missing for count/alpha/precision boundaries, invalid certificate denominators, integer precision endpoints, coverage-count/alpha settings, unfinished/duplicate summaries, and complete robust cases distinguishing kernel nonmembership from event containment. No inspected finding contradicts the measured uncertainty studies.

Verdict before remediation: needs-more-tests-before-merge for these boundaries, not whole-release approval.


## Remediation and scoped closure

`tests/test_confidence_coverage_boundaries.py` adds 36 passing cases covering the listed confidence and Monte Carlo boundaries. Exact dyadic robust fixtures separately show an out-of-set kernel whose event probabilities remain contained and an out-of-set kernel with proven noncoverage; neither is called an in-set certificate contradiction. Tests check separate kernel/event summary counts, per-cell versus per-method dataset denominators, alpha/precision/count limits, repeated sample-count caching, outward rounding and certificate denominator replay.

All 36 also pass against the exact installed cloud-tested wheel, with source-path injection disabled and matching numerical dependency versions. `reports/preflight/confidence-coverage-boundaries-installed-wheel.json` binds their bytes and output. No implementation, fixed confidence procedure or original scientific result changed. Scoped verdict: no identified gap remains in these three modules. Whole-repository ship-ready status remains unassessed.
