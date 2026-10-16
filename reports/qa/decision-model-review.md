# Decision-stage model QA, scoped review

Read `src/behavioral_equivalence/decision_model.py` and `tests/test_decision_model.py` in full. The lift has separate pending/negative/positive labels, two graph transitions per decision and a visible outcome after the first. Its horizon map is 0 at zero decisions and 2H-1 otherwise. These are specified-model semantics, not a human cognition or Markov claim.

The existing tests check emission error in quotient certificates, related-state prediction containment, exact identical-emission reduction and rational certificate replay. `tests/test_decision_boundaries.py` adds 13 cases for invalid decision counts, empty/mismatched dimensions, nonfinite/unnormalized/negative stochastic inputs through the Chain boundary, and a direct valid graph mapping. Assertions inspect initial-stage mass, every outcome-conditioned successor row, stage labels, lifted partition and horizon endpoints.

All 13 added cases pass against the exact cloud-tested installed wheel. `reports/preflight/preparation-support-installed-wheel.json` binds their bytes and output alongside the preparation/support cases. No implementation changed. Scoped verdict: no identified gap remains in this reviewed slice. Whole-repository ship-ready status remains unassessed.
