# Mechanically checked bound components

Four lemmas in `formal/BehavioralBounds.lean` passed Lean 4.19.0 with mathlib commit `c44e0c8ee63ca166450922a373c7409c5d26b00b`. The compiler reports only standard dependencies `propext`, `Classical.choice` and `Quot.sound`, with no project axioms or `sorryAx`. The proof source SHA-256 is `6871d18117cf67cf7c698a1076884ac251b6f47df50c4eee31caf68c9b4a57fc`. Cloud checking took 6.76 seconds after toolchain setup.

The checked statements cover:

1. Induction from initial and one-step match-probability hypotheses to the geometric lower bound.
2. The resulting observable-event error bound, given an indicator-disagreement inequality.
3. Expansion of an outcome interval by an absolute discrepancy allowance and intersection with [0,1].
4. Equality of event probabilities when their absolute discrepancy is zero.

This is a partial mechanical check of the analytical framework. The construction of a coupling from transition-row total variation, the path/event mapping, statistical sampling assumptions and numerical certificate implementation are outside these statements. They must not be presented as mechanically verified by this module. Full independent proof review remains open.

The pinned compiler and public dependencies were downloaded and executed on Modal. No human data, model weights or provider inference were used. The attempt reserved $0.25 inside the existing uncertainty campaign allowance, rather than establishing a new budget or a provider-enforced cap. The exact report and compiler output are in `reports/preflight/formal-bounds-1ced0e17dddc9982.json`; `formal/README.md` gives reproduction instructions.

## Additional checked components

`FiniteCoupling.lean` now passes the finite maximal-coupling construction, including exact row/column marginals and diagonal mass 1 - TV. `WeightedEvents.lean` passes the finite joint-path event-disagreement inequality. Both use the same pinned toolchain, report no admitted dependencies and reject an intentionally invalid proof of False. Their successful checks took 6.76 and 5.53 seconds after setup. Two earlier coupling drafts failed compilation; those failed reports are retained.

The successful reports are `finite-coupling-4ceca2656733c1b0.json` and `weighted-events-15700c304db4d15e.json` in `reports/preflight`. Style warnings are preserved in compiler output. The integration argument in `spec/coupling-integration.md` spells out zero-mass block handling, the concrete-successor lift, joint-path marginals and the geometric survival recurrence. Those integration identities remain analytically proved rather than fully mechanized. Numerical code correctness and human assumptions are separate obligations.

The additional proof-check attempts reserve $0.08 in total inside the existing uncertainty allowance. Reservations are not actual costs or a provider-enforced lifetime cap.
