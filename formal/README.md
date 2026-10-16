# Mechanically checked proof components

The proof modules are checked with Lean 4.19.0 and mathlib commit `c44e0c8ee63ca166450922a373c7409c5d26b00b` (tag `v4.19.0`). Their reported dependencies are the standard Lean foundations `propext`, `Classical.choice` and `Quot.sound`. Project axioms and admitted proofs are rejected. Compiler output retains style warnings.

- `BehavioralBounds.lean`: induction from the uniform match-probability recurrence, its observable-event error consequence, interval expansion under an absolute discrepancy allowance, and equality at zero discrepancy.
- `FiniteCoupling.lean`: a finite maximal coupling with nonnegative joint weights, both exact marginals and diagonal mass equal to one minus total variation. The proof covers both positive residual mass and the zero-residual case.
- `WeightedEvents.lean`: an absolute finite-sum inequality and the probability difference of two observable Boolean events that agree on matched paths.

These are proofs over arbitrary finite index types and real-valued weights, not tests of selected numeric examples. The coupling and weighted-event checks also reject an intentionally invalid proof. Two initial coupling drafts failed compilation and remain in the recorded history; they are not accepted proof artifacts.

The [integration argument](../spec/coupling-integration.md) explains the remaining analytical partition lift and path-marginal construction. That integration, statistical assumptions, human validity and numerical implementation correctness are not one end-to-end Lean formalization. Independent review remains open.

With authenticated Modal access, run:

```sh
modal run --detach cloud/check_formal_bounds.py
modal run --detach cloud/check_finite_coupling.py
modal run --detach cloud/check_weighted_events.py
```

Controllers use a pinned public toolchain on Modal and record proof sources, hashes, compiler versions, dependencies and outputs. They read no human data and make no model-provider calls. Configuration files under `configs/` state each check's scope and internal attempt reservation. Existing local call handles are attached rather than replaced.
