# Mathematical soundness review

Status: implementation-author analytical review, not independent peer review. This review covers the argument in spec/proofs.md, the finite construction in spec/coupling-integration.md and the numerical semantics in integer_bounds.py and uncertainty_abstraction.py. It does not claim that every implementation branch is mechanized or that the human-generating process satisfies the assumptions.

## Finite label-path argument

The concrete and abstract kernels are finite, nonnegative and row stochastic. A label-preserving partition fixes the relation; the actual abstract row is part of the statement. On every related pair, a finite maximal coupling of aggregated concrete and abstract successor rows gives match mass at least 1-epsilon. The checked finite-coupling component includes the zero-residual case.

The partition lift is well defined on positive-mass successor blocks. For a zero-mass block, nonnegativity and the row marginal force both concrete transitions and coupling row weights to zero; assigning zero joint successor weight loses no mass. Summing the lift over abstract successors gives the concrete kernel. Summing over concrete states grouped by their block gives the abstract kernel. Off the relation, the product of the two successor rows preserves both marginals.

A joint initial coupling and the resulting joint transition kernel therefore define a normalized finite path law with both complete path marginals, not just matching timewise marginal distributions. Conditional on a prefix with every pair related, the uniform successor bound still holds. Its survival recurrence requires no independence of mismatch events. For epsilon,eta in [0,1], the checked recurrence gives surviving match mass at least (1-eta)(1-epsilon)^H. Label agreement on those paths makes any common event of labels at times 0 through H identical there. The checked weighted-event inequality bounds probability discrepancy by unmatched mass. H=0, epsilon=0, eta=0 and zero-support successor blocks are included.

Thus the analytical integration supplies the finite-horizon theorem under its stated premises. The finite-coupling, recurrence and weighted-event components have separate pinned Lean checks and retained invalid-proof controls. The block lift and full path-law construction remain analytical, not one end-to-end Lean formalization. Rewards, unbounded time, arbitrary hidden-state events, unrestricted branching-time tests and changing unmodeled policies/environments are outside this result.

## Quantifiers and uncertainty

A nominal radius cannot be applied to every kernel in a confidence set without a uniform argument. Conditional composition requires a corresponding admissible abstract kernel for every concrete kernel, a common uniform error and correctly coupled initial laws. The rectangular block-hull construction supplies a different sound enclosure: every projected concrete row belongs to the abstract box/simplex, as does every conditional mixture over hidden concrete states. A history-dependent abstract adversary can reproduce each feasible concrete label-path law. Its robust extrema enclose fixed stationary concrete kernels but can be more conservative than fixed-kernel extrema. This construction already includes aggregation error; adding a nominal radius again is unnecessary.

The supplementary categorical envelope is restricted to one outcome. It computes a uniform radius across the entire feasible row family, with an abstract kernel equal to the block mean for each admissible parent. Its interval widening is justified by that uniform bound, rather than by the nominal fitted radius. It does not certify a joint law across different feedback endpoints.

## Numerical enclosure argument

Serialized nominal binary floats define exact rational weights normalized within each row. The actual quotient radius and horizon discrepancy are rational. For probability recurrences, nonnegative stochastic averaging preserves endpoint ordering. Integer floor on lower weighted sums and ceiling on upper sums therefore preserve containment by induction from the exact initial indicator. One-sided rounding error accumulates by at most H/2^bits.

For uncertain rows, exact rational feasibility is checked before optimization. Greedy allocation from row lower bounds, in successor-value order, solves the row linear objective: exchanging mass from a worse successor to a better unsaturated one cannot hurt the objective. Four directed recurrences independently enclose robust minimum and maximum. These claims concern per-step rectangular uncertainty and explicitly supplied event semantics. PRISM point agreement within numerical tolerance is separate implementation evidence.

Fixed finite-property certificates directly enclose their listed occupancy/reach/count events. They do not establish all-trace preservation. Count recurrences include the current state's label exactly once; the three-stage behavioral graph exposes one recorded outcome at the corresponding outcome stage, so H decisions use 2H-1 graph transitions. Conditional event ratios require a positive conditioning denominator, with abstraction adjustment before ratio formation.

## Statistical boundary

The fixed-design synthetic Hoeffding statement assumes independently sampled categorical successors with states, representation and row sample counts fixed independently. A simultaneous coordinate event and simplex intersection preserve true-kernel membership on that event. The exact positive Taylor partial sum used in the implementation lower-bounds the exponential, so its checked tail inequality validates the chosen dyadic radius. Floating logarithms only propose a candidate radius. Floating Clopper-Pearson quantiles do not inherit that numerical proof.

The separate capped-user ratio argument requires independent user tables, an appropriate common population, a representation treated as fixed, outcome-blind caps and correct zero-visit user denominators. Known-user learning, common calendar effects and temporal transport remain unverified assumptions. No mathematical argument here establishes human Markov sufficiency, cognitive-state identification, empirical interval coverage or held-out likelihood noninferiority. Those are separately evaluated or explicitly limited.

## Backend boundary and remaining review

The retained PRISM 4.11 interval-engine log rejects a represented zero lower edge. The mathematical rectangular enclosure remains defined for such rows; this software limitation is not a theorem excluding them. Current official documentation describes interval-model robust queries but does not by itself establish coverage of this retained failing case. A separate exact vertex-to-ordinary-MDP cross-check has now passed 144 bounded PRISM queries on four small dyadic fixtures, with maximum solver discrepancy 1.1102230246251565e-16. Every exact vertex-oracle value also lies inside the directed integer robust enclosure. Its scope is small fixtures, not external checking of every large uncertainty model.

Remaining release review concerns complete literature comparison, cross-format certificate interfaces, fresh scientific reproduction, full implementation QA and final claim-to-evidence reconciliation. External mathematical peer review has not been performed. These limitations must remain visible in the publication.
