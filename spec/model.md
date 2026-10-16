# Behavioral model and verification contract

Status: implemented mathematical semantics with analytical proof drafts. Independent proof review remains open. This contract defines statements about specified finite stochastic systems. Empirical human fidelity requires separate held-out evidence.

## Objects and prediction language

For each environment e in a declared family E, a fixed-policy model is M_e=(S,P_e,L,mu). S is finite, P_e is row stochastic, L assigns observable labels, and mu is a probability distribution on S. Environment and policy are fixed along a trajectory unless time, policy state or environment changes are explicitly included in an augmented model. Human histories need not actually form a finite stationary Markov process.

An H-transition observable event depends on L(X_0),...,L(X_H). H=0 includes only the initial label. Supported examples are bounded reachability, survival in active states, occupancy at a particular transition, threshold counts of labeled visits, and explicitly defined conditional events. Events about unobserved intentions or arbitrary hidden-state identities are outside this language. Unbounded temporal or reward properties need additional arguments.

Synthetic completion and disengagement labels belong to the generating environment specification. Real-data labels describe recorded feedback. KuaiRand's interface-dependent click/valid-play signal does not establish purchase, task completion or intentional abandonment. Do not transfer synthetic outcome names to human labels without observations supporting them.

## Exact and approximate quotients

Let q:S->B be a surjective, label-preserving partition map. Write p_e^q(b|s)=sum_{u:q(u)=b}P_e(u|s). An exact stable partition satisfies p_e^q(.|s)=p_e^q(.|t) for all same-block states s,t and every declared e. Its quotient has one state per block and preserves observable bounded path-event probabilities under matching initial laws.

For approximation, specify the actual abstract kernel Q_e. The current nominal quotient averages aggregated concrete rows uniformly within each block. Certify

epsilon = max_{e,s} TV(p_e^q(.|s), Q_e(.|q(s))).

Pairwise embedding similarity, agreement on one conversion rate, or a chain of pairwise epsilon-close states does not establish this condition. Approximate closeness is not a transitive relation. Every returned partition is checked against its actual abstract rows. A certificate for one environment does not cover an untested environment family.

For initial laws mu and nu, eta bounds the probability of an initial block mismatch under a stated coupling. With nu=q_pushforward(mu), eta=0. For any common observable H-transition event E_H,

|Pr_M(E_H)-Pr_Q(E_H)| <= 1-(1-eta)(1-epsilon)^H.

The coupling proof and assumptions are in [proofs](proofs.md). A certificate uniform over deterministic related origins also extends to a mixture using the same weights and matching abstract origins. It does not justify an arbitrary new initial distribution.

## Learned behavioral state construction

A frozen encoder maps the available history and declared current exposure context into a representation. A training-only quantizer maps that representation to a parent state s in {0,...,K-1}. An emission matrix E[s,y] describes the next recorded negative/positive response, and T[s,y,s'] is a specified outcome-conditioned transition estimate. Both matrices are stochastic. Their empirical estimation does not establish human Markov sufficiency or causal validity.

The labeled verification graph has 3K states: pending(s), negative(s), positive(s). It moves from pending(s) to outcome(s,y) with E[s,y], then to pending(s') with T[s,y,s']. A coherent latent partition is lifted separately across all three labels. One decision outcome becomes visible after one graph transition; H decisions from pending require 2H-1 graph transitions for H>=1. Never substitute the decision count directly into a graph-horizon certificate.

The independent K-state quantization baseline refits clusters at each K. It is not a quotient of the K256 parent. Certified compression merges parent states and reports its concrete-to-abstract map. Preserve this distinction in compression ratios and comparisons. See [the learned-state contract](learned-state-model.md).

## Finite-property alternative

An explicit finite suite can certify property-specific discrepancies directly using directed integer enclosures of the parent and abstract probabilities. Each certificate records the exact event mode, labels, horizon, count threshold and deterministic prediction origins. The finite-suite guarantee applies only to those entries, or their declared mixtures. It is narrower than a uniform bisimulation certificate and cannot be reported as preservation of all future behavior.

Direct property-error bounds can be combined with the coupling bound by taking their minimum when both apply to the same event and origin. Refinement must check acceptance using the certified quantities, even if floating predictions select candidate splits. A smaller finite-property model with a wider claim restriction does not erase a failed strict-bisimulation experiment.

## Transition uncertainty and statistical scope

Let U contain feasible concrete kernels, and let V be an abstract uncertainty family. An uncertainty-plus-abstraction certificate requires a mapping Q(P) in V for every P in U and a uniform relation error for all those pairs. Nominal-model closeness alone is insufficient. If abstract robust checking gives [l,u] and the uniform discrepancy is delta_H, widen to [max(0,l-delta_H),min(1,u+delta_H)]. Coverage of P_true by U is a separate statistical premise.

The implemented interval abstraction instead bounds aggregated row probabilities uniformly over hidden concrete origins and feasible concrete rows. A row-rectangular adversary may choose a different admissible row at successive steps. This envelope can already cover both transition uncertainty and hidden-state aggregation. Do not add nominal abstraction error a second time. Per-step rectangular extrema can be conservative for a fixed unknown stationary kernel and are not tight fixed-kernel extrema by default.

Synthetic confidence experiments fix states and independent categorical row samples before confidence-set construction. Row-simplex feasibility and simultaneous coordinate coverage are required. Learned representations, within-user dependence, adaptive selection and nonstationary human sequences do not automatically satisfy that experiment. Human probability calibration is evaluated empirically against an appropriate probability estimand, not by checking whether an interval contains one binary response.

## Numeric semantics and artifacts

The exact numeric layer interprets each finite binary float as a rational weight and normalizes each row exactly. Its TV radius and horizon discrepancy are exact rational quantities. Fixed-point probability recurrences round lower values downward and upper values upward. Serialized arithmetic-mean quotients can have a tiny nonzero radius even when the mathematical stable partition is exact. Report that radius rather than calling it zero.

PRISM nominal checks independently evaluate the supplied model text within recorded numerical tolerance. Certificate audits reconstruct that exact text and bind it to the certified quotient. PRISM agreement is implementation evidence, not a proof of a human model or a proof-assistant formalization. The current PRISM interval backend limitation for represented zero lower edges is retained; the separate integer robust engine handles those rows. Floating Clopper-Pearson quantiles remain explicitly distinguished from the exact Hoeffding tail certificate.

| Object | Authoritative implementation or evidence |
| --- | --- |
| Chain, partition, quotient | `core.py`, frozen per-campaign source |
| Decision-stage lift | `decision_model.py` |
| Exact normalized kernels and enclosures | `integer_bounds.py`, rational fixtures |
| Finite property certificates | `property_bounds.py` |
| Uniform uncertainty envelope | `uncertainty_abstraction.py` |
| Sampling and exact tail construction | `confidence.py`, stated iid assumptions |
| Solver model-text binding | `prism.py`, `certificate_audit.py` |
| Human observation and case timing | [observation policy](data-observation-policy.md), [calendar](confirmatory-calendar.md) |

Every reported result must name its input/source hashes, environment scope, labels, initial origins, horizon convention, uncertainty semantics, numerical status and empirical unit of analysis. The [certificate interface map](certificate-interfaces.md) preserves native claim-specific formats and semantic replay requirements. The initial generic ModelCertificate envelope is narrower and must not replace those records. Independent peer review remains unperformed; final empirical synthesis and release verification remain open.
