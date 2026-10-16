# Coupling integration and checked components

This supplements `spec/proofs.md`. It supplies the finite construction behind the conditional match recurrence. The maximal-coupling, recurrence and weighted-event components are mechanically checked separately. The partition lift and path-marginal identities below remain analytical arguments, not fully mechanized implementation proofs.

## Kernels and block coupling

Let S and B be finite, q:S→B a label-preserving partition, P a concrete stochastic kernel and Q an abstract stochastic kernel. Fix the policy and environment. Define the aggregated row a_s(c)=Σ_{u:q(u)=c}P(s,u). Assume 0≤epsilon≤1 and a uniform bound TV(a_s,Q(q(s),·))≤epsilon for every covered concrete state.

For any two probability vectors p and r, put d_i=min(p_i,r_i), D=Σ_i d_i and R=1-D. If R>0, define

C(i,j)=I[i=j]d_i+(p_i-d_i)(r_j-d_j)/R.

The residual vectors are nonnegative, each sums to R, and their corresponding coordinates have zero product. Consequently C is nonnegative, its row sums are p, its column sums are r and its diagonal mass is D=1-TV(p,r). If R=0, both residual vectors vanish coordinatewise, p=r and the diagonal coupling suffices. `formal/FiniteCoupling.lean` checks this existence statement over any finite index type and real-valued normalized nonnegative vectors, including the zero-residual case.

## Lifting a block coupling to concrete successors

At a matched current pair (s,b) with q(s)=b, take a maximal coupling C of a_s and Q(b,·). Define joint successor weights

G((s,b),(u,d))=C(q(u),d)P(s,u)/a_s(q(u))

when the denominator is positive, and zero otherwise.

If a_s(c)=0, nonnegativity implies P(s,u)=0 for every u in class c and C(c,d)=0 for every d. Thus omitting those terms loses no mass. For a supported class, summing G over d gives P(s,u) by the row marginal of C. Summing first over concrete u within each class gives C(c,d), and summing over classes gives Q(b,d). The block-match probability is exactly the diagonal mass of C, hence at least 1-epsilon.

At an unmatched pair choose G((s,b),(u,d))=P(s,u)Q(b,d). This also preserves both successor marginals; no match guarantee is needed after a previous mismatch.

## Joint paths and the horizon bound

Choose an initial joint law on S×B with concrete marginal mu, abstract marginal nu and matched mass at least 1-eta, with 0≤eta≤1. Exact pushforward initialization admits eta=0. Form finite joint paths by multiplying the initial weight and successive G kernels. Row normalization implies total path weight one.

Summing out the abstract path leaves the original concrete path law: at each step the concrete successor marginal is P(s,·), independent of the abstract current coordinate. Summing out the concrete path gives the analogous Q path law. These are path-law identities, not merely agreement of individual-time marginals.

Let m_n be the joint probability that the blocks match at every time 0 through n. Summing over matched prefixes and using the uniform successor match bound yields m_(n+1)≥(1-epsilon)m_n. No independence between mismatch events is assumed. `formal/BehavioralBounds.lean` checks by induction that m_H≥(1-eta)(1-epsilon)^H.

On matched paths, label preservation makes the two observed label sequences identical. Applying the same Boolean observable event to those sequences gives identical indicators. `formal/WeightedEvents.lean` checks that the absolute difference of their weighted event probabilities is at most the unmatched mass. Together these facts give

|P_M(E)-P_Q(E)|≤1-(1-eta)(1-epsilon)^H.

The finite environment family is covered by taking the certificate radius uniformly across its fixed-environment kernels. This argument does not silently extend to arbitrary environment/policy changes or nested branching-time probability tests. Any such extension needs its own joint process and uniformity argument.

## Scope and numerical connection

The analytical construction uses exact stochastic kernels. Machine certificates apply it to explicitly normalized exact binary-rational interpretations of serialized weights, with directed probability enclosures and the actual quotient radius. This document does not prove every numerical implementation branch or the statistical membership of an unknown human process.

The existing finite-horizon coupling foundation is not a new theorem. The operational partition-to-quotient condition is what the implementation certifies. An event-probability bound does not automatically establish a noninferiority margin in negative log-likelihood, whose behavior near zero requires separate control. Human predictive noninferiority is evaluated empirically on held-out observations.

The combined integration remains independently unreviewed and is not advertised as one end-to-end Lean formalization. Pinned compiler outputs, proof sources, retained failed attempts and invalid-proof controls are separate auditable artifacts.
