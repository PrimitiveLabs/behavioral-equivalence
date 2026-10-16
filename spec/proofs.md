# Model specification and proof obligations

Version 0.1. Status: analytical proof argument with separately checked Lean components and exact numerical certificate constructions described below. An implementation-author soundness review is recorded in reports/mathematical-soundness-review.md. Independent peer review and a single end-to-end mechanization have not been performed. Early floating-point pilot checks remain engineering evidence.

## Finite chain and observable events

For each fixed environment e, M_e=(S,P_e,L,mu) is a finite Markov chain. P_e has nonnegative entries and every row sums exactly to one in the mathematical model. L maps states to an observable finite alphabet. The policy and environment are fixed throughout a trajectory unless a different, explicitly augmented process is specified. An H-transition event is measurable with respect to labels at times 0 through H. An arbitrary property of hidden states is outside this event language.

Let q:S->B be a surjective partition map. Each block preserves L. For each environment define the concrete aggregated row p_e^q(b|s)=sum_{u:q(u)=b}P_e(s,u). Choose a stochastic abstract row Q_e(.|b). The implementation uses the uniform arithmetic mean of concrete aggregated rows in each block. The abstract initial law is q_pushforward(mu).

The certified one-step error is epsilon=max_{e,s} TV(p_e^q(.|s),Q_e(.|q(s))). It applies to the actual abstract row, not merely to neighboring embedding vectors or selected pairs. Equality epsilon=0 establishes a lumpable quotient for this family. A common label-preserving stable partition under all environments is an exact probabilistic bisimulation partition.

## Theorem 1. Finite-horizon label-event preservation

Assume a joint initial distribution for X_0 and B_0 with Pr(q(X_0)!=B_0)<=eta. Assume the preceding one-step bound with epsilon in [0,1]. For every common observable event E over H transitions,

|Pr_M(E)-Pr_Q(E)| <= 1-(1-eta)(1-epsilon)^H.

Proof. Conditional on a matched pair q(X_t)=B_t, maximally couple the finite distributions p_e^q(.|X_t) and Q_e(.|B_t). The chance that their next blocks match is at least 1-epsilon by the total variation coupling identity. Given the concrete successor block, draw the concrete successor from its conditional transition distribution; zero-mass blocks can have any conditional distribution since the coupling never selects them. This preserves both transition marginals. Conditional on all previous matches, the next match probability is at least 1-epsilon. Induction gives a probability of no mismatch through time H at least (1-eta)(1-epsilon)^H. Block labels agree along every matched path. Hence event indicators can disagree only if some mismatch occurs. Taking the expected absolute indicator difference bounds the absolute difference of probabilities. This proves the result.

With the exact pushed-forward initial law eta=0. The union bound gives min(1,eta+H*epsilon), which can be weaker. State-dependent or time-dependent bounds can use a product of per-step bounds only if each is uniformly valid conditional on past matched paths. Averaging epsilon over visited states without such an argument is insufficient.

Corollary. For epsilon=eta=0, all bounded observable path events have identical probabilities. This proof does not establish arbitrary reward or unbounded-time guarantees. It does not establish human Markov sufficiency.

## Theorem 2. Conditional uncertainty composition

Let U be a confidence set of possible concrete kernels on a frozen labeled state space. For every P in U suppose a corresponding Q(P) belongs to an abstract uncertainty family V and satisfies Theorem 1 with uniform epsilon and eta. If robust abstract verification bounds the event probability for every Q in V by [l,u], then every P in U satisfies

Pr_P(E) in [max(0,l-delta),min(1,u+delta)], delta=1-(1-eta)(1-epsilon)^H.

Proof. Fix any P in U, apply Theorem 1 to P,Q(P), and use l<=Pr_Q(P)(E)<=u. The resulting statement holds uniformly over U. If Pr(P_true in U)>=1-alpha under an explicitly stated sampling experiment, the same interval contains the true model probability with probability at least 1-alpha. Dependence between set construction and abstraction is allowed only if the asserted uniform event remains valid. A certificate calculated at one nominal kernel does not supply that uniform statement.

A row-rectangular per-step adversarial abstract family includes stationary kernels with each row inside the permitted simplex. Its extremal finite-horizon probabilities therefore conservatively bound that stationary family. The resulting extrema need not be attainable by a single fixed kernel, especially when the best row choice changes by time. State this conservatism in every report.

## Theorem 3. Fixed-design multinomial confidence set

For n fixed states, row s has m_s independent categorical successor samples from its fixed transition law. States, representation, and sample counts are fixed independently of these samples. Let p_hat_sj be the empirical proportion and r_s=sqrt(log(2*n*n/alpha)/(2*m_s)) for m_s>0. Intersect coordinate intervals [max(0,p_hat_sj-r_s),min(1,p_hat_sj+r_s)] with the stochastic simplex. For m_s=0 use the whole simplex. This set contains all true rows simultaneously with probability at least 1-alpha.

Proof. Each coordinate count is an average of independent Bernoulli indicators. The two-sided Hoeffding inequality bounds a coordinate failure by alpha/(n*n). Apply the union bound over all n*n coordinates. No independence across coordinates or rows is needed for that union bound. The true rows are stochastic, so intersecting with the simplex cannot remove them on the simultaneous event. Zero-sample rows impose no restriction.

This proof does not justify the same intervals after adaptive state selection, outcome-dependent sample sizes, nonstationarity, or dependent human trajectories. Those cases require a different statistical argument or an explicitly empirical diagnostic.

## Robust reachability recurrence

At H=0, v_0(s)=1 if target label holds, else zero. Target states retain value one. For other states, v_(h+1)^min(s)=min_{p in row simplex}p dot v_h^min, with the analogous maximum. Rows are independent and choices may change with the remaining horizon. The row linear program is solved by starting at lower bounds and greedily distributing residual mass to the lowest (or highest) successor values, subject to upper bounds. An exchange argument proves optimality: allocating mass to a worse-valued destination while a better-valued destination has capacity cannot be optimal. Independent scipy linear programming tests and PRISM interval-model comparisons validate implementation.

## Numerical and review obligations

Binary rational small fixtures allow exact Fraction arithmetic. General floating sums, mean rows, and PRISM output are not exact proofs. The pilot reports numerical tolerance and does not claim a directed-rounding guarantee. Before publishing sound machine certificates, either compute rational certificate quantities and normalized rational kernels, or implement outward rounding/error analysis covering serialization, stochastic normalization, robust extrema and horizon propagation. Numerical allowance cannot be selected solely because observed errors fit inside it.

Remaining independent review: event-language scope, uncertainty-to-abstraction mapping, simultaneous coverage under the final sampling design, quotient label augmentation for real data, and all numerical enclosures. Existing theory underlies these results; novelty is not claimed for the coupling argument.

## Exact numeric implementation added after the initial pilot

`rational.certify` defines its kernel by taking each serialized binary float as an exact rational weight and normalizing each row using rational arithmetic. The quotient mean, epsilon, finite-horizon delta, event probabilities and interval endpoints all use Fraction arithmetic. `rational.verify` recomputes the full certificate from the chain and rejects altered output. Exact and perturbed 16-state fixtures are recorded under `data/synthetic/fixtures`. These establish numeric certificates for those specified normalized kernels. They do not retroactively establish directed-rounding soundness for the separate floating-point cloud pilot or uniform confidence-set abstraction certificates.

## Conditional properties require denominator control

The path-event coupling bound applies directly to joint events A intersect B and conditioning events B, rather than automatically to their probability ratio. If the abstract conditioning probability is at least m and both event discrepancies are bounded by delta<m, the concrete conditioning probability is at least m-delta. A conservative conditional-probability discrepancy is min(1,2*delta/(m-delta)): expand the ratio difference into a numerator term plus a denominator term and use that the abstract joint probability is at most its conditioning probability. Rare conditioning events can therefore make the bound vacuous.

Under uncertainty, separately valid lower/upper enclosures for the joint and conditioning probabilities yield a conservative ratio enclosure [lower_joint/upper_conditioning, upper_joint/lower_conditioning], clipped to [0,1], only if the lower conditioning bound is positive. Adjust the unconditional enclosures for abstraction before forming this ratio. These separate extrema may be conservative because the same kernel governs both. Zero-probability conditioning remains undefined and must be reported rather than replaced with zero. An augmented-state reachability encoding does not remove the need to specify conditioning semantics.

## Alternative finite-sample coordinate sets

`uncertainty.clopper_pearson_intervals` inverts two-sided binomial tests for each successor count, with tail probability alpha/(2*n*n). Each coordinate's interval has failure probability at most alpha/(n*n) under fixed-count iid row sampling. The union bound covers all coordinates simultaneously, without requiring independence between multinomial coordinates. Intersecting with the stochastic simplex preserves inclusion on that event. This can be narrower than Hoeffding, particularly near zero probabilities. Numerical beta quantiles are not an outward-rounding proof; the mathematical coverage construction and its floating implementation remain distinct. No dependent-human-history coverage claim follows automatically.

## Directed integer numerical certificates

The `integer_bounds` module interprets each binary64 input as an exact rational weight and normalizes nominal rows exactly, using integers. It computes the actual total variation radius to the supplied quotient kernel, rather than trusting a rounded radius from partition selection. Integer ratios encode epsilon and each finite-horizon delta exactly. The certificate binds the concrete and abstract digests, partition, property labels and horizon.

For numerical probability enclosures, let D=2^b. Store integer endpoints l_h,u_h with l_h/D <= v_h <= u_h/D. At a target state both endpoints equal D. At other states, replace the exact weighted average of the previous lower endpoints by its integer floor, and replace the average of upper endpoints by its integer ceiling. Monotonicity of a stochastic row proves the enclosure by induction, starting from the exact indicator vector. Each step adds at most 1/D to each one-sided numerical error, so interval width is at most 2H/D. An actual transition kernel, not decimal display values, defines the certified probability. PRISM numerical agreement remains a separate independent implementation check.

For interval rows, all binary64 lower/upper endpoints define exact rational constraints, without normalizing interval endpoints. Reject a row when its exact lower sum exceeds one or its exact upper sum is below one, even if a floating sum passes tolerance. Greedy allocation solves each finite rational row linear program exactly. Apply it to lower/upper integer successor endpoints, with directed integer division, to enclose each robust minimum and maximum. This proves numerical containment for the per-step rectangular semantics; confidence-set inclusion still requires its own statistical assumptions. In particular, this numerical proof does not certify floating beta quantiles.

The post-auditor reconstructs each quotient from a saved world and partition, requires byte-for-byte equality with the PRISM model text, and recomputes the integer certificate. Its machine verification checks its implementation and certificate contents. It is not an independent proof-assistant proof of the coupling theorem or the auditor. Certificates cover all related deterministic initial-state pairs. Combining arbitrary initial laws requires the coupling stated in Theorem 1.

For the conditional event used by E1, B means an active state at time one. Its probability discrepancy is at most epsilon, while the joint completion-within-H and B discrepancy is at most delta_H. If d_Q=Pr_Q(B)>epsilon, expanding the ratio difference yields the conservative conditional error min(1,(delta_H+epsilon)/(d_Q-epsilon)). Otherwise return a vacuous interval where the concrete conditional is defined. Zero concrete conditioning mass means the conditional is undefined. The integer implementation encloses the abstract conditional expectation using its exact restricted successor weights. Survival means avoiding either terminal label through H, so use the complement of reachability of their union.

## Uniform uncertainty enclosure through abstraction

A separate construction avoids applying a nominal abstraction radius to an entire uncertainty family. For concrete interval row s and target block C, define

l_sC = max(sum_{j in C} l_sj, 1-sum_{j outside C} u_sj),
u_sC = min(sum_{j in C} u_sj, 1-sum_{j outside C} l_sj).

These are the exact extrema of the block mass over the feasible row simplex. Feasibility supplies enough capacity inside and outside C to attain the extrema. For an abstract origin block B, take lower_BC=min_{s in B} l_sC and upper_BC=max_{s in B} u_sC, intersected with the abstract row simplex. Exact rational arithmetic followed by outward binary64 rounding preserves every projected concrete row. The `uncertainty_abstraction` module records the exact hull and checks its recomputation.

Conditional on any concrete label history and present block B, the next-block law is a convex combination of projected rows of hidden concrete states in B. Every such row belongs to the rectangular abstract envelope, and convexity keeps that mixture inside the envelope. Induction constructs a history-dependent abstract adversary with the same labeled path distribution as the concrete process. Therefore robust abstract lower/upper reachability values enclose every feasible concrete kernel's probabilities, without a further nominal epsilon expansion. This includes hidden-state and transition uncertainty conservatively; it does not claim the envelope extrema arise from one stationary concrete kernel. Apply the fixed-design statistical coverage argument only when its conditions hold. Learned human histories do not automatically satisfy those conditions.

## Exact tail arithmetic for the Hoeffding construction

`confidence.certified_hoeffding` chooses a dyadic radius r_s and verifies the rational inequality sum_{k=0}^{100} (2*m_s*r_s^2)^k/k! >= 2*n^2/alpha. Each term is nonnegative, so this finite rational sum is a lower bound for exp(2*m_s*r_s^2). The verified inequality therefore implies 2*exp(-2*m_s*r_s^2) <= alpha/n^2, exactly the coordinate failure allocation in Theorem 3. Floating log and square root only propose the initial dyadic radius; exact arithmetic accepts it or increases it. A radius of one uses the whole simplex and needs no tail inequality. Each empirical proportion is rational and coordinate bounds round outward to binary64. Consequently numerical logarithm, square root and endpoint conversion do not weaken the stated mathematical coverage. This supplies a numerically certified baseline for fixed-design iid synthetic rows. It supplies no new justification for dependent human samples.

## Explicit finite-property certificates

`property_bounds` supports occupancy at a specified graph time, reachability by that time, and a threshold number of visits to a target label among times 0 through H. A label visit at a graph time is the counting unit. For the three-stage decision lift, a positive outcome appears once per decision, at its outcome stage. Certificates list each event, horizon, target labels, count threshold and permitted concrete prediction origins. These are weaker claims than probabilistic bisimulation and do not cover unlisted events or environments.

For occupancy, initialize v_0(s) to the target-label indicator and update v_(h+1)(s)=sum_j P(s,j)*v_h(j), without making target states absorbing. For at-least-r visits, set v_h(s,0)=1 and v_0(s,r)=1 only if r=1 and s has the target label. Then v_(h+1)(s,r)=sum_j P(s,j)*v_h(j,r-I_target(s)), with the zero-threshold value fixed at one. Reachability is the r=1 case. The recurrence counts the label at the current state and the remaining labels on its successor path exactly once. The implementation shares dynamic-programming tables across requested thresholds and horizons.

Apply exact integer floor/ceiling after each stochastic weighted sum. Monotonicity gives lower/upper probability enclosures by induction. Given enclosures [p_lo,p_hi] and [q_lo,q_hi] for a related origin pair, the absolute discrepancy is at most max(p_hi-q_lo,q_hi-p_lo,0). Maximize this upper bound over the declared concrete origins. If the general coupling certificate also applies, the minimum of that direct upper bound and its coupling bound remains a valid bound. This can certify a specified property even when a small global total-variation bisimulation radius is unavailable.

`property_refinement` uses floating calculations to propose splits, then accepts a partition only after the integer certificate meets the selected probability-error budget. It keeps a common latent partition across all three observation stages and refines one nontrivial block on each rejected iteration. An identity quotient preserves every specified property, so refinement terminates; a zero error budget can require that exact identity argument rather than equality of rounded endpoint enclosures. The certificate checker recomputes the probability enclosures, relation, actual coupling radius and property errors from the input kernels. Independent path-enumeration tests cover occupancy and repeated outcomes. A regression example certifies first-step agreement while exposing a disagreement at a later step, demonstrating why a short or incomplete property suite cannot establish general equivalence.

The human pilot uses all pending states as prediction origins. A certificate for those origins does not automatically assert a bound from arbitrary intermediate states. No numerical model certificate establishes that the human sequence is Markov, that future environments belong to the tested family, or that the learned transition estimator is statistically adequate.

## Monte Carlo coverage uncertainty

The coverage-analysis sidecar treats a sampled transition dataset as one trial. A success can mean inclusion of the entire known kernel or containment of all tested state/property probabilities. These metrics and the two methods share trials; they are not independent extra observations. Conditional on the fixed worlds, separately drawn transition datasets are independent even when their success probabilities differ. Hoeffding's inequality for independent bounded observations gives an interval for the average success probability, with radius sqrt(log(2*K/alpha)/(2*N)); Bonferroni covers the K reported intervals without assuming independence across intervals. The sidecar reports 28 simultaneous intervals at family confidence 0.95. Their numerical evaluation is separate from the exact model-level probability certificates. They describe average coverage over the fixed simulation designs, not a worst-case guarantee across all models or any claim about humans.

## Distinguishing pairwise diameter from quotient radius

For a fixed proposed block, let p_s be its successor distribution over the final abstract blocks, and Q their arithmetic mean. If every pair satisfies TV(p_s,p_t)<=tau, convexity of total variation gives TV(p_s,Q)<=mean_t TV(p_s,p_t)<=tau. Conversely, a certified radius TV(p_s,Q)<=epsilon implies only the pairwise bound TV(p_s,p_t)<=2*epsilon by the triangle inequality. These tolerances are not interchangeable. Checking only selected neighbors does not prove either uniform condition, and chains of near neighbors need not satisfy the same diameter tolerance. Every accepted certificate must therefore recompute its radius against the actual quotient on the final partition. An iterative change to successor blocks requires recomputation.

The source comparison in [theory-comparison.md](theory-comparison.md) identifies the established finite-trace theorem behind the coupling argument. This is an internal comparison, not independent mathematical signoff. The stronger language of minimal predictive causal states or arbitrary branching-time preservation is outside the present proof claims.

## Neutral response witnesses and rechecking after a split

For two Bernoulli response laws with probabilities p and r, total variation is |p-r|. If one common abstract response q approximates both within epsilon, the triangle inequality implies |p-r| <= 2 epsilon. Thus an exact empirical count difference greater than 2 epsilon excludes every common response row at that tolerance for those two fitted count laws. The held-out response study computes this comparison as rational count ratios, avoiding a rounded threshold decision. It does not establish the same separation for unknown human response laws: finite counts, dependent histories, selection and estimation require separate empirical analysis.

For a fixed neutral response table and any finite partition, define each abstract response as its block arithmetic mean. A split can move that mean and increase the maximum within-block error, despite reducing the block's diameter. For example, the block {0, 0.001, 0.019, 0.02} has mean 0.01 and radius 0.01. Removing 0.019 leaves mean 0.007 and radius 0.013. Therefore the implementation rechecks the actual quotient after each intervention-driven split. On a violation it removes a worst-error state into a singleton and repeats. Each repair strictly increases the number of blocks, so at most n minus the initial block count repairs reach the identity partition, whose radius is zero. Every accepted final partition is checked against the exact-normalized serialized kernel. Termination does not imply optimal compression.

Capacity matching likewise accepts only neutral-certified partitions. Its random ordered cuts are a control construction, not uniform sampling from all partitions. Witness splits, neutral repairs and capacity fills are recorded separately. A shared state count alone does not show that the same evidence or the same partition was used.
