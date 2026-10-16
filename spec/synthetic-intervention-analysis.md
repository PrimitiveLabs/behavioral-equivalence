# Multidimensional synthetic intervention experiment

Status: source and experiment design frozen in `configs/intervention-main-v1.json`. The two-world development smoke uses seed 7, outside the confirmatory seeds 40000 through 40009. This analysis detail records statistical interpretation before aggregate confirmatory results are examined. It does not replace the separate human intervention contract.

## Design and query information

The experiment contains five base mechanism families, sizes 32 and 128, ten seed values, and four response surfaces: linear, localized, oscillatory and a structural negative control. These are 400 environment-response worlds built on 100 base mechanism/size/seed combinations. Their structure is synthetic. The localized and oscillatory families remove the monotone endpoint advantage built into the earlier scalar pilot.

Each search receives eight distinct queries from the same fixed 64-point two-dimensional catalogue. Each query reveals the full synthetic transition kernel. The witness score is the maximum Jensen-Shannon divergence between future three-category outcome distributions of states in an original equivalence block at ten decisions. This rich oracle is a controlled synthetic experimental-design setting, not the information available from eight human observations.

Compare random search, an outcome-independent maximin grid, an RBF upper-confidence acquisition rule, and local search with periodic maximin exploration. Adaptive rules use coordinates and previously queried scores only. The RBF acquisition rule is a search heuristic, not a certified statistical confidence bound. Report actual evaluation counts, searched coordinates, selected witness states, runtime and discovered disagreement.

Six partition controls share the same queried kernels within each search method: unchanged, targeted witness, a uniformly selected observed witness, a random nested refinement at exactly the targeted partition's state count, refinement using all observed environments, and the singleton identity partition. The random-witness partition can have a different capacity, which must remain visible. The all-observed comparator uses all information already available in the common query budget. There is no sampled-data advantage because every query reveals a kernel without sampling error.

## Prediction and mathematical scope

Fit the same convex radial interpolator to the neutral training kernel and eight queried kernels. Its bandwidth is fixed at 0.5. It predicts a full transition kernel at each of sixteen fixed held-out coordinates. No held-out kernel enters search, refinement or interpolation. Held-out truth enters scoring only. The evaluation catalogue is fixed and shared; results condition on it.

For each horizon 5, 10, 25 and 50, separately report the maximum observable-outcome probability error due to environment prediction, the discrepancy introduced by quotienting the estimated full kernel, and total prediction error against the held-out true kernel. Also report initial-distribution completion error. Errors are measured over complete, disengage and still-active outcomes. They are not human likelihood scores.

Each targeted-witness quotient receives an exact binary-rational integer abstraction certificate at every held-out estimated kernel. The certificate compares that estimated full kernel with its quotient. It does not cover the environment-response learner's error. Witness/training equivalence does not imply uniform equivalence across all held-out environments. Record the actual held-out radius rather than reusing the witness tolerance. Random-capacity and other controls receive numerical property comparisons; they are not relabeled as certified targeted models.

The negative control changes internal transitions within known exact blocks. It preserves the label-level process while providing a nontrivial internal change. Its construction requires the unperturbed exactly lumpable source. An error above 1e-12 fails the implementation check. Report all observed discrepancies, even below that tolerance.

## Comparisons and uncertainty

The frozen primary comparison is the paired difference in held-out total maximum prediction error at horizon 10 between RBF search with targeted refinement and random search with its random-capacity control. Average first across the sixteen held-out coordinates within a world, then across the three nonnegative-control surfaces and the mechanism/size cells. Record both models' state counts. Capacity can differ across those two search methods; this bundled comparison cannot isolate search quality or targeted splitting alone.

The capacity-controlled comparisons pair targeted and random-capacity refinements within each of the four search methods. They use exactly the same observations, environment learner and reduced state count. The unchanged, random-witness, all-observed and identity results provide the complete decomposition and compression frontier. Report every method and control; do not select a favorable surface or seed.

Shared seeds induce dependence across sizes, families and surfaces. For descriptive uncertainty, average paired differences within each seed across all included cells, then use a paired seed-cluster bootstrap with 5,000 replications and bootstrap seed 20261015. This provides only ten seed clusters. Do not count 300 positive-surface worlds, 400 worlds, sixteen coordinates, four horizons, outcome categories or state comparisons as independent replicates. Report the ten seed-level effects alongside intervals. These intervals describe variation under this generator and fixed environment catalogue, not worst-case guarantees.

Report the four capacity-controlled comparisons with 98.75% bootstrap intervals as a Bonferroni family of four and the frozen bundled primary comparison with a 95% interval. Bootstrap endpoints are numerical descriptive sampling summaries with limited-cluster uncertainty, not distribution-free finite-sample confidence certificates. Horizon and surface breakdowns are descriptive. The structural negative control is reported separately from the three response-changing surfaces.

## Passing and release criteria

Completion requires every frozen world, its searched coordinates and partitions, all retained interpolation inputs, held-out truth, exact certificates, property errors, and failure records. A positive scientific result additionally requires a supported advantage over the prespecified controls. A null or negative result completes the experiment when faithfully reported. This synthetic result never satisfies the human generalization criterion by itself.
