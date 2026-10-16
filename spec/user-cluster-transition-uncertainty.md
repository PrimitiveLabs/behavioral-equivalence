# User-cluster transition uncertainty

Status: implementation and focused tests complete; human-data application and independent review pending.

For a fixed representation, include every sampled user, including users with no visits to a particular state. Let C_i(r,j) be a user's successor count and D_i(r) its row total. Require an outcome-blind, prespecified cap B on the user's entire count table. The estimand is p(r,j) = E[C_i(r,j)] / E[D_i(r)], provided the denominator is positive. It weights users according to expected row visits; it is not the equal-user mean of individual transition probabilities.

Assume independent user tables from the target population. Within-user dependence is unrestricted. A representation learned from the same users does not automatically satisfy the fixed-representation assumption, even when their calibration events occur later. Common calendar shocks can also invalidate user independence. These limitations must accompany real-data results.

## Soundness argument

Normalize each coordinate and row total by B. All lie in [0,1]. For N independent users and m = rows times (successors + 1) coordinates, Hoeffding and a union bound give simultaneous absolute mean error at most t with failure probability at most 2m exp(-2Nt²). The implementation chooses a dyadic t and verifies exp(2Nt²) >= 2m/alpha using a lower Taylor sum with exact rational arithmetic. If t reaches one, the deterministic [0,1] range suffices.

On that simultaneous event, each numerator mean lies in [a,b] and its denominator in [c,d]. Since the true numerator is nonnegative and at most the denominator, its ratio lies in [a/d, min(1,b/c)] when c>0. If c=0 the upper endpoint is one. Float endpoints are rounded outward from exact rational ratios. The true pooled row is therefore in the returned coordinate box intersected with the simplex. Row-coordinate dependencies are relaxed, making subsequent rectangular robust verification conservative for this box.

Zero expected denominator means the ratio is undefined. Such rows are assigned [0,1] coordinate intervals rather than an invented transition law. A model-checking use must document how unreachable or unsupported states are completed.

## Interpretation boundary

These confidence sets do not establish that a finite learned state is Markov, that human dynamics are stationary, or that future users share the calibration population. Propagating their intervals through a model checker produces conditional model-level bounds. Held-out human evaluation must separately use cohort mean compatibility diagnostics, horizon-specific censoring definitions, temporal-shift checks, and explicit unsupported-state reporting. Binary outcome containment is not probability-confidence coverage.

The cap changes precision and potentially the estimand. Do not divide the radius by the number of interactions or remove zero-visit users after observing support. Counts, cap, coordinate family, representation identity and sampling population must be frozen before confirmatory evaluation.

## Operational model compilation

`cluster_ratio_from_totals` uses only the coordinate sums and declared user count. This avoids a dense N by K by K tensor. The upstream stream processor must independently verify every user's cap, cohort membership and unique bucket assignment; those cannot be inferred from aggregate counts. Python-integer summation prevents overflow.

`pooled_decision_intervals` accepts K by 2 by K successor counts and constructs both emission and outcome-conditional transition intervals. Each family receives alpha/2. The union therefore covers both families at total failure probability at most alpha under the assumptions above. The resulting 3K-stage interval chain uses one step to emit and another to advance. These emissions estimate pooled calibration counts and explicitly differ from frozen training-teacher emissions. Comparing this interval model against a teacher-emission point model would conflate two estimands and is prohibited without a separate discrepancy analysis.

Focused tests include perfect within-user dependence, zero-visit users, sparse/vacuous rows, scale invariance under repeated identical within-user counts, aggregate/full-table agreement, invalid-cap rejection, lifted nominal-kernel inclusion and robust reachability enclosure through 99 graph steps. These are implementation checks, not a substitute for independent mathematical review or human-data evaluation.
