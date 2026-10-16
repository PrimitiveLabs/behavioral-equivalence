# Uniform one-outcome uncertainty and abstraction envelope

This is an elementary bound for the specified supplementary categorical model, not a new general theorem or a human coverage result. Implementation: tools/supplementary_intervals.py. The six tests enumerate feasible small binary kernels, cover point and vacuous families, and reject invalid intervals/partitions. They validate implementation examples, not universal mathematical soundness by enumeration.

Let each pending state s have an unknown categorical row p_s in the probability simplex, with coordinate bounds l_sj <= p_sj <= u_sj. The feasible family includes all simplex rows in these boxes. For a fixed partition block B, define the quotient row for each admissible parent kernel by q_B(P) = sum_{s in B} p_s / |B|. Thus the quotient itself varies with the unknown kernel.

Coordinatewise quotient intervals are a_Bj = mean_{s in B} l_sj and b_Bj = mean_{s in B} u_sj. Every admissible q_B(P) lies inside these intervals. For each s in B, p_sj - q_Bj lies between l_sj - b_Bj and u_sj - a_Bj. Consequently

TV(p_s,q_B(P)) <= (1/2) sum_j max(u_sj - a_Bj, b_Bj - l_sj).

Taking the maximum over all states and blocks, and clipping above at one, yields a single uniform error delta. The coordinate maximizers need not be jointly attainable; relaxing their dependence only enlarges this upper bound. The same delta applies to every feasible kernel, not just the nominal smoothed estimate.

For the event that the next outcome is category j, its abstract probability is bounded by [a_Bj,b_Bj]. The original pending state's probability therefore lies in [max(0,a_Bj-delta),min(1,b_Bj+delta)]. This also follows from the H1 coupling argument with matching terminal category labels. The implementation calculates all means, error bounds and widened endpoints from exact binary-rational interpretations of serialized input bounds, then rounds output lower endpoints down and upper endpoints/error bounds up. It does not infer a likelihood-loss bound from a probability-error bound.

A full box can make delta=1 and every expanded interval [0,1]. Report this as vacuity, not successful uncertainty calibration. Parent coordinate intervals may be tighter than expanded quotient intervals; retain both rather than hiding widening due to abstraction. A nominal smoothed forecast is a separate object and need not belong to a frequentist confidence set constructed around an unknown count-ratio parameter.

## Statistical membership and empirical estimands

The count-ratio construction bounds ratios of expected capped per-user class counts to expected row visits, under independent user tables, a common population and a representation treated as fixed. Every eligible user, including zero-visit users for each state, belongs in its denominator. Outcome-blind origins are capped at 64 before missing endpoint labels are removed. Missing endpoints condition their row estimand on recorded target availability and require explicit interpretation.

Alpha=1/6000 per endpoint/state-count/fit model allocates an overall .05 union budget across 10 fixed fits, 5 state counts and 6 endpoints. Within each model the existing count-ratio code allocates across all numerator and denominator coordinates. A union bound does not require the 300 models to be independent. The validity of each model's user-table assumptions is still required. Shared calendar shocks, known-user representation learning, stationarity and predictive-state sufficiency are not established by this construction.

Human test diagnostics compare paired equal-user observed cohort rates to mean probability endpoints with original-user resampling. They are compatibility diagnostics under declared sampling assumptions. They do not measure finite-sample confidence-set coverage by checking whether a single binary observation lies in a probability interval. The test equal-user cohort estimand and calibration count-ratio parameter are different; identifying them requires additional transport and model-validity assumptions. Their agreement or disagreement is empirical evidence, not a proof that those assumptions hold.

The endpoint abstraction is one observation followed by an absorbing category label. It must not be described as a verified sequence of later human decisions, or as a joint distribution across six endpoints. Existing click-trajectory experiments address a separate specified model and preserve their own limitations.
