# Figure 04. Probability bounds under transition uncertainty

Panel A plots completion-reachability interval widths at four transition horizons. Each value first averages over active concrete states within a world, then equally over the 100 worlds of that state-space size. The two methods use the same transition samples. Lines show means, not confidence bands; paired world-bootstrap width comparisons are retained in the campaign summary.

Panel B shows a fixed trust-family example, seed 20000, at 128 concrete states and 10,000 transitions per row. Bars are robust model intervals, diamonds are nominal quotient estimates, and vertical ticks mark midpoints of exact integer enclosures for known generating-model probabilities. The rectangular uncertainty envelope already includes hidden-state aggregation, so no separate nominal abstraction penalty is added.

The Hoeffding construction has an exact tail certificate under the stated fixed-iid-row experiment. Clopper-Pearson beta quantiles remain floating point. Wide intervals are a measured limitation, particularly at long horizons. These are synthetic model results and provide no unconditional guarantee about human behavior.
