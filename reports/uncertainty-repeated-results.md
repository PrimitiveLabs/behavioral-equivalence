# Repeated fixed-world uncertainty results

Status: all 4,000 calibration datasets and 16,000 paired method/support evaluations complete and replayed from retained evidence. The replay regenerates draws, confidence sets, exact-integer model certificates and reported metrics using the original arithmetic implementation. It is not an independent numerical proof implementation. Nominal-prediction and broader uncertainty ablations remain part of the integrated benchmark.

Twenty fixed worlds cover five families, two sizes and two seeds. Each has two fixed allocation designs and 100 independently drawn calibration datasets per design. Both methods and both terminal-support controls share each dataset. The 16,000 evaluations are therefore not 16,000 independent coverage trials. The primary row budget is 1,000 observations per active state on average; the imbalanced design leaves one quarter of active rows unsupported. Both designs draw no terminal-row samples. The known-closure control supplies terminal specification information separately. The earlier 200-world study samples every concrete row, so its widths are not a matched comparison with this extension.

## Containment and precision

All 4,000 datasets for every method/support control contain the known tested outcome probabilities. Hoeffding sets contain all 4,000 generating kernels; floating-quantile Clopper–Pearson contains 3,970. The thirty kernel exclusions do not cause tested property exclusions.

The resulting intervals are often too wide to be useful. At horizon 10, mean completion widths over active states, then over the fixed worlds/repetitions, are:

| Confidence construction | Terminal information | Uniform allocation | Imbalanced allocation |
| --- | --- | ---: | ---: |
| certified_hoeffding | unknown_terminal_rows | 0.9996 | 0.9984 |
| certified_hoeffding | known_closed_terminal_labels | 0.9980 | 0.9892 |
| clopper_pearson_float_quantiles | unknown_terminal_rows | 0.9078 | 0.9725 |
| clopper_pearson_float_quantiles | known_closed_terminal_labels | 0.7833 | 0.9175 |

A width of one covers the whole probability range. Knowing that terminal labels form closed classes helps, especially for Clopper–Pearson under uniform allocation, but does not make these intervals narrow. Unsupported active states and the conservative row-rectangular hidden-state envelope remain material limitations. The control supplies extra true specification information; it does not infer absorbing human outcomes from observed zeros.

## Statistical and numerical scope

The full report includes 320 simultaneous Hoeffding/Bonferroni Monte Carlo intervals across two coverage metrics in all method/support/design cells. Each fixed-world/design cell contains 100 independent calibration datasets. These intervals quantify finite Monte Carlo error conditional on those fixed worlds and allocation schedules. They do not prove worst-case coverage or generalize to human trajectories. Totals above are descriptive sums across paired cells; methods/support variants are not pooled as independent trials.

The exact Hoeffding tail construction assumes ideal fixed-design iid categorical row samples from the specified kernel. Pseudorandom Monte Carlo execution tests the implementation; it does not formally verify the random-number generator. The Clopper–Pearson implementation uses floating beta quantiles without an outward statistical-tail certificate. Directed integer outcome bounds have their own conditional model-level arithmetic guarantee.

The substantive result is a precision failure alongside complete observed containment, rather than merely a positive coverage count. These data do not establish useful uncertainty intervals for real human behavior.

Sources: [full cell-level report](preflight/uncertainty-repeated-summary.json), [replay completion](preflight/uncertainty-repeated-replay-completion.json), and [frozen protocol](../configs/uncertainty-repeated-v1.json). The earlier 200-world sample-size study and Figure 04 remain a separately counted experiment.

The secondary [point-estimate ablation](nominal-ablation-results.md) is now complete. It reuses all 4,000 count matrices, retains the same abstraction envelope and reports both containment and paired width differences. Its design followed inspection of the original uncertainty results.
