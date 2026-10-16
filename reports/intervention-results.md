# Full synthetic intervention results

Status: complete 400-world execution, structural/model-binding audit, seed-cluster analysis and measured Figure 05. Human intervention transfer remains unevaluated. The artifact audit reconstructs means and checks retained model/certificate bindings; integer bounds are checked during generation, rather than independently re-proved by this structural audit.

The benchmark uses five mechanism families, sizes 32 and 128, ten shared seeds, three response-changing surfaces and one structural negative control. Each method receives eight full-kernel queries. Six partition controls share the same observations within each method. Sixteen held-out coordinates and horizons 5, 10, 25 and 50 remain separate dependent evaluations.

## Horizon-10 findings

Targeted refinement improves held-out prediction compared with a random nested split at exactly the same state count and with the same queried kernels. This establishes a synthetic distinction-selection benefit relative to that control. It does not establish that adaptive environment acquisition is better than random environment acquisition.

| Environment search | Mean targeted error | Targeted minus matched random-split error | 98.75% seed-cluster interval |
| --- | ---: | ---: | --- |
| random | 0.137385 | -0.016933 | [-0.019235, -0.013972] |
| maximin_grid | 0.140679 | -0.018347 | [-0.020147, -0.016432] |
| rbf_ucb | 0.149882 | -0.015356 | [-0.017304, -0.013360] |
| local_search | 0.151476 | -0.013412 | [-0.015039, -0.011889] |

Error is the mean of the maximum absolute outcome-probability error over states and outcome categories within a held-out environment. Values are probabilities, not conversion rates or human prediction losses. World/environment errors are averaged within each seed. Ten shared seed clusters determine the reported bootstrap intervals; the four matched-control comparisons use Bonferroni-adjusted interval levels. These limited-cluster summaries are conditional on this generator and fixed catalogue.

The frozen bundled primary compares RBF acquisition with targeted refinement against random search with its random-capacity split. Its mean difference is -0.004436, with a 95% seed-cluster interval [-0.007354, -0.001434]. Capacity can differ across those two strategies. This positive bundled result must not be presented as isolated evidence for active-search superiority.

With targeted refinement in both strategies, RBF search has 1.250 percentage points more mean error than random search and 0.920 points more than the maximin grid. These are descriptive full-frontier comparisons; state counts can differ slightly across searches.

Selecting the maximum-disagreement witness also provides little additional benefit over choosing a random already-observed witness in this suite: mean error differences are within 0.000080 across the four search methods. The full-state identity and all-observed controls show that environment-response learning error dominates once the partition is extensively refined. The fixed witness example grows from 32 to 122 blocks in a 128-state world, so improved preservation is accompanied by substantial state growth. This experiment does not demonstrate that a highly compressed representation remains sufficient across all interventions.

All 100 structural negative-control worlds preserve observable behavior within the 1e-12 implementation tolerance; the largest reported world-level mean discrepancy is 8.27e-16. Across the full matrix, 25,600 estimated-model abstraction certificates are generated and input-bound by the audit. These certificates compare an estimated full kernel with its quotient. They do not certify the response-surface learner against the true held-out environment.

## Measured artifacts

[Figure 05](figures/figure-05/figure05-a1131dd6a645/figure-05-interventions.pdf) contains the fixed witness, matched-capacity intervals, and every search/control combination. Its source CSVs, caption, rendering code, hashes and visual-QA record accompany the PDF. See [the analysis contract](../spec/synthetic-intervention-analysis.md), [the complete analysis](preflight/intervention-main-analysis.json), and [the execution completion record](preflight/intervention-main-completion.json).

The supported conclusion is narrower than the original ambition: refinement can preserve distinctions that matter under synthetic intervention changes, while the tested adaptive search does not outperform stronger random/grid strategies. No human-fidelity, causal-effect or arbitrary-interface-transfer claim follows.
