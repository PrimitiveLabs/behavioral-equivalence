# Controlled environment-diversity ablation

Status: full exploratory extension complete, with persisted-kernel arithmetic replay. No human finding.

Four hundred synthetic response worlds cover five mechanism families, two sizes, ten shared random-seed clusters and four response surfaces. Each world receives three training spans: narrow, medium and broad. Every span makes exactly eight distinct nonneutral queries and shares the same neutral model and sixteen held-out evaluation points. A fixed geometric maximin selector determines coverage; this is not a new adaptive-search result.

The comparison retains unchanged, targeted, random-capacity, all-observed and identity controls. The complete output has 96,000 metric rows. All 19,200 targeted estimated-model certificates were reconstructed from persisted model arrays and replayed with the existing integer verifier. The certificate scope is the estimated interpolated kernel, not the error from learning the unseen environment response.

The primary descriptive coverage contrast uses the identity control, keeping every original state. Across the 300 response-changing worlds, broad minus narrow coverage changes mean absolute initial completion-probability prediction error by **-0.012345**, with descriptive 95% seed-cluster bootstrap interval **[-0.013800, -0.010859]**. This is about a **1.23 percentage-point reduction in prediction error**, not a change in completion rate. Ten shared seed clusters, rather than worlds, states or held-out coordinates, determine resampling uncertainty.

With broad coverage held fixed, targeted minus capacity-matched random refinement changes the same error metric by -0.000638, with descriptive interval [-0.000906, -0.000355]. The two contrasts answer different questions. Cross-span targeted models can have different capacities, while the identity coverage contrast isolates environment-learning effects and the within-span targeted/random comparison matches capacity.

All 100 response-invariant negative-control worlds remain stable: the maximum mean prediction error is approximately 4.75e-16. Every planned surface/control contrast is retained in the raw summary, including all four error metrics and per-seed effects. No favorable surface is selected for the headline contrast.

This extension was designed after the original synthetic intervention results. Its intervals are exploratory descriptions, not new confirmatory discoveries or distribution-free guarantees. It is conditional on the generator and fixed held-out catalogue. It does not overturn the earlier finding that the tested adaptive RBF search underperformed random/grid search, and it supplies no human intervention-transfer evidence.

Generation source: `a750f43e1a3a284a1e83aed6539a4d299f6798a20139b483d13f99c0c2966b77`.

Terminal campaign manifest SHA256: `1bfde5274eb6c2b6f006343576872b0f447f768e6546c9198fb0aae222d9d6cd`.

Replay/analysis source: `4baf4ad60c79b8baac10474dbeb1d2a4424b5894a97e70446513dfc3ad6f40ab`.

Complete audit-manifest SHA256: `3d2356297634137c96b0a33b4d9c7f5054bd39c438cab13c7d65bd887c8cb7cd`.

Summary SHA256: `15fe472035a8f7002b1d32cfd3d557af9b9c5c266229f6f5a4d9d94f9d322d58`.
