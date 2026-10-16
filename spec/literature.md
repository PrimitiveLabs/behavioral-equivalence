# Primary-source map

Status: initial source map, full novelty review pending. Checked 2026-10-15.

Bian and Abate, On the Relationship between Bisimulation and Trace Equivalence in an Approximate Probabilistic Context (2017), arXiv:1701.04547v3, https://arxiv.org/abs/1701.04547 . Establishes the existing connection between approximate bisimulation and finite-horizon trace total variation. Our finite coupling argument specializes this kind of guarantee to a concrete partition and its actual abstract rows. No new theorem claim.

Zacchia Lun, Wheatley, D'Innocenzo, and Abate, Approximate Abstractions of Markov Chains with Interval Decision Processes (2018), https://arxiv.org/abs/1804.08554 . Existing abstraction construction and interval semantics directly overlap with the planned verification layer. A mean-row quotient alone is not differentiated research.

PRISM 4.11, https://github.com/prismmodelchecker/prism/releases/tag/v4.11 . Pin Linux x86 binary SHA256 6d4677b6ab64daa16a8bb5ad05f79c46313ad67835cb8445ee770e7f57f611c8. Official uncertainty syntax https://www.prismmodelchecker.org/manual/ThePRISMLanguage/UncertainModels and property semantics https://www.prismmodelchecker.org/manual/PropertySpecification/UncertainModels support interval DTMCs and robust min/max verification. Independent dynamic programming is an implementation check, not independent evidence of human fidelity.

Gao et al., KuaiRand: An Unbiased Sequential Recommendation Dataset with Randomly Exposed Videos (CIKM 2022), DOI 10.1145/3511808.3557624, https://github.com/chongminggao/KuaiRand . Sequential observations and randomized content exposure support recommendation-domain evaluation. They do not identify responses to arbitrary software redesigns.

Novelty candidates requiring comparisons: learning certificate-aware history partitions, informative uncertainty under sparse/dependent behavioral sampling, and capacity/data-matched counterexample refinement with real observed intervention transfer. Required additions: foundational probabilistic bisimulation, computational-mechanics predictive-state equivalence, predictive-state representations, adaptive confidence sequences, and active experimental design. Full-text theorem-assumption comparison remains open.

The focused [theory comparison](theory-comparison.md) checks the existing finite-horizon trace theorem and interval-abstraction overlap and adds computational-mechanics and predictive-state foundations. It records which full texts could be inspected and which retrievals failed. The general independent review and full novelty comparison remain open.

Follow-up full-text review retrieved the computational-mechanics paper from [arXiv](https://arxiv.org/pdf/cond-mat/9907176) and the PSR paper from the [official proceedings](https://papers.nips.cc/paper_files/paper/2001/file/1e4d36177d71bbb3558e43af9577d70e-Paper.pdf). The theory comparison now records the relevant stationarity, complete-history and all-controlled-test distinctions. It also adds classical probabilistic testing and [time-uniform confidence sequences](https://arxiv.org/html/1810.08240v4). This is a scoped related-work review, not an exhaustive novelty certification.
