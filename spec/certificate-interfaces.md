# Certificate interfaces and verification dispatch

The research produces several native certificate formats because their claims differ. Do not flatten them into one probability interval or infer claim strength from the word certificate. Artifact identity verification is necessary but separate from semantic verification. The source-only release retains native payloads and their source-bound verifiers.

| Native format | Required semantic inputs | Authoritative replay | Claim boundary |
| --- | --- | --- | --- |
| integer-abstraction-v1 | Concrete Chain, actual abstract Chain, complete partition, horizons, property catalog and fixed-point bits | integer_bounds.verify_abstraction | Exact-normalized one-step radius and finite-horizon bounds; deterministic related origins, initial mixtures require coupling |
| finite-property-certificate-v1 | Both Chains, partition, listed LabelEvents, deterministic concrete origins and bits | property_bounds.verify_properties | Only enumerated events/origins; not general bisimulation |
| integer-robust-v1 | IntervalChain, target, horizons and bits | Recompute integer_bounds.robust_enclosures | Directed enclosures for rectangular per-step extrema, conditional on kernel membership |
| uncertainty-envelope-v1 | Concrete IntervalChain and partition | uncertainty_abstraction.verify_envelope | Uniform label-path rectangular enclosure; abstraction already included |
| hoeffding-rational-tail-v1 | Integer transition counts, labels and declared construction settings | confidence.verify_hoeffding | Numerically certified fixed-design simultaneous coordinate set under stated iid assumptions |
| empirical-point-numeric-envelope-v1 | Integer counts, labels, fixed closure/rounding configuration | nominal_ablation.verify_empirical_point | Numerical enclosure for the estimated point model, not a statistical confidence set |
| rational schema 0.1 | Exact rational finite chain and supplied reachability query | rational.verify | Exact arithmetic on the specified rational fixture |

Function names and arguments are checked against the release source. A caller must supply original typed models and counts where required; a digest alone cannot recreate them. Verifiers recompute the native payload rather than trusting its claimed bound. Digests bind serialized model identities, while exact arithmetic interprets the documented normalized binary-rational kernel. The distinction remains visible in every benchmark.

The initial contracts.ModelCertificate schema is a narrow generic interval envelope. It does not encode arbitrary LabelEvents, deterministic-origin suites, exact fractions, simultaneous confidence-set construction or a uniform uncertainty-family quantifier. It must not be used as a lossy substitute for the native records above. Existing completed campaigns retain their native schemas and frozen sources rather than being rewritten to fit that draft envelope.

Cross-workstream integration uses EvidenceRegistry entries with stage, domain, content-addressed ArtifactRef, scope and lineage. Its verifier checks file identity only and explicitly returns scientific_completion=not_assessed. Benchmark claim decisions additionally require the corresponding native semantic replay and the stated empirical evaluation. The typed human primary-result contract remains separate from model certificates.

Numeric status is format-specific: rational exactness; directed integer bounds for exact-normalized float weights; tolerance-only external solver comparisons; or floating statistical quantiles. Do not label floating Clopper-Pearson endpoints exact merely because another component uses integer arithmetic. No certificate format here proves that human behavior belongs to the model family.
