# Emission-aware human point models

Status: full adjacent calibration inputs and the first three known-user GRU point-model estimations are complete. Those GRUs use seeds 20261015 through 20261017 and the complete 1,393,257-transition calibration cohort. Remaining fitted seeds and the unseen-user protocol continue through the same construction. Nested abstraction and final test scoring remain open. A context-only point model supplies a separate engineering check.

## Frozen inputs and state construction

The encoder, item vocabulary, intrinsic-duration normalization and training sample are immutable training artifacts. Quantization uses the frozen head's 128-dimensional penultimate GELU activation. A seeded sample of 100,000 training cases fits representation normalization and each independent MiniBatchKMeans quantizer. All training cases then estimate state occupancy and the mean teacher probability within each state. No calibration or test representation fits centroids or normalization. Empty states use the global training teacher probability and retain an explicit support count.

The K256 quantizer defines the parent for nested reductions. Independently fitted K16, K32, K64 and K128 quantizers are separate unverified baselines. Report nominal and occupied state counts. A compressed baseline cannot be relabeled as a quotient when it uses different state assignments.

The context-only engineering run processes 6,623,154 training cases and reproduces the frozen teacher's validation probabilities exactly on the checked CPU execution. Its K256 model has 254 occupied training states. This support diagnostic is not a confirmatory compression or noninferiority result. Final predictive test scores remain unopened.

## Actual adjacent transitions

The calibration input generator uses all audited timestamp bundles, chooses one exposure uniformly within each bundle, and samples at most 64 eligible origins per user. Both an origin and its immediate next bundle must belong to the calibration calendar role under the 24-hour feedback-availability policy. Sampling does not consult response values. Retained bundles between selected prediction cases are never treated as missing process transitions or skipped over.

The full prepared input has 1,393,257 known-user transitions from 25,522 supported users. The unseen-user restriction has 139,378 transitions from 2,542 users whose declared user role is calibration. These protocols overlap and are not independent replications. Timestamp, user, source-row and bucket bindings remain private and hash-bound on Modal.

Inference reconstructs every selected endpoint from its complete available preceding history. Item encoding and duration scaling reuse training artifacts. The estimator validates that each pair has the same user and consecutive source-bundle indices before accumulating counts C[z,y,z_next]. A calibration query's recorded response enters the transition count only after its pre-response latent state is computed. It never enters that query's predictor features.

## Specified stochastic model

For K latent states, estimate T[z,y,:] by adding total prior mass one, distributed uniformly over K successors, to each outcome-conditioned count row. This is a nominal smoothing rule. A zero-sample row therefore becomes uniform; it is not supported by human observations and is not a calibrated confidence set. Report unsupported state/outcome rows and sensitivity to the prior.

Use the training teacher's distilled positive probability for E[z,positive], with its complement for the negative outcome. The point model therefore combines training-distilled emissions with calibration-estimated transitions. It does not replace emissions with final-test frequencies. Its initial mixture gives each supported calibration user equal weight and divides that user's mass equally among sampled origins.

The decision lift has 3K labeled states. A pending latent state emits a negative or positive observation, then the observed outcome moves to the next pending state through T. Positive and negative observations are nonterminal. They do not mean purchase, completion or intentional abandonment. An H-decision event from a pending state uses 2H-1 graph transitions. Specify whether a property concerns occupancy, first occurrence or a count threshold, and which initial stages are allowed.

## Verification and empirical interpretation

Exact or approximate nested reductions operate on this specified point kernel. Their certificates can establish how quotienting changes its finite-horizon label predictions. They cannot establish human Markov sufficiency, correctness of the smoothing prior, feedback-timestamp provenance, or sampling coverage for an unknown human-generating process.

The latent state includes current exposure context. Its estimated transitions average over the observed recommendation policy and released cohort. Multistep properties concern recorded exposure-indexed outcomes under that model. They are not arbitrary intervention forecasts. Any human comparison requiring H subsequent recorded bundles must report the conditional observed-horizon cohort and censoring support.

The existing sequence models use available bundle-response-rate history, current item identity, scenario and intrinsic duration. They do not encode a feature-complete history of item identities, timing and metadata. Stronger history comparators and sensitivity to omitted history remain part of the comparative-baseline review. Predictive preservation relative to this parent must always be reported alongside its loss relative to the continuous comparators, with this input limitation visible.

The implementation writes the underlying K-state transition/emission arrays, the labeled-chain digest, original counts, state support, initial weights, per-bucket state assignments and provenance hashes. Aggregate reports can leave Modal; individual human rows and histories remain private.
