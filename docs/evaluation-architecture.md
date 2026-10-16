# Human predictive evaluation architecture

Status: comparator bridge implemented and undergoing validation on the two fixed seed20261015 protocols. Final test scoring and the joint immutable analysis freeze remain incomplete.

```mermaid
flowchart LR
  A[Audited case and context buckets] --> B[Frozen training-only preprocessing]
  B --> C[Markov, context-only, GRU, Transformer]
  C --> D[GRU head activations]
  D --> E[Five training-only quantizers]
  E --> F[K256 calibrated point parent]
  F --> G[All nested variants and matched controls]
  C --> H[Private aligned bucket forecasts]
  E --> H
  F --> H
  G --> H
  H --> I[Complete-user bucket statistics]
  I --> J[Shared original-user bootstrap]
  J --> K[Aggregate reports and evidence hashes]
```

Every comparator receives the same immutable case identifiers within its protocol. The raw-case bridge reproduces saved neural validation probabilities within the declared 1e-6 numerical tolerance and checks exact saved quantizer assignments and forecasts. It independently applies the current-item vocabulary and duration scaling from frozen training inputs. The calibrated parent's binary emissions follow the exact-normalized row convention used by the certificate engine. Each nested file contains a next-positive forecast indexed by the original K256 parent state, so that lookup must not apply the latent map twice.

D6's quantized activation includes history and current item/context. The history-only E4 response states are a separate representation, with separate coarsenings, certificates and response studies. Neither representation is a recovered psychological state, and their certificates cannot be interchanged.

## Memory and statistical units

Neural inference batches within an audited source bucket. Scoring summarizes the complete cohort for each user in that bucket. A user occurring in a second bucket is an error, because independently normalizing partial user batches would give that user multiple votes. Persistent in-memory statistics scale with users times fitted candidates, rather than cases times candidates. Raw forecasts remain private in the research Modal volume.

Point metrics include case-weighted and equal-user NLL/Brier scores and 20 fixed calibration bins. Each bin retains its share of the original equal-user mass. Empty bins and empty cohorts have no invented outcome rate. Cohort inference requires at least 50 users and 2,000 cases. Unsupported cohorts retain support and descriptive point summaries without bootstrap claims.

For paired comparisons, one matrix contains each original user's mean loss for every fitted candidate. The same bootstrap draw of user multiplicities applies to every contrast. Multiplicity-count matrix products avoid allocating a bootstrap-by-users-by-models tensor. Regression tests compare this calculation against the existing explicit user-index resampling implementation, including unequal user case counts, identity contrasts and reversed contrasts. Intervals are pointwise percentile intervals conditional on fitted models. They do not include retraining or common calendar shocks and do not establish adjusted significance for selected secondary findings.

The strict primary is never replaced by the best secondary operating point. All ten strict primaries retain256states, so they fail the2xcompression target. This is a result for these fitted models, kernels, tolerances and construction rules. It does not establish an information-theoretic lower bound for human behavior or prove that other abstractions cannot compress.

## Validation and final scoring separation

`configs/comparative-inference-validation-v1.json` declares exactly two validation probes, one per user protocol at the first fixed seed. The controller rejects a test split. These full-cohort probes exercise all three neural architectures, the Markov predictor, all five independently fitted quantizers, the calibrated parent, the complete nested frontier and capacity-matched controls. Per-user statistics and bucket forecasts remain private; report artifacts contain aggregates, input identities, model counts, numerical reproduction errors and output hashes.

The probes do not replace the final ten-fit human benchmark. Post-calibration point models are applied retrospectively to validation timestamps, and neural checkpoints already used validation selection. Thus the resulting comparisons validate the engineering pipeline and numerical conventions, not prospective human predictive fidelity. Final scoring still requires the exact source/model/case identities, full baseline and ablation grid, multistep estimands and support, uncertainty diagnostics and statistical settings in the joint immutable analysis freeze.

Human probability-interval evaluation must use a declared cohort event-probability estimand and original user clusters. The scoring engine must not describe containment of individual binary outcomes as confidence-interval coverage. Unknown human-generating probabilities remain unobserved; synthetic coverage, empirical calibration and model-level verification are separate claims.
