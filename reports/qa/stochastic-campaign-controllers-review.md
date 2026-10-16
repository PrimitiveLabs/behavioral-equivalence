# Intervention, uncertainty, repeated-calibration and point-ablation controllers

Read intervention_campaign.py, uncertainty_campaign.py,
uncertainty_repeated_campaign.py and nominal_ablation_campaign.py in full.
No source edits. Generic QA defaults apply; no root CLAUDE.md exists.
Whole-source ship-ready acceptance remains unproven.

Campaigns bind source/protocol semantics before execution. Intervention worlds
retain complete reports, queried/held-out kernel evidence and artifact hashes;
negative controls must preserve behavior. Cache hits check source/spec/protocol
and artifacts. Full versus smoke manifests remain separate. Later analysis,
not a successful coordinator response alone, validates complete control matrices.

Initial uncertainty workers draw fixed row samples and pair Hoeffding/CP methods
on the same counts. Confidence-tail/numeric and hidden-state envelopes remain
separate. Early cached records rely on a fixed source tree; later audit rehashes
certificates. These are rectangular per-step bounds, not universal fixed-kernel
or human-data guarantees. Resource validator checks hard function limits and
conservative allowance. Initial complete evidence covers 200 worlds/600 paired
datasets, not the later 4000-dataset repeated-calibration experiment.

Repeated calibration fixes row allocation before samples, pairs two methods and
two terminal-support interpretations, and seeds each repetition by world/regime/
index. Known terminal closure is additional specified information, not learned
from observed zeros. Cache artifacts authenticate retained draws/certificates.
Independent units remain calibration datasets, not four paired evaluations.
Nominal ablation authenticates the original full manifest, protocol, original
replay and shared numerical implementation; it reuses retained counts without
resampling and explicitly claims no statistical coverage for the point estimator.
Smoke and full output identities are separate. Partial failures remain retained.
Legacy launchers lack some later attempt/lifetime-lock guards; not invoked here.

Sixteen added tests exercise semantic/source locks and cache artifact/source/spec
checks using actual controller bodies with local filesystem/volume bindings.
Successful cache reuse performs no new numerical work or commit. Retained counts
also reconcile: initial 200 worlds/1200 method-dataset evaluations and repeated
4000 independent datasets/16000 paired method-support evaluations/400 replayed
chunks. Every method-support denominator remains 4000. These aggregate checks
are not fresh arithmetic replay or a general coverage proof.

Full local suite: 1627 passed, three Torch skips, three retained horizon warnings.
Ruff passes. Evidence: reports/preflight/stochastic-campaign-local-validation.json
and stochastic-campaign-count-reconciliation.json. No cloud compute, provider
inference or original scientific-output mutation. Remaining controller QA,
strict final environment validation and release acceptance remain open.
