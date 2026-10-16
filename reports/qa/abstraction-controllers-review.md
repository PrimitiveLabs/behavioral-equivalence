# Learned parent and nested abstraction controller review

Read estimate_calendar_parent.py, nest_calendar.py, fit_history_states.py and
quantize_calendar.py in full. No executable edits or provider execution occurred.

Calendar quantization authenticates completed neural fits and exact frozen model
and scoring implementations. Scaling and centroids fit an outcome-independent
training sample; emissions distill teacher forecasts over training rows. Validation
case IDs, users, targets and teacher forecasts must reproduce within absolute
1e-6 tolerance. Smaller independently fitted quantizers are unverified baselines,
not quotients of K256. Empty training states use the explicit emission fallback.
The current fallback dtype correction is separately documented and does not make
historical cloud execution a current-source validation.

Parent estimation authenticates training, quantization and adjacent calibration
artifacts and exact encoder/preprocessing versions. It selects actual calibration
pair endpoints, preserving unseen-user roles and checking same-bucket user
ownership. Counts condition transitions on observed outcomes; additive priors
close unsupported rows. Initial mass weights users equally rather than weighting
all interactions equally. Teacher emissions and estimated transitions define an
emission-aware point model. Nonterminal positive/negative observations are not
completion, intentional abandonment or a human confidence set.

Nested construction authenticates the full K256 parent and quantizer and checks
its reconstructed chain digest. Strict refinement verifies label-path coupling
certificates; finite refinement verifies only the fixed event suite. Capacity-
matched embedding, predictive and random controls are explicit separate rows.
Each generated certificate is replayed before saving. Cached individual records
are bound to the run identity and output hashes, with no new arithmetic replay
on a cache hit. A completed cached record is historical evidence, not a new
independent proof. The primary method, initial count and budget are fixed in
configuration, and the compression decision requires at most 128 states.

History-state fitting authenticates the frozen GRU and training rows, excludes
physical randomized exposures from the standard training sample and uses only
pre-exposure history tokens for centroids. Current item/context information is
used by the training teacher to initialize predictive blocks, not by history-only
state assignment. These initial blocks are candidates for response refinement,
not verified behavioral equivalences. The randomized cohort is used for physical
ID exclusion, not for fitting to its held-out responses.

Retained reconciliation binds all ten nested reports to exact parent and
quantizer report bytes, all ten history maps to the corresponding frozen neural
fit, and checks support arithmetic and complete latent partitions. All ten
primary strict abstractions retain 256 states and fail the 2x compression target.
Evidence: reports/preflight/abstraction-controller-reconciliation.json. No private
weights/NPZ arrays were downloaded or rehashed live, no inference was rerun and
no certificates were freshly replayed by this local aggregate check.

These legacy entrypoints retain handles and finite ledger reservations but do
not all have exclusive lifetime submission locks; the nested smoke entrypoint
has no ledger reservation. No entrypoint was invoked. Provider transport, image
build, concurrent cache mutation and failure recovery remain outside this scoped
review. Report-only changes required no test rerun. Strict final environment
validation and whole-release acceptance remain open.
