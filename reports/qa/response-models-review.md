# Response model scoped review

Scope: response_chain.py, response_coarsening.py, response_replicas.py,
response_refinement.py, neutral_response.py and heldout_response.py, together
with their associated tests and tests/test_response_numeric_boundaries.py.
This is a scoped source review, not whole-repository ship-ready acceptance.

Two numeric defects were corrected. Teacher-response prior aggregation now
normalizes accepted unsigned environment keys to signed 64-bit integers before
forming bounded cell indices. Exact empirical response gaps retain integer
success counts rather than passing them through floating point, including the
range comparison. A one-count distinction above 2**53 is now preserved.

The new 22-test boundary suite covers these defects, original typed-domain
compatibility, partitions, pair shapes, split probabilities, coarsening support,
input immutability, replica groups and query-only masks. Original source for
both corrected modules is retained in source-only release fixtures. Restoration
of all eight corrected modules exactly reconstructs all four frozen final-human
source maps. Original forecasts, weights and empirical tables were not changed.

Held-out response masks exclude nonquery and unknown groups. Replica predictions
remain distinct; averaging losses does not imply averaging predictions. Smoothing
and unsupported-cell fallback are point-model choices, not confidence guarantees.
Neutral-model certificates describe fitted one-step models. An exact empirical
count gap does not establish human statistical significance or causal transfer.
The recorded negative human generalization findings remain unchanged.

Validation: reports/preflight/response-numeric-corrected-wheel-validation.json.
Installed-wheel tests: 1279 passed, 7 optional Torch/sklearn skips. All Python
wheel modules match current source; exactly eight differ from the historical
cloud-tested wheel. This is local validation, not a fresh cloud zero-skips run.
Corrections: reports/preflight/response-exactness-corrections.json.
Frozen identity: reports/preflight/frozen-human-source-restoration.json.

Remaining: whole-source QA, refreshed source/release packages and repository
publication. No commit or push gate has been declared satisfied.
