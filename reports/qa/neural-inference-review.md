# Neural definitions and case inference scoped review

Read neural.py, exposure_neural.py, content_neural.py and case_inference.py in
full. Reviewed tests/test_case_inference.py, test_content_neural.py,
test_exposure_neural.py and test_horizon_inference.py. No source changes.

GRU and Transformer branches embed only supplied history. Transformer attention
has a padding mask but no causal mask within that preceding history; that is
appropriate only when the caller has already excluded current/future outcomes.
The canonical history validator requires right alignment and either feedback
tokens 1..16 or the single empty-history marker 17. An all-padding history is
rejected there. The low-level neural modules rely on validated callers.

Exposure models concatenate tab and training-vocabulary item embeddings. Content
models append two intrinsic-context features. The context-only branch ignores
history by construction. Existing Torch tests assert finite gradients for every
registered parameter, context influence, deterministic evaluation and factorized
head/batch parity. They are not newly executed locally: the three Torch test
modules remain explicit collection skips. This source review does not upgrade
the current corrected wheel to cloud-validated zero-skips status.

Case inference preserves physical ordering, matches immutable context IDs,
filters known-user vs unseen-user splits, counts only available history, forms
exact serialized emission ratios and applies fine/coarse/class maps once.
Twenty-eight new tests cover malformed alignment, duplicate/negative/large IDs,
empty cohorts, both protocols and splits, invalid history rows, nonfinite or
nonstochastic emissions and invalid mapping/environment probabilities.
Thirty-two focused inference tests pass. Full installed-wheel suite: 1307 passed,
7 optional skips. Ruff passes. Evidence:
reports/preflight/inference-boundary-installed-wheel-validation.json.

Scope is source definitions and pure inference boundaries, not a full review of
all training/inference cloud controllers. No new provider compute, training or
human inference. Whole-repository QA and final release remain open.
