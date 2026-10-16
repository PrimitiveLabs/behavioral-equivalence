# Trajectory likelihood and frozen-input QA, scoped review

Read tools/trajectory_likelihood.py, trajectory_backward_audit.py, trajectory_design.py, trajectory_targets.py and freeze_trajectory.py in full with their corresponding tests. No repository CLAUDE.md exists.

The forward model emits the origin outcome first, then propagates latent mass through outcome-conditioned transitions. It accepts only the origin state, not observed future state assignments/context. Scaling avoids long-sequence underflow; exact zero probability remains infinite loss. The backward replay uses a separately ordered recurrence over already validated stochastic inputs. Existing exhaustive hidden-path and all-binary-sequence comparisons, long/empty/impossible sequences, prefix checks, decision-stage extraction and aggregate reconstruction tests cover core numerical semantics. Per-user losses preserve infinity rather than clipping it away.

Sampling uses immutable case IDs only and never duplicates a small cohort. Reconstruction independently checks original contiguous user/role slices against all stored support/count/last-outcome arrays. The supplementary freeze binds every sequence bucket, ten original forecast reports and all 32 forecast bucket identities. It is exploratory and does not change the primary lock.

Before remediation, remaining direct assertions were needed for sampling/model/sequence domains, empty inputs, model/input immutability, extraction topology, original-user loss boundaries, reconstruction shape/horizon gates, freeze status/source/bucket/matrix mutations and immutable CLI reruns. A read-only rebuild should also reproduce the actual frozen trajectory configuration. None of these tests establishes an unconditional human trajectory law.

Verdict before remediation: needs-more-tests-before-merge for the listed gates, not whole-release approval.


## Remediation and actual freeze replay

`tests/test_trajectory_tool_boundaries.py` adds 56 passing cases. Forward-domain, sampling, stage extraction, original-user losses, reconstruction and freeze-source/status/matrix gates are asserted directly. Empty batches stay empty; impossible prefixes remain infinite; model arrays remain unchanged. The actual freeze CLI is exercised with a complete synthetic input matrix, accepting identical reruns and refusing replacement without overwriting existing contents. The independent backward recurrence remains an oracle for already validated model/outcome arrays, not a public raw-input ingestion API.

A read-only build_config replay against actual aggregate reports exactly reproduces the frozen trajectory configuration bytes (d835fc6ded1b46f5982efabed5677910ca9ecea64911bf2a429f2be7b12ab6cf), all ten fits and 340 forecast-output identities. reports/preflight/trajectory-freeze-reconstruction.json binds that check. It does not re-create the forecasts or replace the independent numerical trajectory audit.

All new cases and the full available suite pass against the unchanged corrected installed wheel, with source-path injection disabled. reports/preflight/trajectory-tools-corrected-wheel-validation.json retains exact commands and test bytes. No implementation, freeze or original experimental output changed. Scoped verdict: no identified gap remains in these five reviewed tools. Whole-source QA remains open.
