# Supplementary target, cohort and validation controller review

Read attach_supplementary_outcomes.py, attach_supplementary_calibration.py,
prepare_supplementary_fit.py and validate_supplementary.py in full. No source edits.
Generic QA defaults apply; no root CLAUDE.md exists. Whole-source readiness open.

Target attachment binds prepared/case manifests and each physical source/case
file. Production decoded ordinals are crosschecked by independent Arrow take,
field-specific feedback reconstruction and explicit watch-time boundary slices.
Both stages require all 32 source buckets. Adjacent calibration attaches query
rows; the later pair_current restriction, not this stage, imposes origin caps.
Individual arrays remain private and no predictive scores are computed here.

Fitting preparation binds canonical target-report identity plus case/context
manifest and file hashes. It verifies case order across all three sources, chooses
train/validation roles only (with additional unseen-user role filtering), and
replays outcome-blind global ordinal sampling independently of bucket slicing.
The same cohort is used across fitted architectures/seeds. This does not select
or score supplementary test cases. Validator runs pinned outcome/head tests with
warnings-as-errors and refuses successful completion if pytest reports skips.
Its subprocess timeout propagates without persisting a successful result.

Entry points reserve finite existing-ledger amounts before exclusive submission
markers, retain call IDs and attach old calls. An attempt without a call ID needs
inspection rather than a replacement launch. These local guard tests do not
establish distributed publication transactions, Modal transport or upload safety.

Read-only source-map audit verifies all four historical report identities and
32-bucket coverage for attachment reports. Current code is not identical to every
historical capsule: first target attachment predates helper additions; fitting
cohort has an earlier config; original validation predates controller/helper/test
updates. Those exact maps remain embedded and hash-bound. Later current tests
must not be relabeled as the original cloud validation. Evidence:
reports/preflight/supplementary-pipeline-identity-replay.json.

Added eight tests for wrong manifest rejection before arrays, validator compiler
status/skip rejection and timeout behavior. Focused suite passes with warnings-as-
errors. Full local suite: 1561 passed, three Torch skips, three retained horizon
warnings. Ruff passes. Evidence:
reports/preflight/supplementary-controller-local-validation.json.
No provider execution, human rows downloaded or original outputs changed.
Remaining controller QA, strict final validation and release acceptance stay open.
