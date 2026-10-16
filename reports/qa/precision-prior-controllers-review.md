# Precision planning and teacher prior controller review

Read plan_response_precision.py, plan_heldout_response_precision.py,
plan_validation_precision.py and fit_response_teacher_priors.py in full.
No executable changes or provider execution occurred.

Validation planning authenticates stored validation forecasts from training-only
quantizers. K128 versus K256 loss variance is explicitly a proxy, not the nested
primary comparison. User-level variance supports only a conditional normal
approximation, assuming independent representative user clusters and transported
future variance. It is not a finite-sample power guarantee or an across-seed
independence claim.

Response planners authenticate history/frontier models and frozen encoders,
score only validation-role known-duration responses, and count test eligibility
from role, physical IDs and intrinsic duration without reading test targets or
generating test predictions. Their retrodictive validation proxy uses models
fitted on later calibration data, so it must not be described as clean prospective
validation performance or used as confirmatory test evidence.

Held-out planning requires all six two-of-four group holdouts and checks structural
forecast identity. Identical forecasts cannot establish improvement through more
samples. Each known-duration case appears in three holdouts; aggregate planning
combines these paired losses within original users rather than treating fold
copies as independent observations. The revised aggregate report retains identical
fold metadata and eligibility, with variance/SD rounding differences at most
1.3877787807814457e-17 from the earlier report. An initial exact-equality check
exposed these differences; the reconciliation records them explicitly with an
absolute 1e-15 comparison tolerance. No scientific report was modified.

Teacher priors use the frozen standard-only training sample, authenticated parent
weights and original training preprocessing. Source case IDs must match, intrinsic
duration bins must agree with stored training features, and teacher per-state
means must reproduce within absolute 1e-6. Fine/coarse prior probabilities use
training teacher forecasts, not randomized calibration, validation or test
responses. The later heldout-response integer-identity correction is separately
documented, not newly executed in historical cloud evidence.

Retained reconciliation covers ten validation proxy rows, five response rows,
thirty held-out fold rows, five original-user aggregate rows and five training
prior reports. Evidence: reports/preflight/precision-prior-controller-reconciliation.json.
This checks metadata and arithmetic, not fresh private forecasts or power.
Legacy entrypoints persist handles and finite ledger reservations without later
exclusive lifetime submission locks; none was invoked. No tests were rerun for
report-only changes. Strict final environment validation and release acceptance
remain open.
