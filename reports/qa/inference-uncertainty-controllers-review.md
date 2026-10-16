# Inference validation, human count uncertainty and nominal replay review

Read verify_case_inference.py, validate_comparative_inference.py,
human_cluster_uncertainty.py and replay_nominal.py in full.
No executable edits, private-data download or provider calls occurred.

Raw-case inference verification is validation-only. It authenticates the frozen
encoder source/weights, training-only preprocessing and independently audited
case/context sidecars, selects matching physical case/user identities, and
reproduces saved validation forecasts within absolute tolerance. It does not
access target fields. Comparative validation repeats the bridge for all three
neural comparators, Markov baseline, independently fitted quantizers, calibrated
parent and nested abstractions. Frozen quantizer assignments and forecasts must
match exactly; neural forecasts allow only the configured absolute error.
Identity-sized primary abstractions must reproduce exact-normalized parent
forecasts. Comparator supports are paired within original users, with explicit
unsupported subgroup reporting. This validates the inference bridge and does not
substitute validation comparisons for frozen final test evaluation.

Human count uncertainty reconstructs outcome-conditioned calibration transition
counts using stored state assignments, checks physical identity and calibration
roles, verifies per-user caps and bucket ownership, and matches original parent
count tensors exactly. Bounded-user ratio intervals feed robust finite-horizon
arithmetic. Their independent-user, fixed-representation and Markov assumptions
remain necessary; human multistep coverage is explicitly unestablished. Every
retained K256 parent has vacuous emission intervals. This negative informativeness
result must stay visible rather than being described as useful human guarantees.

Nominal replay authenticates original and point-ablation manifests and frozen
protocol/source semantics, requires matching complete chunk families, and binds
each cached replay to both original input report hashes. The replay reconstructs
retained point-model evidence and checks complete dataset/certificate totals.
It is distinct from event interval coverage and full-kernel membership. Cached
reports remain historical evidence, not fresh independent numerical execution.

Retained reconciliation checks both validation comparator protocols, ten human
uncertainty parent records, all K256 vacuity reports and 400 nominal replay
chunks totaling 4000 datasets and 8000 point certificates. Evidence:
reports/preflight/inference-uncertainty-controller-reconciliation.json. No
private NPZ arrays were reread or fresh inference/certificates computed locally.
Current corrected helper source must remain distinct from historical cloud source.

Legacy launchers use saved handles and finite ledger reservations without the
later exclusive lifetime attempt controls; none was invoked. Provider transport,
concurrent mutation and fresh image-build behavior are untested by this scoped
review. Report-only changes required no test rerun. Strict environment validation
and whole-release acceptance remain open.
