# Calendar sequence and direct-horizon training controller review

Read train_calendar.py, train_calendar_replicates.py, train_pilot.py and
train_horizon_heads.py in full. This is a source and retained report review.
No training, inference, provider submission or private-data download occurred.

The calendar and replicate trainers stage only train and validation calendar
roles, additionally requiring matching user roles for unseen-user evaluation.
Intrinsic sidecars are aligned by physical case IDs. Item vocabulary and duration
scaling fit training data only, with explicit unknown-item handling. Training-only
Markov counts provide a common baseline. Configured upstream staged artifacts are
hash-bound and rehashed, not rebuilt from test data. Replicate training shares
the frozen inputs rather than independently redefining cohorts.

All neural epochs fit on training cases. Best checkpoints and matching saved
validation forecasts are chosen by equal-user validation NLL, using strict
improvement with the first minimum retained. Seeds control initialization and
shuffling. Saved reports include parameter counts, hardware/software and output
hashes, while explicitly not asserting bitwise reproducibility across CPU/library
environments. Neither calendar training controller computes final test metrics.
Finite training losses are checked before parameter updates.

Direct horizon heads require the audited train/validation origin cohort and an
exact source-bound frozen parent encoder. Parent artifacts, training-only item
vocabulary and scaling are authenticated. One outcome-blind global training-origin
sample is shared across architectures and seeds within each protocol. Origin-only
representations predict explicitly observed horizon event targets. Missing horizon
targets remain excluded for that event, not reinterpreted as negatives. Fixed
head hyperparameters fit training data; validation predictions are saved separately.
No final test predictions or metrics are produced by this controller.

The initial 1K GPU pilot uses fixed small samples and reports engineering pilot
limitations, including the earlier availability assumption and selected-field
deduplication. Its exploratory test metrics are not confirmatory 27K evidence.
The later independent 27K audit, pretest freeze and final scoring remain distinct.
The pilot has no durable source-bound resume or budget-ledger entrypoint and was
not relaunched. Calendar run entrypoints retain handles and resource reservations
but lack the later exclusive lifetime attempt controls. Horizon jobs add attempt
allowance accounting, without making provider concurrency failure paths tested.

Retained reconciliation covers thirty neural fits, thirty direct-horizon fits,
all thirty validation checkpoint choices, all thirty exact parent report links,
shared sample identities and six exploratory pilot cells. See
reports/preflight/training-controller-reconciliation.json. This checks retained
aggregate metadata, not fresh neural forward/backward passes or current remote
weight hashes. Current item-context preprocessing includes a separately reviewed
integer-identity correction; old cloud success is not fresh current-source testing.

No executable source changed and no tests were rerun. Three local Torch test
modules remain skipped without Torch, and numerical horizon warnings remain
visible. Strict final environment validation, remaining source QA and whole-release
acceptance remain open.
