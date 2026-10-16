# Initial synthetic and learned human model pilot controller review

Read pilot.py, learned_pilot.py, property_pilot.py and quantize_pilot.py in full.
No source edits or new functionality. Generic QA defaults apply; no root CLAUDE.md
exists. Whole-source readiness remains unproven.

Initial synthetic pilot compares all related origins for finite horizons, checks
PRISM point values on reduced initial state zero, and records a rational oracle.
Its robust check is a small engineering fixture and the report explicitly says
formal numeric soundness is not established. Later integer certificates and
external model-checking scope remain separate. An initial checkpoint is not a
completed report, and no pilot changes the full frozen benchmark design.

Quantization pilot loads the saved GRU safely, preserves physical case/target
alignment and requires absolute-only teacher forecast reproduction. Normalization,
clustering and teacher-mean emissions use training representations only. Clustering
is unverified and independently fit at each K, so reducing nominal K is not a
nested quotient of the K256 parent. Nominal/effective state counts and all
exploratory K comparisons remain visible. CPU reproduction disables TF32 but is
not a general cross-hardware bitwise identity guarantee.

Learned strict pilot reconstructs complete histories, samples one representative
per timestamp bundle independently of outcomes, uses genuinely adjacent same-user
calibration pairs and a declared transition pseudocount. All full/abstract model
certificates are verified, and empirical losses use identical test cases. Those
certificates cover estimated point kernels, not human transition confidence.
Exploratory epsilon choices and single fitted seed preclude a confirmatory claim.
Finite-property pilot explicitly follows the strict pilot's negative result,
authenticates the same parent, enumerates its bounded property suite and verifies
only those properties. It does not rescue bisimulation or establish preservation
for arbitrary events/environments or unknown human behavior.

Read-only retained report reconciliation verifies both complete twelve-case
frontiers, their state-compression arithmetic, twenty quantization metric cells,
eight paired contrasts and eighty synthetic property comparisons within recorded
floating tolerance. Exact inputs/hashes are recorded in
reports/preflight/learned-pilot-controller-reconciliation.json. No new certificate
arithmetic, model inference, raw-data download or provider execution occurred.
No tests were added or rerun for this read-only review; last full local suite
remains 1627 passed, three Torch skips and three retained horizon warnings.

Legacy direct entrypoints lack later bounded/cached launcher controls; none were
invoked. Current source and historical source versions are not interchangeable.
Remaining controller QA, strict final environment validation and release
acceptance stay open. Scoped review does not establish whole-source ship-ready.
