# Human random-response calibration support

Status: first construction complete. Human predictive transfer is not evaluated.

The known-user seed 20261015 history-only K256 model was applied to the complete audited random calibration cohort. Entire users were assigned using the fixed SHA256 split with seed 20261015. Discovery contains 100,168 exposures from 10,495 users; estimation contains 97,705 exposures from 10,516 disjoint users. These totals reconcile to the independently audited 197,873 calibration exposures from 21,011 users.

Neither subset has a parent-state/environment cell meeting both 50 users and 2,000 exposures. No supported pairwise witness exists within the initial 64 merged classes. The unrefined point model can be constructed, but this support result does not justify a human refinement claim. Thresholds remain unchanged. It is a negative support result at the chosen resolution, not proof that every coarser abstraction or human dataset lacks useful distinctions.

The next design check is a training-defined resolution sensitivity analysis with the same cell thresholds, before final predictive scoring. Any resulting design and operating-point selection must be dated and frozen. The original K256 result remains in the report. Final human response loss, uncertainty calibration, refinement controls and intervention transfer remain unfinished.

Report SHA256: `26de3da1da96877ec92974429d57cd99224daccb325877b203825973fa168887`.

Calibration source SHA256: `d6495cfe02aadfccfc3c947772b242f1d672583e2f5571c308f749bf6ff0b9b3`.

Private fitted tables and witnesses remain on Modal. This study concerns observed random-exposure responses in the released cohort, not population causal effects or arbitrary interface changes.

## Outcome-free training-rank resolution sensitivity

After the K256 sparsity observation, a dated support-only extension evaluated all five fixed resolutions without reading response values. These are nested coarsenings of the same frozen 256-state history model, ordered using standard-training sampled teacher means. They are not independently fitted K-state quantizers. Four adjacent rank states form each initial class at every resolution.

For known-user seed 20261015, requiring each state/environment to meet the unchanged threshold in both disjoint subsets gives:

| Parent states | Jointly supported cells | Supported within-class state pairs |
|---|---:|---:|
| 16 | 17 | 24 |
| 32 | 11 | 6 |
| 64 | 0 | 0 |
| 128 | 0 | 0 |
| 256 | 0 | 0 |

Support is not outcome disagreement. These counts establish where a candidate witness could be investigated; they do not establish a predictive improvement. Additional fixed seeds and the unseen-user protocol remain under evaluation. The original K256 result and support thresholds remain unchanged. Final intervention analysis is not frozen or scored.

## Complete five-seed support matrix

All ten history-only models and all 50 resolution checks are complete. Joint supported pair counts below require the same state/environment to satisfy the fixed thresholds in both disjoint subsets.

| Protocol | Seed | K16 pairs | K32 pairs | K64/K128/K256 pairs |
|---|---:|---:|---:|---:|
| known_users | 20261015 | 24 | 6 | 0/0/0 |
| known_users | 20261016 | 24 | 6 | 0/0/0 |
| known_users | 20261017 | 24 | 4 | 0/0/0 |
| known_users | 20261018 | 21 | 7 | 0/0/0 |
| known_users | 20261019 | 24 | 9 | 0/0/0 |
| unseen_users | 20261015 | 0 | 0 | 0/0/0 |
| unseen_users | 20261016 | 0 | 0 | 0/0/0 |
| unseen_users | 20261017 | 0 | 0 | 0/0/0 |
| unseen_users | 20261018 | 0 | 0 | 0/0/0 |
| unseen_users | 20261019 | 0 | 0 | 0/0/0 |

Known-user K32 is support-feasible across all five fixed models. Unseen-user state-pair inference is unsupported at every tested resolution under the unchanged thresholds. This is a limitation of this cohort/design; it does not establish an absence of behavioral differences. No final loss comparison, human witness significance or causal effect has been measured.
