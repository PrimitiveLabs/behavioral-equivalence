# Human abstraction results

All ten fixed GRU point-model abstraction runs are complete: five fitted seeds for each of the known-user and unseen-user protocols. Every fixed primary strict tolerance 0.01 output retains all 256 latent states, failing the required at least 2x reduction. This determines failure of the joint primary efficiency criterion even before predictive noninferiority is evaluated. Final human test predictions remain unscored; no noninferiority result is claimed.

| Protocol | Seed | Primary states | Smallest finite-property variant | Total candidates |
|---|---:|---:|---:|---:|
| known_users | 20261015 | 256 | 210 | 32 |
| known_users | 20261016 | 256 | 204 | 35 |
| known_users | 20261017 | 256 | 205 | 35 |
| known_users | 20261018 | 256 | 211 | 32 |
| known_users | 20261019 | 256 | 207 | 35 |
| unseen_users | 20261015 | 256 | 204 | 32 |
| unseen_users | 20261016 | 256 | 200 | 32 |
| unseen_users | 20261017 | 256 | 204 | 35 |
| unseen_users | 20261018 | 256 | 211 | 35 |
| unseen_users | 20261019 | 256 | 202 | 32 |

The secondary finite-property suite checks a fixed event/horizon family and does not establish bisimulation. Its smaller models cannot replace the failed strict primary. Different candidate counts arise from matched controls at each distinct output capacity. Predictions, calibration, runtime and the full compression versus loss frontier still require final evaluation.

Evidence: the ten `preflight/human-nested-{protocol}-{seed}.json` reports and `preflight/human-nested-completion-checkpoint.json`. All refer to estimated finite point kernels, not known human-generating processes. Source, parent identities and per-candidate output hashes are recorded. These data-derived model certificates do not establish unconditional human fidelity.
