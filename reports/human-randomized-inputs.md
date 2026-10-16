# Human randomized-response cohort and independent audit

Status: input construction and independent audit complete. Model transfer and human fidelity are not evaluated.

The frozen source cohort retains 1,143,022 canonical random exposures across 32 partitions. It preserves concurrent exposures and immutable physical identities. Calendar availability rules exclude 201,069 exposures from eligible prediction roles. Counts below are protocol-specific; the protocols overlap and must not be added.

| Protocol | Training exposures/users | Validation exposures/users | Calibration exposures/users | Test exposures/users |
|---|---:|---:|---:|---:|
| Known users | 245,612 / 17,828 | 111,325 / 17,290 | 197,873 / 21,011 | 387,143 / 23,345 |
| Unseen users | 170,727 / 12,512 | 11,737 / 1,751 | 20,382 / 2,100 | 40,507 / 2,337 |

The independent audit reconstructs selection, available prior histories, role assignment, intrinsic metadata contexts and reported support from canonical source partitions. It uses independent NumPy selection/history calculations. Shared raw inputs and underlying feature conventions remain assumptions; this is not an independent audit of dataset collection or human-process validity.

Cohort report SHA256: `82b9c911061fce07f429977af499e26d42ce7f3a9ddc92684d387f21260ea071`.

Audit source SHA256: `ff202e997093eb0cd609671f8b07659fc622be660a2938968c7aa8253fbc7bfb`.

Audit report SHA256: `7576df68fbf90fbc491d5733390485eebbcd3c42a380c8013ec5b6c19d3913d2`.

Raw human rows remain private on Modal. Local reports contain aggregate support and hashes. The difference between earlier raw-source row totals and canonical cohort totals must be reconciled explicitly before claiming raw archive row preservation.

The response-analysis helpers provide outcome-independent, deterministic user-level discovery/estimation separation and supported cell witnesses. Unknown-duration contexts are retained descriptively and excluded from active witness selection. These helpers are tested but have not yet been applied to fitted history states. Model calibration, matched refinement controls, final analysis freeze and held-out response scoring remain incomplete. No population causal effect, arbitrary interface effect or multistep counterfactual transfer follows from this cohort.
