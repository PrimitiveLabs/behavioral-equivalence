# Multistep fitting inputs

Status: complete across32source buckets. Independent physical-row audit is complete across all32buckets; no direct heads or final human test forecasts have been generated.

| Protocol | Training origins | Training users | Validation origins | Validation users |
|---|---:|---:|---:|---:|
| known_users | 1,654,438 | 26,079 | 1,396,339 | 25,652 |
| unseen_users | 1,158,132 | 18,256 | 139,656 | 2,557 |

These are selected origin counts before horizon-specific censoring. Origins are sampled without future labels; each target requires consecutive bundles within the same actual fitting or validation role. H includes the origin, matching the event suite graph horizon2H-1. Roles and user-role restrictions remain distinct. Raw features and targets remain private on Modal.

The fitting-source description inherited one legacy test-role phrase from its precursor. Actual output role codes0/1, explicit splits, role-window tests and fitting scope are authoritative. The original source is preserved; this description correction changes no generated numeric inputs.

The secondary direct horizon comparator extension is specified before final test scoring in `configs/human-horizon-heads-v1.json`. Prepared labels alone establish no predictive fidelity. Source digest: `dc9b74c174ca508f263fdd6b73b7318db2b46ed119668281cd9e51773a92c4da`. Complete report SHA256: `451ebbc40c50f5b5852f66fe50d851bbd0a430e559c5c6cb94a90d5b90546415`.

The audit independently reconstructs both fitting roles from physical source rows, NumPy representative draws, bounded-heap history availability and direct future slices. Source digest: `a783bbabf73c547799b94bd8335ad0e3eee246fa7e72ab5124de2f2305ddb21b`; full audit report SHA256: `93565442cd23ad7cfb2fa9eb44578cf9335ce37081ff3b58c9516d56d890a80f`. Counts match preparation separately for train and validation, including unseen-user restrictions. Users shared across the two chronological roles are not counted as independent replications.
