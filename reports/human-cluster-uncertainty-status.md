# Human pooled-count uncertainty application

Status: all10representations complete;100stored outputs hash verified. See `human-cluster-uncertainty-results.md` for the complete construction results.

The frozen application source is `65cd0d180d0f9089aff95ba88a3e6dbcd962561ec131a9de87200b060de26c1c`. Modal call: `fc-01M51A9WY0ZTNKGDPPBNGHSSH6`. It processes all ten GRU representations across both user protocols and five seeds, at K=16,32,64,128,256. It checks calibration/report/state hashes, calibration roles, case alignment, unique user buckets, the 64-origin cap and exact reconstruction of stored transition counts.

The model estimates pooled count-based emissions and outcome-conditional next-state laws. It does not reuse teacher emissions. Each model's alpha=0.05 covers its emission and transition families jointly, not all fifty models simultaneously. Robust integer enclosures cover positive-response reachability at decision horizons 1,5,10,25,50 under rectangular model uncertainty. No terminal completion or abandonment interpretation is assigned.

The uncertainty assumptions remain conditional: independent user tables, a fixed representation and an unchanged population. Known-user representation learning and common calendar shocks can violate those assumptions. This application cannot establish true human Markovity or multistep confidence coverage. Final held-out cohort diagnostics and comparison with nominal count models are still required.

The run reserves $0.50 within the existing $8 uncertainty campaign allowance, bringing that ledger's reservations to $6.741612. These reservations are not actual billing or a provider-enforced cap. No provider inference calls are used. Local checks: Ruff passes; 311 tests pass and 8 skip. Private human rows and model weights remain on Modal.

## Preliminary collected results

Seven of ten representation reports have been collected. All seven have entirely vacuous [0,1] emission intervals at K=256. For the four collected known-user fits (25,522 eligible users), mean emission widths at K=16 range from 0.7226 to 0.7608, with 7 to 9 fully vacuous state rows. For the three collected unseen-user fits (2,542 eligible calibration users), K=16 widths range from 0.9216 to 0.9347, with 12 or 13 vacuous rows. Every collected unseen-user fit is wholly vacuous from K=64 upward.

These are partial construction results, not final human predictive performance or a completed certificate audit. They show the severe precision cost of allowing arbitrary within-user dependence with a uniform 64-origin cap and many simultaneous coordinates. The confidence level and user unit remain fixed. Sharper procedures, if later developed, must be reported as separate methods.
