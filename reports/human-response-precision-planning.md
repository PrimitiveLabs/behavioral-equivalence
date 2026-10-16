# Random-response precision proxy and test eligibility

Status: five-model variance proxy and outcome-free eligibility audit complete. Final human test predictions and metrics are not generated.

The all-group response models are fixed after calibration. Their paired validation-user loss variation is therefore a **retrodictive planning proxy**, not prospective validation accuracy. The source exports variances and normal planning calculations, not validation mean loss. Only validation responses contribute to paired losses. The comparison is targeted-b8 versus matched random-capacity-b8, with 107,799 known-duration validation cases from 17,128 users per model.

Across the five fits, user-level SD ranges from approximately 0.0210 to 0.0300 nats. At an assumed true improvement of 0.005 nats, a conditional normal calculation with one-sided alpha 0.025 and target power 0.8 requires 139 to 283 independent users. This is conditional on the chosen effect and transport of validation variation to future users. It is not a finite-sample power guarantee or evidence that the method improves predictions. Temporal shift, checkpoint selection, common calendar effects and the retrodictive design limit confidence in that transport.

The outcome-free primary test cohort has **375,340 cases from 23,284 users**, with recorded test role and available intrinsic duration. All five fitted models produce the same eligibility identity. Unknown-duration observations remain outside this primary cohort and must be reported separately; their outcomes did not drive selection.

Ordered eligible case-ID digest: `4831e001139d26eaffeb406dab8dbae560866981a65d40a8c10bde084ba860f6`.

This planning result applies to the all-group future-response comparison. The separate held-out-condition experiment remains unimplemented and requires its own models, eligibility and analysis freeze. This proxy does not complete that experiment.

Report SHA256: `767d01218f1aa783fc261cff6aee144e8cd131509e5a9c6a900dac4fe490f53f`.
