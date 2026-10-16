# Validation-only precision planning

Status: approximate planning proxy complete for all ten GRU quantizer fits. Final predictive scores unopened.

For each fixed seed/protocol, compare the separately fitted K128 and K256 training-only quantizer forecasts on validation cases. Calculate paired NLL differences within user, then the unbiased variance across user means. These comparisons are a variance proxy, not the nested primary's accuracy result. No validation mean loss is exported or used to change the primary operating point.

The planning calculation uses one-sided alpha 0.025, target power 0.8 and the normal approximation `ceil(((z_0.975 + z_0.8) * user_SD / gap)^2)`, with a minimum of two users. For noninferiority, gap means margin minus assumed true added loss. It is not the actual unknown gap. Users must behave as independent representative clusters and future variation must resemble validation variation for this proxy to transport. Common calendar effects, selection and nonstationarity can invalidate that assumption. Confidence in transport of the planning variance is unknown.

Across the ten fits, validation user-level SD ranges from 0.016164 to 0.019703 nats. At a hypothetical 0.01-nat gap to the decision boundary, the normal approximation requires 21 to 31 users. At a 0.005-nat gap it requires 83 to 122 users. These small numbers describe a conditional normal calculation for this proxy; they are not a finite-sample power guarantee for the final nested models or intervention refinement. All declared users/cases remain in the experiment; no sample is reduced on this basis.

Known-user validation has 25,652 users and unseen-user validation has 2,557 users. Those are validation counts, not final eligible test counts. Required-user calculations for the complete gap sensitivity grid are retained in `reports/preflight/human-validation-precision.json`. E4H requires its own precision assessment because its randomized-response population and models differ.

The fixed D6 primary remains strict epsilon 0.01 from 64 initial classes, with at most 128 reduced states and a paired 95% upper NLL endpoint below 0.01. Its observed compression failures are not changed by this planning analysis.

Report SHA256: `34dd28e8d3b5acb79b35e82ac85667e7392787818b485f4d220280ce69d4d64a`.
