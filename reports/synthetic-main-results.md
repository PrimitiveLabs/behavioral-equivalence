# Full synthetic correctness results

Status: the frozen 1,200-world matrix, its integer certificate audit, and complete descriptive aggregation are finished. This is a specified-model result. Human predictive fidelity, human interventions, independent proof review, and the full research release remain open.

## Complete design and measurements

The benchmark crosses five mechanisms, 32/128/512 states, perturbations 0/0.005/0.01/0.03, and twenty fixed seeds. Every world receives all 22 variants: exact rational partitioning, three approximate tolerances, embedding/predictive/random clustering at three capacities, and nine feature-initialized verified refinements. Each variant uses horizons 5, 10, 25 and 50.

The completed aggregation binds every world report and every audit report to the exact hashes in the 1,200-world integer-audit manifest. It retains 26,400 method records, 316,800 unconditional property comparisons, and 105,600 conditional comparisons. Each comparison reports the maximum discrepancy over the specified related origins. These are paired benchmark conditions, not 422,400 independent statistical observations.

The floating benchmark reports zero bound violations at its frozen numerical tolerance of 1e-10. The separate integer audit verifies 26,400 certificates using exactly normalized binary-rational kernels and directed probability enclosures. It binds the reconstructed quotient to the actual serialized PRISM input. Certificate arithmetic is separate from floating screening, but it does not constitute an independent proof-assistant check of the general theorem.

PRISM 4.11 completes 55,200 independent point checks: completion and disengagement at horizon 50 from state zero, on each original model and every quotient. The largest absolute difference from the floating dynamic program is 2.3314683517128287e-15. PRISM does not independently check every origin, conditional property, or horizon in the larger table.

## Compression and numerical limitations

For unperturbed worlds, mean exact-partition compression is 4.8x at 32 states, 9.6027x at 128 states, and 28.8723x at 512 states. The respective ranges are 4–8x, 4–32x, and 4–128x. Different mechanism families permit different extra equivalences beyond deliberately duplicated states. These are descriptive means over the fixed benchmark, not a prediction of human compression.

Among nonzero-perturbation descriptive cells, approximate bisimulation has fewer mean states than exact partitioning in 20 of 27 cells. Feature-initialized verified refinement does so in 40 of 81 cells. A cell pools the five families and twenty paired seeds at one size, perturbation and method setting. These counts establish that approximation sometimes adds compression; they do not establish uniform superiority or identify an optimal algorithm. The complete 264-cell table includes all settings, runtime, graph size and compression ranges.

Ten requested epsilon budgets are exceeded under exact-normalized interpretation, by at most 1.7355908377147956e-17. The floating algorithm accepts within its documented numerical tolerance. Its certified error bound must therefore use the actual serialized radius rather than silently substituting the requested budget.

Nine hundred serialized quotients of exact rational partitions have nonzero radius, at most 4.470165557157735e-17. Exact equivalence of the input partition and exact representation of its floating serialized quotient are distinct numerical obligations. The certificates account for the measured serialized kernel. The final human strict-refinement implementation compares exact radii against its budget before accepting.

Runtime records reflect the documented concurrency increases, worker reuse and preemptions. Peak RSS is a worker-process cumulative high-water mark. Neither measurement supports an isolated per-method peak-memory or controlled hardware-speedup claim.

## Operational reconciliation

The original coordinator terminates with one recorded 180-second input timeout for the trust, 512-state, perturbation-0.03, seed-10013 world. A completed canonical report for the same frozen input is available when recovery runs, and the full integer auditor verifies it. Recovery therefore returns a cached completed report. It does not supply evidence of a new numerical retry.

The original coordinator manifest remains intact. The reconciled manifest replaces only its terminal failed result entry with the completed source-bound world. The retained `original-attempt` folder is a snapshot at reconciliation time, including the completed report, rather than a forensic reconstruction of the directory at the timeout instant. Two initial recovery wrappers fail module hydration and are stopped; they produce no accepted scientific results. The standalone wrapper mounts the original worker source explicitly and preserves its CPU, memory and timeout limits.

## Artifacts and remaining scope

- [Complete aggregate report](preflight/synthetic-main-summary.json), including all 264 descriptive cells and 1,200 artifact bindings.
- [Integer audit manifest](preflight/integer-campaign-audit-manifest.json), with 26,400 verified certificates.
- [Original coordinator manifest](preflight/synthetic-main-original-manifest.json), [reconciled manifest](preflight/synthetic-main-reconciled-manifest.json), and [recovery interpretation](preflight/synthetic-recovery-interpretation.json).
- [Figure 1](figures/figure-01/figure01-5fcd76d2f800/figure-01-abstraction.pdf), a measured 32-to-8-state reduction with complete transition tables.
- [Figure 3](figures/figure-03/figure03-4018f31e66d7/figure-03-preservation.pdf), all 105,600 unconditional error-versus-bound points across four horizons.

Cloud aggregate root: `synthetic-main-v3-d9f1fdcc7c35/summary-a8a177633a7d8758`. The compressed method and property tables remain checksum-bound there. Source digest: `efb1f2efdad610e42c84d6e1ff786df54e4674510ffdba1a8b628cc739ff1089`. Protocol digest: `d9f1fdcc7c355f047b4b5a1128b0ae88d0ca56d873d0996e52d0364516ebbbec`.

General soundness still rests on the documented theorem assumptions and mathematical argument. A finite benchmark cannot establish universal algorithmic soundness or human model validity. Independent proof review, integrated human comparisons, uncertainty ablations, intervention-response evaluation, and release reproduction remain separate obligations.
