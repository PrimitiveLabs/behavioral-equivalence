# Transition uncertainty ablation results

Status: all 400 chunks, 4,000 retained calibration datasets and 8,000 paired point/support evaluations complete. This secondary ablation was designed after inspecting the original uncertainty results.

The original experiment and the ablation share the same 20 generating worlds, two fixed row-allocation regimes and 100 repetitions per world/regime. Methods and support controls share each dataset. They are not additional independent worlds or independent samples.

## Generating-probability containment

A dataset passes only if every tested concrete-state/event/horizon probability is enclosed. The suite covers completion, disengagement and continued activity at horizons 5, 10, 25 and 50. This is not a comparison between a confidence interval and a single binary response.

| Terminal support | Row allocation | Point control contained | Proven noncontainment | Numerically unresolved |
| --- | --- | ---: | ---: | ---: |
| known_closed_terminal_labels | imbalanced_active | 102/2,000 | 1898 | 0 |
| known_closed_terminal_labels | uniform_active | 1877/2,000 | 123 | 0 |
| unknown_terminal_rows | imbalanced_active | 0/2,000 | 2000 | 0 |
| unknown_terminal_rows | uniform_active | 0/2,000 | 2000 | 0 |

The original Hoeffding and coordinate Clopper-Pearson outcome envelopes contain all tested probabilities in every corresponding dataset. Full-kernel inclusion is a different metric. All point-control kernels fail exact full-kernel membership, including under known terminal closure. The control combines finite-sample empirical rows with imputed unobserved rows, so full-kernel membership is a stricter criterion than observable-property containment. Within-label transition details need not affect outcome probabilities. Neither kernel nonmembership nor point-control noncontainment constitutes a failure of the conditional numerical certificate: the generating kernel is outside the point family.

## Completion interval width at horizon 10

Values below average each dataset's mean width over its active states, with equal dataset weighting. The point mean is derived from the original Clopper-Pearson mean plus the paired point-minus-reference difference.

| Terminal support | Row allocation | Point mean width | Clopper-Pearson mean width | Paired difference | Simultaneous 95% interval for difference |
| --- | --- | ---: | ---: | ---: | --- |
| unknown_terminal_rows | uniform_active | 0.0956 | 0.9078 | -0.8123 | [-0.8981, -0.7264] |
| unknown_terminal_rows | imbalanced_active | 0.2634 | 0.9725 | -0.7092 | [-0.7951, -0.6233] |
| known_closed_terminal_labels | uniform_active | 0.1062 | 0.7833 | -0.6771 | [-0.7630, -0.5912] |
| known_closed_terminal_labels | imbalanced_active | 0.2730 | 0.9175 | -0.6445 | [-0.7304, -0.5586] |

The point control is narrower in every horizon-10 cell. The complete summary retains all four horizons and both statistical references. Forty simultaneous Monte Carlo intervals use Hoeffding and Bonferroni conditional on the fixed worlds. Paired width differences lie in [-1,1]. Initial-distribution-weighted width differences are descriptive. The intervals are numerically evaluated statistical summaries, not exact arithmetic model certificates.

## Interpretation and limitations

Removing statistical transition uncertainty can produce narrow ranges that miss known generating probabilities, particularly with unsupported active rows. Known-closure uniform sampling has observed containment 93.85%; its simultaneous interval includes 95%, so this analysis does not establish a precise below-95% coverage claim for that cell. The imbalanced known-closure estimate is 5.1%, demonstrating a much larger failure under this fixed sparse design.

For zero-sample rows the control imputes a uniform law over allowed successors. Unknown terminal support therefore permits departures from terminal labels and yields proven noncontainment in every dataset. The known-closure control isolates that specification choice. Changing support also changes the point imputation, so comparisons between its two support conditions are not pure uncertainty-width comparisons.

The same hidden-state abstraction envelope remains in every method. A rational point transition is enclosed with outward-rounded binary endpoints only to preserve numerical feasibility. Those tiny intervals have no confidence interpretation. The original statistical guarantees require fixed-design independent row samples; none follows automatically for dependent human histories.

All original uncertainty evidence was separately replayed. The nominal workers verify count-file identities, point-row construction and the abstract envelope while generating directed probability enclosures. The new summary validates the complete pairing and the failure-count extraction checks the frozen manifest; these do not constitute a separate independent arithmetic replay of all nominal certificates.

## Provenance

Point source: `a0450320cc941c72da068fb7873cd294c60d4b1d3bb6c8c0f2ee58e7b006960d`.
Analysis source: `2dc0028c982330fed7450605a13027b7d5002202236a39369a7ee05d66e42b25`.
Original manifest: `38ebd31d33c62c5ea172123273804ba134af0c7e1735314ba31bde90d1cd6905`.
Nominal manifest: `41683bc731fb206428ebd6e38aa7f34e19fa2509217ee511f88f90c883c68a7e`.
Summary SHA256: `acb6832263d4a4cbd5b091c49d11a1fbbe30c2d8ea1f9ff498ea0cf5d3da1ea4`.
Failure-count report SHA256: `e84a21a88f47135650f4e747c9e0bbf21bfe092178c7c1af703b7147a338827f`.

Raw aggregates are `preflight/nominal-ablation-summary.json` and `preflight/nominal-ablation-failure-counts.json`. Full per-dataset evidence remains on Modal. Total planning reservation, including smoke, full campaign, paired summary and failure classification, is $1.999609 under the provisional $2 campaign allowance. This is not an authoritative billed total or a global provider spending cap.


Full retained-evidence arithmetic replay is complete: all400chunks,4000original calibration datasets and8000point-support certificates match canonical recomputation, with no discrepancies. Report SHA256 is `fd6e245847cebd3d20a5a2959d7302c11b6ae44de0c216e4ee78e7439643db43`; source digest is `15ecee307ea68a1611f7e29b5ce16c8dd3646784c12cbc1ba35887905a8f446f`. This separately executed replay uses the existing numerical implementation. It is not a second independent algorithm, theorem proof, new calibration sample or human coverage result. The original negative nominal-envelope containment findings remain unchanged.
