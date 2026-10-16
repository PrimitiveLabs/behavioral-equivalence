# Human pooled-count uncertainty results

All ten representations and fifty interval models completed. All100saved model/certificate outputs have verified hashes. The cloud run took281.05seconds. These are conditional model-construction results; final held-out human diagnostics remain pending.

| Protocol | K | Mean emission-width range across five seeds | Fully vacuous rows / total |
| --- | ---: | ---: | ---: |
| known_users | 16 | 0.7226 to 0.7608 | 38 / 80 |
| known_users | 32 | 0.8806 to 0.9063 | 115 / 160 |
| known_users | 64 | 0.9595 to 0.9715 | 268 / 320 |
| known_users | 128 | 0.9972 to 0.9992 | 618 / 640 |
| known_users | 256 | 1.0000 to 1.0000 | 1280 / 1280 |
| unseen_users | 16 | 0.9216 to 0.9347 | 63 / 80 |
| unseen_users | 32 | 0.9844 to 0.9985 | 145 / 160 |
| unseen_users | 64 | 1.0000 to 1.0000 | 320 / 320 |
| unseen_users | 128 | 1.0000 to 1.0000 | 640 / 640 |
| unseen_users | 256 | 1.0000 to 1.0000 | 1280 / 1280 |

Known-user fits use25,522eligible calibration users; unseen-user fits use2,542. Every reconstructed user contributes at most64origins. Rows are state instances across five fits, not independent statistical replications.

Every K256emission row is [0,1]. All unseen-user emission rows at K64andK128are also [0,1]. The conservative independent-user Hoeffding ratio construction provides little useful precision at these granularities. This negative result is retained. Confidence levels, cohorts, seeds and cap were not changed to improve the finding.

The fitted stochastic model uses pooled calibration-count emissions, not training-teacher emissions. Alpha0.05covers the emission and transition coordinate families within one fitted model, not all50models jointly. Integer robust reachability certificates use decision horizons1,5,10,25,50and rectangular per-step uncertainty. Positive responses are nonterminal observed labels; no completion or abandonment interpretation is assigned.

Statistical interpretation requires independent user tables, a fixed representation and a stable target population. Known-user representation learning and common calendar shocks can violate those assumptions. The calculation does not prove human Markovity or multistep probability coverage. Hash checks verify artifact identity and do not constitute independent review of the statistical theorem.

The application reserved$0.50inside the existing$8uncertainty allowance, bringing reservations to$6.741612. These are internal reservations, not actual costs or a provider spending cap. No provider inference calls were used.
