# Multistep human test inputs

Status: all 32 buckets constructed and independently reconstructed. No model prediction scores have been computed.

Source: `49fe45c8024a0c88032aa0444e48419970c1d1ac3077afc4b0d9975a5a2eda62`. Aggregate report SHA256: `200b8c2f603467ac63a807e8ae4e294286c671f35d377f49eb5fe7d6e95b198f`. All individual rows, histories and targets remain on Modal.

| Protocol | Horizon | Selected origins | Observed origins | Censored origins | Observed users |
| --- | ---: | ---: | ---: | ---: | ---: |
| known_users | 1 | 1,539,530 | 1,539,530 | 0 | 25,796 |
| known_users | 5 | 1,539,530 | 1,493,987 | 45,543 | 25,624 |
| known_users | 10 | 1,539,530 | 1,437,856 | 101,674 | 25,422 |
| known_users | 25 | 1,539,530 | 1,277,212 | 262,318 | 24,655 |
| known_users | 50 | 1,539,530 | 1,039,299 | 500,231 | 22,947 |
| unseen_users | 1 | 153,863 | 153,863 | 0 | 2,582 |
| unseen_users | 5 | 153,863 | 149,316 | 4,547 | 2,563 |
| unseen_users | 10 | 153,863 | 143,590 | 10,273 | 2,540 |
| unseen_users | 25 | 153,863 | 127,567 | 26,296 | 2,469 |
| unseen_users | 50 | 153,863 | 103,570 | 50,293 | 2,285 |

Origins are capped at 64 per user before future support is inspected. H decisions include the origin and H-1 immediately following original timestamp bundles. Events use the outcome-independent representative exposure from each bundle. Excluded bundles cannot be skipped.

This is a conditional observed-horizon cohort. Censoring is retained in support denominators, not recoded as disengagement or a negative response. Known-user and unseen-user results overlap and are not independent studies. Eventual recorded response targets and availability-proxy histories have different timing semantics.

This multistep cohort supplements the primary immutable one-step cases. It cannot replace them for the preregistered NLL test. The independent audit reconstructs all 1,539,530 selected origins, all histories and horizon targets, the unseen-user restriction, intrinsic context, and censoring masks using separate physical-row mapping, bounded-heap histories and direct future slices. Final frozen model/input hashes, prediction generation, user-clustered analysis and interpretation remain unfinished.
