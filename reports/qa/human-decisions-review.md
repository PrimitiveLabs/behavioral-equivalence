# Human cohort and scientific-decision QA, scoped review

Read human_comparison.py, human_protocol.py and tools/summarize_human_primary.py in full with their corresponding tests. No repository CLAUDE.md exists.

The primary intersection-union decision requires both protocols and every fixed seed, at most 128 of 256 states, and an upper confidence endpoint strictly below 0.01 nats. A complete negative result is unsupported, not incomplete; missing fits remain not_evaluated. The response decision separately requires strict improvement for each fixed seed and cannot count identical forecasts as improvement. The summarizer binds the joint analysis-lock digest, native controller source and primary partition-report identity. File hashes supplied to the decision model are identities, not substitutes for the separate output audits.

Cohort exclusion hashes only sorted timestamp/tab multisets and row counts, retaining duplicates and excluding pilot identities even when user IDs are reindexed. The split then uses outcome-free hashed identity order. Archive schema validation requires all five observed log shards and all source columns. Existing tests cover cutoff thresholds, missing/duplicate/outside fits, seed/margin checks, invalid intervals, pilot exclusions and major schema failures.

Before remediation, direct boundary assertions remained for adjacent compression threshold, empty fixed matrices, additional identity/schema failures and the complete summarizer entrypoint with freeze/source/partition mutations. A read-only replay of the actual final reports should also confirm that code reproduces the retained negative decisions. No new model evaluation or human inference is needed.

Verdict before remediation: needs-more-tests-before-merge for the listed boundaries, not whole-release approval.


## Remediation and actual report replay

`tests/test_human_decision_boundaries.py` adds 19 cases. The actual summarizer entrypoint processes a complete temporary fixed-study matrix and retains failed compression/strict-response decisions. Independent mutations of freeze/source/partition identity reject before publishing an output; a missing fit remains incomplete/not_evaluated. Added cohort/schema checks retain order invariance and reject malformed identity/shard/column inputs. Exact adjacent state-count boundaries are asserted separately from NLL noninferiority.

A read-only replay of all 50 existing final reports into temporary output reproduces the saved decisions exactly: ten primary fits, ten compression failures, and unsupported outcomes for both five-seed response studies. `reports/preflight/final-human-decision-replay.json` binds the source, lock and original decision identities. Original reports and decisions were not overwritten. This is an aggregate decision replay, not a new neural forward pass or replacement for the independent numeric output audits.

The added tests and full available suite pass against the unchanged corrected installed wheel, with source-path injection disabled; reports/preflight/human-decisions-corrected-wheel-validation.json retains the exact commands. No implementation changed. Scoped verdict: no identified gap remains in the three reviewed files. Whole-source QA remains open.
