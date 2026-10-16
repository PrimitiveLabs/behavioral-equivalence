# Final source reproduction reservation

The updated checker reserves $0.14 under the existing $40 nested-human ledger. No ceiling or earlier reservation changes. The source identity changes with the checker, so this is a new fixed reservation, not a modification under an old run ID.

At the [official Modal pricing](https://modal.com/pricing) checked October 16, 2026 UTC, CPU costs $0.0000131 per physical-core second and memory costs $0.00000222 per GiB second. The checker fixes two CPU cores, eight GiB, 1800-second execution timeout, zero retries and one maximum container. Its full-timeout resource estimate is $0.079128. Adding 1200 seconds at that same full envelope yields $0.13188, below the $0.14 reservation. The extra period covers an allowance for loading and idle overhead. This is a conservative compute planning estimate, not an enforceable bound on every provider charge. Cached image builds, storage, egress, credits and invoice reconciliation remain distinct.

Before submission the ledger has $39.852368 reserved and $0.147632 available. The proposed reservation yields $39.992368 reserved and $0.007632 available. An atomic reserve rejects any conflicting concurrent allocation that removes this capacity. Tests verify the exact reservation, decorated resource limits, estimate arithmetic and rejection before spawning when capacity is insufficient.

No GPU, model API or human-data work is part of this reproduction. The run builds and tests the installed wheel and executes the small ten-world scientific subset.
