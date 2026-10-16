# Reproduction and recovery

The source package contains the complete implementation, tests, pinned study configurations, Lean proofs and rights notices. The measured result package is separate. Neither includes raw human data, individual forecasts, trained weights, credentials or runtime ledgers.

## Fresh public scientific check

Install the pinned Python dependencies and run the complete tests, then invoke tools/scientific_reproduction.py with the official checksum-pinned PRISM 4.11 binary. This script generates ten fresh 32-state worlds spanning all five mechanisms and exact/perturbed regimes, runs all six abstraction-method families at four horizons, recomputes certificates and makes 20 external PRISM queries. It is a small scientific reproduction, not a second 1200-world campaign. The original full matrix and independent 26,400-certificate audit remain separate measured artifacts.

```sh
uv sync --group dev
uv run pytest -q -W error
uv run ruff check src tests cloud tools
uv run python tools/scientific_reproduction.py --output reports/fresh-scientific-run --prism /path/to/prism/bin/prism
```

Optional neural/data tests require the pinned Torch, scikit-learn, Polars, DuckDB and PyArrow versions listed in cloud/reproduce_source_package.py. The release cloud check builds a wheel, confirms imports resolve to that installed wheel, disables source-path injection, requires zero skipped tests and runs the scientific subset in a temporary fresh environment. Its report binds the archive, checker source, wheel, dependency versions, command logs and scientific artifacts.

## Full recorded campaigns

configs/program.json and docs/goal-architecture.md describe the entire dependency graph. Frozen per-campaign protocols retain all seeds, sizes, tolerances, estimands and resource reservations. The full synthetic matrix is configs/synthetic-main-v3.json; uncertainty, nominal-control, intervention and diversity protocols are separate. Do not multiply horizons into independent world counts or treat reused diversity worlds as fresh samples.

Human reproduction requires the official licensed dataset and the source-bound private Modal layout. Dataset roots, calibration/fit artifacts and exact output hashes are in the frozen configurations and source snapshots. Acquire data from its official distribution, honor dataset-specific terms and preserve pilot exclusion, calendar roles, vocabulary construction and outcome availability. Public source does not grant access to private volumes or silently recreate missing human evidence.

The 50-job primary runner is tools/run_final_human_jobs.py. Supplementary heads/calibration, final six-endpoint scores and joint trajectories have separate source-bound controllers. The primary pre-test lock is immutable; supplementary studies remain exploratory. Run the saved frozen source for an existing campaign, rather than editing a historical configuration to fit a new environment. An unprovided generic cold-start DAG submitter is not claimed: the complete study uses explicit campaign controllers and dependency/evidence manifests.

## Original synthetic source

The original 1200-world protocol intentionally rejects the newer working source. The permitted result package includes [the matching frozen tree](../reports/reproduction/frozen-synthetic-main-v3/snapshot-manifest.json). Its manifest records every file hash and the originating archive hash. The controller and nine protocol-listed modules recompute to `efb1f2efdad610e42c84d6e1ff786df54e4674510ffdba1a8b628cc739ff1089`, exactly the original protocol source digest. This is source recovery, not another campaign run.

To reconstruct an isolated original synthetic tree, copy `reports/reproduction/frozen-synthetic-main-v3/` into a new directory. Copy the release's `tools/guarded_modal.py` and `tools/run_final_human_jobs.py` into its `tools/` directory. Install that tree's locked Python dependencies and Modal before invoking its controller. Preserve any existing `.campaign` state when recovering an original call. A fresh tree without original provider metadata represents a new, billable experiment, not recovery of the old call.

## Recover existing calls safely

Retain each original provider call ID. Attach to a live call or retrieve its terminal report. A local timeout, network error or missing observation is not evidence that execution stopped. For the historical synthetic, uncertainty and intervention single-call controllers, use `tools/guarded_modal.py` instead of invoking their unguarded native entrypoints. For example, from a matching source tree:

```sh
python tools/guarded_modal.py cloud/synthetic_campaign.py --protocol-path configs/synthetic-main-v3.json
```

This command can submit billable compute if no original state exists. The guard derives the native state path from the frozen protocol, holds controller and state locks through the invocation, and saves an exclusive submission-attempt marker before calling Modal. Repeated invocations observe the saved original handle. It preserves that handle when native metadata disappears and refuses ambiguous attempts or conflicting handles. Protection applies to local invocations through this wrapper; it is not distributed exactly-once execution. Native historical source remains unchanged. The separate final-human runner has its own lifetime lock and attempt markers. Inspect provider history if a marker exists without a saved handle. Do not delete locks/markers to force a retry or promote an observation error to terminal failure.

Budget ledgers reserve conservative campaign costs atomically and reject increases past their existing ceiling. Reservations are not posted spending, credits, net invoices or provider-enforced global caps. No frontier provider calls or external language-model inference are needed. GPU training and CPU verification run on Modal; permitted archives use the isolated private R2 bucket with readback checksums.

The reports/evidence-registry.json inventory checks artifact identity, not scientific completion. Use native certificate replay, complete empirical result matrices and the original claim scopes as well. A model-level error certificate cannot establish that human behavior belongs to its model family.

## Corrected library versus frozen human source

The release library corrects one dtype bug in `quantization.emission_probabilities`: integer fallback values 0 and 1 previously selected an integer NumPy output buffer. The corrected buffer explicitly uses float64. Both recorded training callers supplied Python floats, for which the original and corrected buffer operations are identical. See [the correction evidence](../reports/preflight/quantization-dtype-correction.json). Historical measured outputs remain frozen; the local corrected-wheel check is not a new cloud reproduction.

A second correction prevents signed/uint64 promotion in support-cell counting. It retains exact user identities in separate structured fields. The recorded pipeline uses signed user IDs and narrow environment codes, for which counts remain identical. See [the support correction evidence](../reports/preflight/support-integer-identities-correction.json).

The identifier integrity review also corrects response-cell grouping, mixed-integer item lookup, and timestamp fingerprint serialization (including the independent audit). See [the correction evidence](../reports/preflight/identifier-integrity-corrections.json).

The response review additionally preserves exact integer successes in empirical gap witnesses and handles unsigned environment keys in teacher priors. See [the response correction evidence](../reports/preflight/response-exactness-corrections.json).

All four archived final-human source maps retain all eight original modules. Their source digests deliberately reject the corrected working library. To reproduce an existing frozen human campaign, first make an isolated copy of the release tree, then restore only that copy's original modules:

```python
import json
from pathlib import Path

frozen_copy = Path("/path/to/isolated/frozen-copy")
for module, fixture_name in [
    ("quantization.py", "quantization-dtype-v1.json"),
    ("resolution_support.py", "support-integer-identities-v1.json"),
    ("random_response.py", "response-integer-identities-v1.json"),
    ("item_context.py", "item-integer-identities-v1.json"),
    ("human_protocol.py", "fingerprint-integer-identities-v1.json"),
    ("data_audit.py", "audit-integer-identities-v1.json"),
    ("heldout_response.py", "heldout-integer-environments-v1.json"),
    ("neutral_response.py", "neutral-exact-counts-v1.json"),
]:
    fixture = json.loads(
        (frozen_copy / "data/synthetic/fixtures/release" / fixture_name).read_text()
    )
    (frozen_copy / "src/behavioral_equivalence" / module).write_text(fixture["source_module"])
```

Use that restored tree only with its original frozen configuration and provider state. The frozen maps, original cloud-tested archive and fixtures agree on the original module bytes. Do not rewrite a historical source digest or relabel corrected-source execution as the original campaign.

For the frozen supplementary calibration or final-scoring campaign, also restore
its original counting tool in that isolated copy. The corrected tool accepts
unsigned state keys without mixed-integer promotion; historical signed-state
counts remain unchanged. See [the correction record](../reports/preflight/supplementary-integer-states-correction.json).

```python
fixture = json.loads(
    (frozen_copy / "data/synthetic/fixtures/release/supplementary-integer-states-v1.json").read_text()
)
(frozen_copy / "tools/supplementary_counts.py").write_text(fixture["source_module"])
```

The descriptive state-sufficiency and supplementary audit tools also preserve
original source fixtures after integer-identity and finite-calibration corrections.
See [the audit correction record](../reports/preflight/result-audit-corrections.json)
for the frozen source maps affected. Restore the corresponding original tool only
when reproducing that historical source map in an isolated copy:

```python
for module, fixture_name in [
    ("state_sufficiency.py", "state-sufficiency-integer-identities-v1.json"),
    ("supplementary_result_audit.py", "supplementary-calibration-finite-v1.json"),
]:
    fixture = json.loads((frozen_copy / "data/synthetic/fixtures/release" / fixture_name).read_text())
    (frozen_copy / "tools" / module).write_text(fixture["source_module"])
```
