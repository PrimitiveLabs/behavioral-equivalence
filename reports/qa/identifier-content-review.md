# Exact identifier and prediction-time content QA, scoped review

Read intrinsic_context.py, item_context.py, item_metadata.py, content_scaling.py, randomized_cohort.py, random_response.py, data_audit.py and human_protocol.py in full with the corresponding existing tests. Rechecked changed integer operations and their new regressions. No repository CLAUDE.md exists.

## Findings and corrections

The common-pattern review found four additional defects beyond the earlier two dtype fixes. Response-table grouping could fail on accepted uint64 keys. Mixed signed/unsigned item lookup could miss an adjacent valid training item above 2**53. Timestamp/tab column stacking could round distinct integer timestamps into the same fingerprint; the independent audit had the same serialization weakness. The fixes preserve exact integer keys: structured cell/user fields, a common nonnegative uint64 item-search domain, per-column signed64 fingerprint encoding and independent DuckDB BIGINT selection before streaming hashes. Values outside the fingerprint codec's declared signed64 source schema reject rather than wrap or round.

Original modules are retained as hash-bound source fixtures and match the historical cloud-tested archive and all four final-human source maps. Restoring all six corrected modules reproduces those source maps exactly. The recorded source ingestion uses signed int64 nonbinary columns and narrow environment codes. Synthetic signed-domain regressions retain the old results exactly; no original source rows, identity manifests, support tables, model outputs or decisions were overwritten. This is operation compatibility, not a neural-forward rerun.

## Regression and independent checks

The 29 new cases in tests/test_identifier_integrity.py and tests/test_independent_identifier_audit.py reproduce the original failures. They verify exact uint64 user counting, adjacent mixed-dtype item lookup, signed64 timestamp byte hashes against independent struct.pack, signed-path equivalence and real DuckDB/Arrow audit behavior for both signed and mixed columns. Original frozen functions are exercised to demonstrate the failures, not presented as independent scientific evidence. Additional checks cover supported vocabulary/context shapes, training normalization, empty cohorts and preservation of simultaneous random exposures while rejecting reverse-time input.

Existing tests retain frequency/ID tie-breaking, unseen-item index zero, strictly prior upload-day filtering, unknown-duration groups, training-only normalization, outcome-free exposure selection and supported witnesses at fixed scenarios. Metadata availability is a declared restriction, not proof of the logger's real-time information availability. Random-exposure flags do not supply arbitrary treatment propensities, and witness ranking is calibration selection, not newly acquired human outcomes.

The rebuilt six-correction wheel passes 1257 tests locally with seven optional skips, with source-path injection disabled and pinned Polars/DuckDB/Arrow. reports/preflight/identifier-integrity-corrected-wheel-validation.json binds all module differences, test bytes and commands. No new provider compute or human inference occurred. The historical zero-skips cloud report remains separate.

Scoped verdict: the identified integer defects and reviewed content/context gates are closed. This is not whole-source ship-ready approval. Remaining unreviewed source and final release/publication remain open.
