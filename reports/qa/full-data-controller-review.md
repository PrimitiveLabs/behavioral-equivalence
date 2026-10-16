# Full-data ingestion, pilot identity, independent audit and metadata controllers

Read ingest_27k.py, pilot_identity.py, audit_27k.py and item_metadata_audit.py
in full. No source edits. Generic QA defaults apply; no root CLAUDE.md exists.
Whole-repository ship-ready acceptance remains unproven.

27K ingestion resumes a partial archive only with a matching 206 Content-Range,
otherwise restarts the partial file. Full length/MD5 are required before rename;
SHA256 and verified-download manifest are retained. Archive members are streamed
for schema/license inspection, not blindly extracted. Embedded license text takes
precedence; absent text requires exact deposit filename/checksum/size evidence.
Download commit precedes schema inspection, so an incomplete inspection is not a
completed research dataset. No download or network test performed here.

Pilot fingerprints use all three raw converted logs, sort user/time/tab, retain
raw duplicates and exclude outcomes/items/release-specific user IDs from payloads.
The old controller assumes its recorded signed timestamp/tab schema; the corrected
library codec is separately preserved. Fingerprints are exclusion-only, not model
features. Actual retained 1000-user exclusions and uniqueness are established by
the full cohort audit, not by treating missing mappings as acceptable.

Full audit authenticates all raw/prepared file hashes, reconstructs raw identity,
pilot matching and user split assignments, then independently audits each bucket
with DuckDB. Cached bucket results compare input file hashes. Cohort/identity
files are fixed-tree assumptions on cache reuse; no fresh raw parse is implied by
reading aggregates. It does not perform a second original CSV parser pass or prove
real feedback availability. Metadata audit authenticates the full archive and
random-source shards, rejects duplicate/out-of-range item IDs, retains unknown
metadata, and checks observed pool support and timing discrepancies. These are
not treatment assignment propensities or causal-effect estimates.

Read-only reconciliation matches exact deposit metadata bytes to inspection,
archive hashes to item audit, all 32 bucket aggregate counts, 26285 eligible users
and 1000 pilot exclusions. Evidence: data-controller-evidence-reconciliation.json.
No raw human rows were downloaded or reparsed locally. Ten new tests execute
streaming checksum helpers and fabricated pilot fingerprint function bodies
(root redirected to temporary files), plus invalid bucket/missing-shard guards.
They do not test real HTTP resume, large archive parsing or Modal transport.

Full local suite: 1540 passed, three Torch skips, three retained horizon warnings.
Ruff passes. Evidence: reports/preflight/ingestion-controller-local-validation.json.
Legacy direct entrypoints lack later budget/launcher guards and were not invoked.
Remaining controller QA, strict final validation and release acceptance stay open.
