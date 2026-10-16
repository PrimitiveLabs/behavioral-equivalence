# Pilot/full preparation, calendar support and confirmatory case construction

Read prepare.py, prepare_27k.py, calendar_support_27k.py and
confirmatory_cases_27k.py in full. No source edits. Generic QA defaults apply;
no root CLAUDE.md exists. Whole-source ship-ready acceptance remains unproven.

Pilot preparation enforces its embedded 1K archive license, fixed source schema,
complete-record deduplication and timestamp bundles. Candidate event keys are
explicitly not unique event IDs. It preserves tied observations rather than
selecting an outcome-dependent order. Full preparation streams five expected logs
into user-disjoint modulo buckets, closes writers on failure, and freezes all
conversion source. Identity matching uses raw duplicate-retaining fingerprints;
1000 pilot exclusions precede deduplication. Fixed source means current corrected
library cannot silently reuse the original conversion capsule. Use frozen source
recovery for reproduction. Conversion consumes its verified immutable archive
from ingestion; the later audit independently checks converted records, not a
second complete original CSV parse. Failed conversion state is not success.

Prepared histories retain max/summed feedback-duration timing proxies and complete
user/time ordering. Actual feedback availability remains an assumption. Early
cached preparation records rely on the frozen source/data tree; independent
hash audits provide later integrity checks. Calendar support reads no click counts
or predictions and covers both user protocols, proxies and embargo choices. Role
support and timestamp/history counts are not a statistical power calculation.

Confirmatory construction hashes prepared inputs, binds source/config/tests,
requires smoke and independent full-data audit, namespaces physical rows by bucket
and ordinal, and samples with declared role caps. Cached output bytes are checked
before reuse. Three embargo case sets overlap; they are not independent datasets.
Targets/histories stay private and preparation completion is not final scoring.
Legacy launchers lack later attempt/lock guards; none invoked here.

Thirteen added tests exercise bucket bounds, exact source-freeze rejection and
case-cache source/output binding. Actual retained support report reconciles all
32 buckets, twelve conditions and 48 role/history cells including timestamp
extrema, and matches prepared identity with independent audit/case configuration.
Evidence: reports/preflight/preparation-support-controller-reconciliation.json.
Full local suite: 1611 passed, three Torch skips, three retained horizon warnings.
Ruff passes. Validation: reports/preflight/preparation-controller-local-validation.json.
No provider jobs or human-data download/reprocessing. Remaining controller QA,
strict final environment validation and release acceptance remain open.
