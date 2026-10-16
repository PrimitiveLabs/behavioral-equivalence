# Confirmatory case, context, adjacent and multistep audit controller review

Read audit_confirmatory_cases.py, audit_case_context.py,
audit_adjacent_calibration.py, audit_multistep_test.py and audit_multistep_fit.py
in full. No source edits. Generic QA defaults apply; no root CLAUDE.md exists.
Whole-source ship-ready acceptance remains unproven.

Confirmatory audit binds prepared source/case files, reconstructs physical rows,
sample selection and histories for all declared embargoes, and aggregates only
after all bucket audits succeed. Context audit hashes metadata/source arrays,
uses authenticated physical row/item lookup and a strictly prior UTC upload day,
retaining missing-context flags and standard/random support separately. Metadata
consistency is an assumption about availability, not full online provenance.

Adjacent audit binds source/case reports, all files and origin-pair identity,
reconstructs histories independently and checks intrinsic context by a separate
formula. Origin pairs remain genuinely adjacent full timestamp bundles. Multistep
fit/test audits independently reconstruct selected origins, histories, future
slices and censoring for the declared calendar role. Training/validation and test
cohorts are distinct. Summed user counts rely on the declared user-disjoint bucket
layout. No audit establishes real human Markov sufficiency or unconditional
future engagement under censoring.

Cached audits rehash retained source/case inputs and compare relevant report or
case hashes; their source-namespaced directories do not imply a new raw replay.
Fixed cohort/metadata inputs remain part of cache reuse assumptions. Smoke tests
check source-bound local algorithms; they do not validate Modal transport. Some
legacy entrypoints lack later attempt/launcher guards and were not invoked.
Known terminal job observation, not an error while polling, governs replacement.

Read-only reconciliation verifies all five full 32-bucket audit aggregates:
160 bucket records, case/context report binding, all embargo totals, context
strata, adjacent transitions and multistep fit/test origin/user support.
Evidence: reports/preflight/case-audit-controller-reconciliation.json.
Twenty-three added tests validate exact report bytes, source digests, complete
bucket matrices and bucket bounds using actual function bodies with local input
bindings. They do not substitute for original full-source audits.

Full local suite: 1598 passed, three Torch skips, three retained horizon warnings.
Ruff passes. Evidence: reports/preflight/case-audit-controller-local-validation.json.
No new provider compute, human inference or raw-data downloads. Remaining
controller QA, strict final environment validation and release acceptance open.
