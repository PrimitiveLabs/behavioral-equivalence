# Early cloud data, pilot and coverage controllers scoped review

Read cloud/coverage_summary.py, nominal_failure_counts.py, analyze_training.py,
data.py and audit_data.py in full. No source changes. Generic QA defaults apply;
no root CLAUDE.md exists. Whole-repository acceptance remains unproven.

Coverage summary validates the campaign path component, binds the input summary
hash and stores its analysis source. It delegates the fixed-world Monte Carlo
calculation to the reviewed library. Its early direct entrypoint lacks later
campaign ledger/handle controls and must not be used as a general autonomous
launcher. Nominal failure classification binds the exact full manifest hash,
checks full evaluation/cell counts and partitions containment/proven-noncoverage/
unresolved cases without treating numerical ambiguity as a proven failure.
Its entrypoint reserves bounded CPU/memory timeout costs and attaches retained
handles. Failure to observe a handle is not authority to replace its job.

Pilot training analysis uses paired original-user NLL comparisons and labels
interaction-weighted calibration as descriptive. It explicitly removes the
noninferiority success flag because the comparison is not compression. Its
single fitted seed and pilot timing limits cannot support confirmatory claims.
Pilot data ingestion verifies the archive checksum before rename or reuse and
stores CSV schemas, not user rows, in the returned report. The archive_license
field describes deposit metadata only: the embedded 1K archive/repository CC-BY-SA
license is separately retained in data/license-register.json. Do not transfer
27K permissions to the 1K archive or redistribute raw data through this release.
Pilot audit reconstructs bundle counts, chronology/user assignments and declared
availability proxies, explicitly listing unestablished timing/dedup/causal claims.
It is distinct from the later full 27K confirmatory audit.

Ten new tests compile actual controller function bodies with Modal decorators
removed and explicit temporary filesystem/volume bindings. These validate local
path rejection, frozen hash/status/count checks, contradictory numeric categories,
no commit on failure and output hash readback. They do not test Modal transport,
image builds, real remote mounts, concurrency or network ingestion. No jobs were
launched. Full local suite: 1492 passed, three Torch skips, three retained horizon
warnings. Ruff passes. Evidence:
reports/preflight/cloud-summary-local-validation.json.

Remaining controllers, strict final environment validation and final release
acceptance remain open. Scoped review links are not whole-source ship-ready gates.
