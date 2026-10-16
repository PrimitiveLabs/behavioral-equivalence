# Primary and supplementary analysis freeze tools scoped review

Read tools/freeze_human_analysis.py and tools/freeze_supplementary.py in full,
with test_final_human_job_plan.py and test_supplementary_freeze.py. Added
11 cases in test_freeze_tool_boundaries.py. Generic QA conventions apply;
no root CLAUDE.md exists. Whole-repository ship-ready acceptance remains unproven.

Primary freeze binds all four controller snapshots and configuration hashes,
case/audit/comparator/transition-uncertainty artifacts, statistical specifications
and fixed fifty-job matrix. Every bound remote artifact is hash checked, then the
analysis-freeze verifier checks each phase. Whole-matrix reservation is checked
against the existing allowance. Default mode validates without uploading or
reserving; publish mode checks for an existing destination, uploads/readbacks
and then reserves planned jobs. It is not a distributed transaction or WORM store.
A concurrent publisher is outside its single-launcher execution contract.
Publication-before-reservation and partial local recovery remain operational
failure risks; old freezes must not be republished to repair them. Existing
artifacts/ledger identities must be inspected instead. No publish invoked here.
The lock explicitly discloses known pre-test findings and is internal, not
external preregistration or blinded absence of all prior knowledge.

Supplementary freeze requires thirty trained heads and ten calibration reports,
fixed endpoint order, original source digests, all five seeds/both protocols and
120 replayed certificates per calibration report. Existing differing config bytes
cannot be overwritten; identical reruns are idempotent. This exploratory study is
explicitly unable to replace or rescue primary click/intervention hypotheses.

Reconstructed the actual supplementary config byte-for-byte from retained reports
and matched every primary job field to the retained fifty-job plan. Its historical
$21.56144 reservation total is not spend, a new reservation or a user-global cap.
Evidence: reports/preflight/analysis-freeze-tool-replay.json. Focused tests:
19 passed with warnings-as-errors. Full local suite: 1482 passed, three Torch
skips, three retained horizon warnings. Ruff passes. Validation:
reports/preflight/freeze-tool-local-validation.json.

Remote upload failures and distributed publication are not executed by these
local tests. This review does not assert full remote fault-path acceptance.
All tools now have scoped review records; controller QA, strict final environment
validation, whole-source acceptance and final release remain open. No provider
compute, model calls, historical freeze changes or original result mutations.
