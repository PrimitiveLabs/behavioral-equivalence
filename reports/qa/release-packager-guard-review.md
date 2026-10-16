# QA audit: release packager and local launcher guard

## Summary

Read tools/package_source.py and tools/guarded_modal.py in full, together with complete source-package and guard tests. No repository CLAUDE.md exists. This supersedes the specifically listed packager/guard gaps in the historical execution/release review, not the whole-source gate.

Scoped result: no identified gap remains in these two reviewed tools. Whole-repository ship-ready status remains unassessed. No implementation changed in this review.

## Permutation matrix

| Boundary or behavior | Evidence | Coverage |
| --- | --- | --- |
| Deterministic source archive, private-format exclusion and exact archived-byte manifest | tests/test_source_package.py:15,76 | filesystem integration |
| File/directory/ancestor/dangling symlinks and unexpected binary source | tests/test_source_package.py:44,58,69 | filesystem integration |
| Top files and explicit fixtures cannot be symlinks | tests/test_release_packager_boundaries.py, named fixture cases | filesystem integration |
| Cache/metadata exclusions | tests/test_release_packager_boundaries.py, cache case | filesystem integration |
| Actual CLI source pointer, archive digest/bytes, repeated identical archive | tests/test_release_packager_boundaries.py, CLI case | subprocess/filesystem integration |
| Original-call retention, removed metadata, ambiguous or failed submission | tests/test_guarded_modal.py, submission/recovery cases | real subprocess/filesystem, simulated provider boundary |
| Timeout, exception, interrupt and changed original handle | tests/test_guarded_modal.py, observation and handle cases | boundary unit plus persistence integration |
| Same controller/state locks and independent-controller concurrency | tests/test_guarded_modal.py, cross-process case | real concurrent subprocess integration |
| Literal argv, known protocol-derived native state, invalid paths/waits/protocols | tests/test_guarded_modal.py, named gate cases | unit and subprocess/filesystem integration |
| Legacy handle observes without CLI, private payload omitted, non-string status | tests/test_guarded_modal.py:316,341 and new release boundary cases | filesystem integration with simulated provider observations |
| CLI exit and provider adapter only reads supplied original handle | tests/test_guarded_modal.py:367,404 | boundary unit |

## Required test additions

Nine added cases in tests/test_release_packager_boundaries.py close the specifically identified CLI/fixture/cache/status assertions. The complete packager/guard group passes 51 tests. No remaining test addition is identified in this scoped review. Full available suite passes against the unchanged corrected wheel, with exact commands and test bytes in reports/preflight/release-guard-corrected-wheel-validation.json.

## Non-coverage findings and limits

The guard provides POSIX local coordination for invocations through it. It does not patch historical running processes, protect manual bypasses or provide distributed exactly-once submission. Existing native handles and ambiguity markers must be retained. Command/provider observations do not establish scientific acceptance. Packaging binds bytes read from the selected tree; release staging and scanning remain separate controls. No live provider fault injection, secret transfer or new cloud compute occurred.
