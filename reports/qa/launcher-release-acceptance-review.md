# Launcher and release acceptance checkpoint

Read tools/run_final_human_jobs.py and the full tests in test_final_human_runner.py,
test_source_reproduction_runner.py and test_guarded_modal.py. Earlier complete
source review of guarded_modal.py remains recorded in tool-evidence-review.md.

| Obligation | Assertion evidence | Current result |
| --- | --- | --- |
| Keep original call through observation failure | tests/test_guarded_modal.py:114; tests/test_source_reproduction_runner.py:81 | Pass |
| Ambiguous attempt is never automatic retry | tests/test_final_human_runner.py:88; tests/test_source_reproduction_runner.py:101 | Pass |
| Lifetime cross-process exclusion | tests/test_final_human_runner.py:97; tests/test_guarded_modal.py:270 | Pass |
| Partial state, failed submit, sibling failure | tests/test_final_human_runner.py:131 | Pass |
| Source/result/freeze/user/cohort identity | tests/test_final_human_runner.py:13; tests/test_source_reproduction_runner.py:111 | Pass |
| Insufficient allowance prevents submission | tests/test_source_reproduction_runner.py:119 | Pass |
| Release archive/build/test/science failure gates | tests/test_release_checker_boundaries.py:50; tests/test_release_checker_boundaries.py:98 | Pass |

Local replay of these four modules passes eighty cases without skips under
warnings-as-errors. Provider adapters are controlled, while local child processes,
locks, filesystem writes and recovery assertions are real. Results and exact test
hashes: reports/preflight/launcher-release-acceptance-tests.json. Sixty-four cases
repeat existing coverage and sixteen are the new checker slice. No new source
implementation or provider job was introduced.

This closes the identified launcher/release failure-gate slice. It does not turn
legacy unguarded direct entrypoints into generally safe unattended schedulers or
establish whole-repository acceptance. Source release instructions must retain the
guarded launcher and original frozen-source requirements. Remaining release gate:
consolidate the whole-source QA verdict, rebuild the final candidate for the
sixteen-test extension, verify artifact closure, then publish permitted source.
