# Release checker failure-gate review

The source checker was read in full in diagnostic-release-controllers-review.md.
New tests/test_release_checker_boundaries.py directly executes its undecorated
function body with real temporary tar archives and filesystem artifacts. Modal
is never imported and no provider is contacted. Sixteen passing cases exercise
archive digest mismatch; traversal/absolute/symlink/directory rejection; extracted
manifest mismatch; build, virtualenv, installation and import failures; nonzero
test exit and skipped-test rejection; scientific rerun failure; and successful
report/source retention. Assertions verify that scientific execution is skipped
on earlier failures, installed-wheel source injection is disabled, warnings are
errors, exact source digest survives and stored JSON matches returned output.

Subprocess outcomes are controlled fixtures, not live pip/PRISM fault injection.
The successful real pinned cloud run remains separately authoritative for the
unchanged implementation. The actual detached-launch mount failure is retained
in its original report and was corrected by explicit mounting in the launcher.

No executable research code changed. The 491-file source is a verified test-only
extension of the 490-file strict-tested archive. Sixteen new cases pass against
the installed corrected library with warnings as errors. They do not constitute
a new full 1651-test cloud execution or whole-repository ship-ready verdict.
