# Pilot uncertainty, refinement and CPU preflight controller review

Read benchmark_cpu_training.py, refinement_pilot.py, uncertainty_pilot.py,
compare_uncertainty.py and reproduce_teacher_cpu.py in full. No source edits.
Generic QA defaults apply; no root CLAUDE.md exists. Whole-source readiness open.

CPU benchmark uses synthetic full-size tensors, fixed warmup/timing steps and
finite-loss checks. Hardware bfloat16 flags and actual autocast mode are distinct
fields. Synthetic throughput is not human predictive quality or a controlled
claim of faster final training. Its legacy direct entrypoint lacks later budget
and retained-handle launch controls; not invoked here.

Refinement pilot compares equal environment-evaluation budgets over a small
one-dimensional intervention family. It explicitly lacks matched-capacity/data
controls, uses few seeds and makes no human intervention claim. Later full
experiments, not this pilot, determine confirmatory synthetic conclusions.
Uncertainty pilots use paired fixed iid row draws and rectangular per-step robust
bounds, comparing point/kernel containment as separate fields. Their floating
calculations and literal epsilon==0 ground-truth assertion are legacy engineering
checks. The later serialized-radius/integer certificate logic, rather than this
literal assertion, supports release soundness claims. Original pilot outputs must
not be silently reinterpreted as corrected-source confirmatory executions.

CPU teacher reproduction requires source-file equality, rebuilds pilot histories,
loads weights safely and checks case/target/user alignment. It returns diagnostic
booleans and manual-head/probability differences without changing saved outputs
or using those flags to accept a model. Status complete means diagnosis finished,
not all comparisons succeeded. Its np.allclose call retains a default relative
term in addition to atol=1e-6; this is not the final strict forecast audit.

Thirteen new tests execute real controller function bodies without Modal:
unsafe run-ID rejection, one-family small synthetic refinement across all policies,
equal eight-query budgets, nondecreasing partition capacity, finite probability
errors, explicit pilot limitations and refusal to overwrite the run directory.
This test-restricted generator is not a new full pilot or human result. Focused
suite: 13 passed with warnings-as-errors. Full local suite: 1553 passed, three
Torch skips and three retained horizon warnings. Ruff passes. Evidence:
reports/preflight/pilot-controller-local-validation.json.

No cloud work, model provider calls or human inference. Remaining controller QA,
strict final environment validation and final release acceptance stay open.
