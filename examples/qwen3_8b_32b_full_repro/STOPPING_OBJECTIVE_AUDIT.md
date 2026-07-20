# Stopping-objective audit

This file is populated after OPD. The audit reports rollout length
distributions, 16K cap hits, think closure, eligible final answers, EOS versus
`<|im_end|>`, answer events and corrections, terminal teacher/student
log-probabilities, and repetition indicators.

`audit_opd_rollouts.py` reads the preserved rollout tensors after training and
writes per-trajectory JSONL plus an aggregate JSON/Markdown report. Global IDs
combine the root rollout ID with within-rollout order; they never rely on a
local sample index that may reset. OpenThoughts responses are not treated as
verified gold, so correctness is explicitly unavailable rather than inferred.

The interpretation is fixed in advance: a 16K cutoff bounds compute but does
not teach stopping. Capped trajectories contain no synthetic EOS and remain in
the faithful OPD loss. Behavioral threshold violations are reported but do not
abort or modify the run.
