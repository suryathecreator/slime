# Risks and unresolved assumptions

## Sourced facts

The public result uses Qwen3-8B-Base, part of OpenThoughts3, Qwen3-32B OPD,
four samples per prompt, a 16,384 rollout cap, LR 1e-6, and pure OPD
coefficients. Issue #1493 supplies the approximate cleanup counts, 200K/100K
usage, one SFT epoch, and roughly 1.5K OPD prompts. Exact public SFT batch,
optimizer, scheduler, split code, detector, evaluator, and scorer are absent.

## Pinned assumptions and deviations

- Lingua 2.2.0 whole-text classification at confidence 0.50 is the simple
  deterministic interpretation of “mixed language.” The count gate may fail;
  it will not be weakened automatically.
- Splitting is by seed-1234 shuffled raw eligible rows, not prompt groups. Text
  duplicates may cross SFT and OPD. This follows the final local contract and
  differs from a grouped leakage-control design.
- There is deliberately no MATH-500/OpenThoughts3 contamination check.
  MATH-500 itself stays read-only and unchanged.
- SFT batch/optimizer/parallelism are local hardware choices. OPD temperature
  0.8 and top-k -1 are requested deviations from the public script's stated
  temperature 1.0 setup.
- Four-H200 colocated OPD reuses the successful local 3 actor/rollout plus one
  teacher layout; it differs from the public eight-GPU example.

## Execution risks

- Cleanup language counts may fall outside ±5%, which blocks all downstream
  jobs by design.
- SFT full-state and eight weight snapshots may require about 250 GB; OPD may
  require about 180 GB plus rollout tensors. Checkpoint validation happens
  before older full states are pruned.
- A 16K cutoff can still expose nontermination. Those metrics are reported but
  do not abort or alter OPD under the final instruction.
- Teacher startup, model conversion compatibility, optimizer resume, and H200
  memory pressure remain infrastructure failure points. Failures preserve
  artifacts and block dependent jobs.
- Queue time and stochastic evaluation variance may extend the multi-day chain.
  The 76% and 94% values are targets, never tuning constraints.
