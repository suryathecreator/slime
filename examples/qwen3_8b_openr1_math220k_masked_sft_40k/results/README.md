# MATH-500 results

No evaluation was part of the Tillicum training-only chain. The transferred
final checkpoints are evaluated with the exact current policy used by the
completed 2K runs. One result directory is created for each of `correct_only`,
`weighted_tau_0p20`, and `unmasked`; the shared base result lives in the 2K
example's `results/base_8b` directory.

Native Axolotl decisions are retained as provenance. Comparable reported
scores live under each result's
`rescored/math500_strict_boxed_units_scorer_v4` directory and are computed
from the saved generation without further inference.

See
`../../qwen3_8b_openr1_math220k_masked_sft_eval_handoff/README.md`
for the exact checkpoint locations and frozen eval policy.
