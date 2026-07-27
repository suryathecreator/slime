# Deferred MATH-500 results

No evaluation is part of the Tillicum training-only chain. Import externally
completed, verified bundles with
`../../qwen3_8b_openr1_math220k_masked_sft_eval_handoff/import_result_bundle.py`.
The importer creates one directory for each of `correct_only`,
`weighted_tau_0p20`, and `unmasked`. The shared base result lives in the 2K
example's `results/base_8b` directory and is reused in the final 40K table.

See
`../../qwen3_8b_openr1_math220k_masked_sft_eval_handoff/README.md`
for the exact checkpoint locations and frozen eval policy.
