# Deferred MATH-500 results

No evaluation is part of the Tillicum training-only chain. Import externally
completed, verified bundles with
`../../qwen3_8b_openr1_math220k_masked_sft_eval_handoff/import_result_bundle.py`.
The importer creates one directory per checkpoint, including `base_8b`,
`correct_only`, `unmasked`, all five random-token masks, both inverse-margin
weights, regular margin masking, and probability-ratio masking.

`base_8b`, `correct_only`, and `unmasked` are evaluated once and reused
verbatim in every 2K comparison table. See
`../../qwen3_8b_openr1_math220k_masked_sft_eval_handoff/README.md`
for checkpoint sources and the frozen eval policy.
