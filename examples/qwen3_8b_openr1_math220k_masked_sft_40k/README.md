# Qwen3-8B OpenR1-Math 40K masked full-SFT completion

This example uses Slime's existing fast OPD-adjacent path to train
`Qwen/Qwen3-8B-Base` itself with full-parameter SFT. Axolotl-Masked-SFT is the
semantic reference for selection and inverse-margin weighting; it is not the
trainer.

The three variants start independently from the pinned base and train for one
40,000-trace epoch at global/rollout batch 200:

1. `correct_only`: 40,000 unique correct traces.
2. `weighted_tau_0p20`: the same ordered 20,000 correct + 20,000 wrong rows as
   unmasked. Wrong-token margin is
   `logp(answer-conditioned) - logp(unconditioned)`. Weight is 1 for
   nonnegative margin and `1 / (1 + (-margin / 0.20))` otherwise.
3. `unmasked`: the mixed rows with every assistant token at weight 1.

Training rows are pre-tokenized. All assistant tokens, including literal think
tags, have positive weight. `<|im_end|>` has weight 1; template trailing
whitespace has weight zero; no synthetic `<|endoftext|>` is appended.
Fractional weight mass is encoded for Megatron's integer schedule counter with
a fixed-point scale of 256. Numerator and denominator are scaled together, so
the weighted mean is preserved and binary masks remain exact.

This completes the already-started run. Its trace preparation, base-model
margin scores, and weighted dataset finished successfully and remain
immutable under the original scratch contract. Training failed before the
first optimizer step when Megatron tried to add a float loss-weight sum to an
integer token counter. The completion chain reuses the hash-pinned data and
writes checkpoints and evaluations under a new contract root.

MATH-500 uses the latest borrowed OPD
`math500_strict_boxed_scorer_v3`, greedy decoding, and the dynamic budget
`32768 - rendered_prompt_tokens - 64`. Both stop IDs 151643 and 151645 are
accepted. The base 8B evaluation is rerun once and shared with the 2K tables.

The authoritative launcher is
`../qwen3_8b_openr1_math220k_masked_sft_2k/submit_completion_chain.sh`. It
serializes the base eval, every 40K train/eval pair, and the 2K suite with
strict `afterok` dependencies and at most four GPUs active.

Operational recovery: job 185501 completed 49 training steps but failed while
creating its first async torch_dist checkpoint because Python's AF_UNIX
manager socket inherited the long GPFS contract `TMPDIR`. The partial
checkpoint has no `latest_checkpointed_iteration.txt` and is not resumable.
`resubmit_after_tmpdir.sh` preserves that attempt, restarts correct-only from
the pinned 8B base with a short node-local temp path, and rewires the existing
eval 185502 so the already-submitted downstream chain is reused.
