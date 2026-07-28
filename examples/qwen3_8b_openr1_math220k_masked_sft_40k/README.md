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

The active Tillicum completion launch is training-only. The finished 40K
correct-only checkpoint is retained; after every 2K checkpoint is trained,
the chain trains only `weighted_tau_0p20` and `unmasked`. Each training job
records a content-addressed manifest for its final Hugging Face checkpoint.
The exact paths are in
`../qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_sources.json`.

MATH-500 was deferred until checkpoint transfer and is now evaluated on the
destination system. The base and all three final 40K checkpoints use the exact
generation path already used by the completed 2K evaluations: instruction
before problem, greedy decoding, vLLM 0.10.2, and a request for 32,768 output
tokens within a 32,768-token model context. No explicit 64-token buffer or
manually calculated per-problem cap is used; vLLM naturally clamps generation
according to prompt length. Both stop IDs 151643 and 151645 are accepted.

The earlier SLIME evaluations used problem-before-instruction prompts, a
64-token buffer, and vLLM 0.24.0. Those differences are documentation only;
the historical SLIME driver is not given compatibility switches. All saved
2K, base, and 40K generations are scored verbatim with
`math500_strict_boxed_units_scorer_v4`, which normalizes recognized explicit
answer units before mathematical equivalence checking. Original Axolotl and
SLIME V3 decisions are retained as provenance. The base evaluation is
completed and rescored before the 40K jobs are released. The exact launcher,
artifact paths, and rescore procedure are documented in
`../qwen3_8b_openr1_math220k_masked_sft_eval_handoff/README.md`.

The authoritative launcher is
`../qwen3_8b_openr1_math220k_masked_sft_2k/submit_training_only.sh`. It
serializes 2K preparation, scoring, and all 11 2K full-SFT runs before the two
remaining 40K full-SFT runs, using strict `afterok` dependencies and at most
four GPUs active. It schedules no evaluation or report jobs.

Operational recovery: job 185501 completed 49 training steps but failed while
creating its first async torch_dist checkpoint because Python's AF_UNIX
manager socket inherited the long GPFS contract `TMPDIR`. The partial
checkpoint has no `latest_checkpointed_iteration.txt` and is not resumable.
`resubmit_after_tmpdir.sh` preserves that attempt, restarts correct-only from
the pinned 8B base with a short node-local temp path, and rewires the existing
eval 185502 so the already-submitted downstream chain was reused. That
historical eval chain is now superseded by the training-only launch; its
records remain as provenance.
