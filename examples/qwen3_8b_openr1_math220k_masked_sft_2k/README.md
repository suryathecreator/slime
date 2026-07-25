# Qwen3-8B OpenR1-Math 2K masked full SFT in Slime

This example uses the existing speedy Slime/OPD SFT pipeline to train
`Qwen/Qwen3-8B-Base` itself: every run is full-parameter SFT, not LoRA and not
an Axolotl training job. The Axolotl experiment contributes the frozen
unpaired seed-42 trace identities and the masking definitions.

Selection is fixed once. Correct-only uses 2,000 correct traces. Every mixed
variant uses the same 1,000 correct and independently sampled 1,000 wrong
traces; the correct and wrong prompt IDs are disjoint. Only the wrong-token
loss weights differ between mixed variants.

The training recipe is one epoch, global/rollout batch 200, LR `1e-6`,
constant schedule, weight decay `0.1`, zero attention/hidden dropout, and 32K
training context. Correct tokens always have weight 1. Regular margin keeps a
wrong token iff `logp(conditioned) - logp(unconditioned) > 0`; inverse-margin
weighting uses `1` for nonnegative margins and
`1 / (1 + (-margin / tau))` otherwise. Probability-ratio masking uses
`min(1, exp(-margin))` as the mask probability. Random and probabilistic
decisions use stable SHA-256 draws keyed by seed, trace ID, and token index.

Every literal `<think>` or `</think>` tag token present in the source and the
training `<|im_end|>` are always supervised. No synthetic `<|endoftext|>`
target is appended, and the template newline after `<|im_end|>` has weight
zero. The frozen selection includes an empty `<think></think>` block, 57
source rows without a closing tag, and one row with two closing tags. They are
retained and audited instead of changing the canonical trace set; all literal
tag tokens that are present remain supervised.

MATH-500 uses the latest borrowed OPD strict boxed scorer, greedy decoding,
and a per-prompt generation budget of
`32768 - rendered_prompt_tokens - 64`. Both `<|im_end|>` and
`<|endoftext|>` stop IDs are accepted. Base, correct-only, and mixed-unmasked
will be evaluated once and reused verbatim in each result table.

The active Tillicum launcher is `submit_training_only.sh`. It submits no eval
or report jobs: it serializes trace preparation, margin scoring, and all 11
2K full-SFT jobs before either remaining 40K job. Priority after the two
control trains is random 70%, inverse-margin tau 0.20, inverse-margin tau
0.05, then the remaining variants. The same ordered traces are reused across
every mixed variant, and strict `afterok` dependencies keep the maximum
concurrent allocation at four H200s.

Each completed train writes a portable checkpoint manifest under
`/gpfs/scrubbed/suryadv/slime-qwen3-8b-openr1-masked-sft-2k/v1/5c3230a293f7acb2/handoff/checkpoints/`.
The final checkpoints, their exact source paths, the one-H200 resumable eval
wrapper, all frozen eval settings, and the fail-closed result importer are in
`../qwen3_8b_openr1_math220k_masked_sft_eval_handoff/`. Evaluation is planned
on a different system after checkpoint transfer.

All 11 2K full-SFT jobs completed successfully. `TRAINING_STATUS.json`
records their Slurm completion times, final iteration, and content-addressed
checkpoint identities.
