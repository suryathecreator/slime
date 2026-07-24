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
are evaluated once and reused verbatim in each result table.

The submission script deliberately serializes every 4-GPU job and places each
evaluation immediately after its training job. Priority after the two control
trains is random 70%, inverse-margin tau 0.20, inverse-margin tau 0.05, then
the remaining variants.
