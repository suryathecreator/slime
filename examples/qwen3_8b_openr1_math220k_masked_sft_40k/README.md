# Qwen3-8B OpenR1-Math 40K masked full SFT

This example uses Slime's existing fast OPD-adjacent SFT path to **train the
model itself with full-parameter SFT**. Axolotl-Masked-SFT is a semantic
reference for selection and inverse-margin weighting; Axolotl is not used as
the trainer and is not modified.

The three serial variants each start independently from the pinned
`Qwen/Qwen3-8B-Base` checkpoint and train for one 40,000-trace epoch (200
updates at global batch 200):

1. `correct_only`: 40,000 unique correct traces.
2. `weighted_tau_0p20`: the exact same ordered 20,000 correct + 20,000 wrong
   examples as the unmasked run. Wrong-token margin is
   `logp(answer-conditioned) - logp(unconditioned)` under the frozen base.
   Its weight is 1 for nonnegative margin and
   `1 / (1 + (-margin / 0.20))` otherwise.
3. `unmasked`: the same mixed rows and order, with every assistant token at
   weight 1.

Selection uses seed 42, the Axolotl correctness-field priority, one seeded
trace per problem, the source `is_reasoning_complete` flag, and disjoint
correct/wrong problem IDs. Every trace must contain exactly one nonempty
`<think>...</think>` block plus a nonempty answer tail, contain no embedded
chat controls, and fit the complete 32,768-token training target without
truncation. Wrong traces must also fit the answer-conditioned scoring prompt.

Training rows are pre-tokenized to preserve exact token/weight alignment.
Prompt/template-prefix weights are zero. All assistant tokens—including both
think tags—have positive weight. `<|im_end|>` is supervised at weight 1;
template trailing whitespace is zero-weight; no synthetic `<|endoftext|>` is
appended. Slime's per-token SFT loss divides the weighted token-loss sum by the
sum of weights.

MATH-500 uses the existing full-reproduction vLLM evaluator and
`math500_event_scorer_v1`: temperature 0.8, top-p 0.7, top-k -1, one sample,
seed `1234 + row`, 16,384 max new tokens, and both terminal IDs 151643 and
151645. The already completed and hash-pinned 8B-base result is reused rather
than rerun.

Run `./submit_chain.sh` only from the clean, pushed
`qwen3-8b-openr1-masked-sft-40k` branch. Large data, checkpoints, raw scores,
and generations stay under the contract-hashed scratch root; the final compact
results are copied back here and pushed after completion.
