# Qwen2.5-7B Nous AIME mixed-outcome SFT

This experiment tests whether correct versus incorrect reasoning supervision
transfers differently as evaluation moves away from the exact trained problem
distribution. It independently full-finetunes four copies of
`Qwen/Qwen2.5-7B`: AIME 2024 correct, AIME 2024 incorrect, AIME 2025 correct,
and AIME 2025 incorrect. Every checkpoint starts from the same verified base
with a fresh AdamW optimizer and LR schedule; no checkpoint continues another.

## Selection rule

The pinned source is
`NousResearch/eval-Qwen3-235B-A22B-reasoning` revision
`67bba1eb0642bc9a39d6971fd6c149e6bbbbb0c8`. We first render every trajectory
in its complete Qwen2.5 training form and reject totals over 32,768 tokens
without truncation. We then rescore the retained assistant generation with
SLIME's whole-response last-balanced-`\boxed{}` AIME integer scorer.

For each year, the authoritative held-in set is every problem that has at least
one retained scorer-correct and one retained scorer-incorrect generation. We
take **all** retained incorrect generations from those mixed-outcome problems,
then conditionally uniformly subsample exactly the same number of distinct
correct generations from the exact same problems, requiring every problem to
appear. Correct and incorrect conditions therefore have the same total trace
count and identical unique-problem coverage, but their per-problem frequencies
may vary naturally.

The expected post-rescore diagnostic is 181 incorrect traces over AIME 2024
`doc_id` values `1,2,5,10,13,14,20,25,28,29`, and 268 over AIME 2025 values
`1,6,8,9,10,11,12,18,19,20,21,22,23,26,27,28,29`. These lists are not forced.
If the pinned scorer/cutoff produces a different mixed set, preparation records
the difference and uses the observed set: it never forces a stale expected ID
or drops a newly mixed problem.

Each fixed selected set is expanded to exactly 3,000 physical rows. For `N`
source traces, every trace gets `floor(3000/N)` copies and a seeded random set
of `3000 mod N` traces gets one extra. Under the expected counts, 2024 uses 16
copies each plus a 17th for 104 traces; 2025 uses 11 each plus a 12th for 52.
The 47 global batches consume 3,008 presentations, so the physical ordering
places the eight wrap rows on distinct lower-copy traces. Realized source-trace
exposure still differs by at most one.

Using only mixed-outcome problems removes problem identity and coverage as a
confound between correct and incorrect supervision. Same-year held-out problems
have a real selection shift because they tend toward all-correct/all-incorrect.
Same-year held-out and cross-year AIME are treated as two different
generalization checks; neither is assumed to be more distribution-aligned.

## Exact training and evaluation prompts

The source `/think` system message is deliberately omitted. The exact Nous
`query` bytes become one Qwen2.5 user message, and the tokenizer inserts its
default system turn. The assistant message is preserved byte-for-byte; no
whitespace normalization, think wrapper, answer suffix, or synthetic EOS is
added. Training renders:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{NOUS_QUERY_VERBATIM}<|im_end|>
<|im_start|>assistant
{NOUS_ASSISTANT_GENERATION_VERBATIM}<|im_end|>
```

System/user/control-prefix tokens have loss weight zero. Every assistant token
and `<|im_end|>` has weight one. The final template newline has weight zero.

For AIME 2024 `doc_id="1"`, the actual source query is:

```text
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\boxed{ANSWER}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$, and let $\overline{AD}$ intersect $\omega$ at $P$. If $AB=5$, $BC=9$, and $AC=10$, $AP$ can be written as the form $\frac{m}{n}$, where $m$ and $n$ are relatively prime integers. Find $m + n$.
```

Evaluation uses that identical query and identical assistant transition:

```text
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
Solve the following math problem efficiently and clearly.  The last line of your response should be of the following format: 'Therefore, the final answer is: $\boxed{ANSWER}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering.

Let $ABC$ be a triangle inscribed in circle $\omega$. Let the tangents to $\omega$ at $B$ and $C$ intersect at point $D$, and let $\overline{AD}$ intersect $\omega$ at $P$. If $AB=5$, $BC=9$, and $AC=10$, $AP$ can be written as the form $\frac{m}{n}$, where $m$ and $n$ are relatively prime integers. Find $m + n$.<|im_end|>
<|im_start|>assistant
```

The eval prompt stops there. No stored Nous answer is included. A validation
gate proves every transferred query is byte-identical to the source user
message and that its rendered token IDs equal the corresponding training prefix.

## Training recipe

All four variants use full BF16 SFT on four H200s with TP4/DP1/CP1/PP1 and
ZeRO-0. Global and rollout batch size are 64. The hard rendered-sequence cap is
32,768; dynamic microbatch packing uses a 16,384-token soft GPU bin, allowing a
longer complete sample to run alone. AdamW uses LR `5e-6`, betas `(0.9,0.95)`,
epsilon `1e-8`, weight decay `1e-4`, gradient norm `1.0`, 3% linear warmup from
zero, and cosine decay to `1e-6`. Training uses synchronous `train.py`, 2,048
token loss chunks, optimizer CPU offload, recomputed loss, and resume-safe full
state every 24 updates. One epoch is 47 updates and ends at iteration 46.

Preparation, a five-update runtime canary for each variant, four full trainings,
and finalization form a strict serial ten-job chain. Every delayed job pins the
submitted Git commit and clean worktree. Canary prompt-copy generation is
nonblocking, while decoded shifted-label/loss-weight, longest-sequence,
topology, schedule, checkpoint, and HF-shard checks fail closed.

```bash
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/submit_training_only.sh --dry-run
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/submit_training_only.sh --submit
```

## Hyak evaluation handoff

The handoff transfers four final HF checkpoints, the base only if its identity
is not already verified, two 30-problem eval JSONLs, and compact provenance. It
does not transfer selected/expanded/tokenized training datasets, raw Nous
generations, optimizer/full state, or canary payloads.

Hyak will create the evaluation scripts. Evaluate base and all four checkpoints
on all 60 prompts. For every checkpoint/prompt pair, draw 16 independent
completions with temperature 0.7, top-p 0.8, top-k -1, `n=1`, and seed
`1234 + 60 * draw_index + global_eval_index`. Score all draws independently and
average their 0/1 correctness indicators. This is a Monte Carlo estimator of
expected single-sample accuracy—not pass@16, best-of-16, or majority voting.
The full run is `5 × 60 × 16 = 4,800` completions.

For each year-trained correct/incorrect pair, report held-in mixed problems,
held-out same-year problems, same-year all-30 overall, and complete other-year
accuracy. Include incorrect-minus-correct deltas, the base reference, Monte
Carlo standard errors, valid-box and cap-hit rates, and length diagnostics.
