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

The AIME 2024 held-in set is every post-rescore mixed-outcome problem, and it
must contain exactly 10 problems. Its realized incorrect-trace count becomes
the matching target. For AIME 2025, preparation enumerates every unique
10-problem subset of the full mixed pool, assigns each combination a seeded
SHA-256 random priority, and accepts the first whose incorrect count differs
from the AIME 2024 target by at most epsilon. Epsilon is zero, so this is an
exact match. The resulting sample is uniformly randomized conditional on the
trace-count constraint; it is not an unconditional draw or the 10 problems
with the most errors.

For each selected year-specific set, we take **all** retained incorrect
generations from those 10 problems, then conditionally uniformly subsample
exactly the same number of distinct correct generations from the exact same
problems, requiring every problem to appear. Correct and incorrect conditions
therefore have the same total trace count and exact problem coverage, while
their per-problem frequencies may vary naturally.

The expected post-rescore result is 181 traces over AIME 2024 `doc_id` values
`1,2,5,10,13,14,20,25,28,29`. The full AIME 2025 mixed pool contains 17
problems; the deterministic seed-42 conditioned selection accepts candidate
359 (zero-based rank 358): `6,8,10,19,20,21,22,26,27,29`, also totaling 181
incorrect traces. These values are derived rather than forced, and validation
fails closed if the pinned source, scorer, or algorithm produces a different
diagnostic.

Each fixed selected set is expanded to exactly 3,000 physical rows. For `N`
source traces, every trace gets `floor(3000/N)` copies and a seeded random set
of `3000 mod N` traces gets one extra. With 181 traces in every condition, each
uses 16 copies per trace plus a 17th for 104 seeded traces.
The 47 global batches consume 3,008 presentations, so the physical ordering
places the eight wrap rows on distinct lower-copy traces. Realized source-trace
exposure still differs by at most one.

Using only mixed-outcome problems removes problem identity and coverage as a
confound between correct and incorrect supervision. Every variant now has 10
held-in problems and 181 distinct source traces before expansion; the literal
problem identities remain year-specific. Same-year held-out problems have a
real selection shift because they tend toward all-correct/all-incorrect.
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

The completed contract `2d556f01dbd853a1` is published under
[`results/training_2d556f01dbd853a1`](results/training_2d556f01dbd853a1).
That immutable compact record includes checkpoint identities, selection and
schedule audits, all 47 loss/gradient-norm observations for every run, and a
human-readable first/last-loss summary. It deliberately excludes model
weights, optimizer state, training datasets, raw source generations, and large
canary token dumps.

### Four-epoch repeat

The `four_epoch` profile (contract `2f2b580a2f52b1b7`) repeats this experiment from the same Qwen2.5-7B
base with the same four byte-identical 3,000-row datasets, prompts, loss
weights, 32K context, TP4 runtime, and evaluation contract. Only the training
horizon changes: every variant receives four epochs, 12,032 presentations,
188 optimizer updates, and a final snapshot at iteration 187. Each run starts
independently from base with a fresh optimizer and cosine schedule; it does not
continue the one-epoch checkpoint. The original one-epoch contract and results
remain immutable.

The completed four-epoch training record is published under
[`results/training_2f2b580a2f52b1b7`](results/training_2f2b580a2f52b1b7). It
contains all 188 loss and gradient-norm observations for every variant,
checkpoint identities, selection and schedule audits, and the finalized handoff
metadata. Large model, optimizer, dataset, and canary payloads remain excluded
from Git.

```bash
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/submit_training_only.sh --dry-run --four-epoch
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/submit_training_only.sh --submit --four-epoch
```

After finalization, transfer this profile with the same single-session handoff:

```bash
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/rsync_to_klone.sh --check --four-epoch
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/rsync_to_klone.sh --transfer --four-epoch
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/rsync_to_klone.sh --verify --four-epoch
```

On Hyak, the existing evaluator is reused against the new transferred root.
The published 960-completion one-epoch base bundle is reused only after its
checkpoint identity, prompts, datasets, scorer, coordinates, summary, and
artifact hashes match exactly. Only the four trained checkpoints run new
inference, producing 3,840 new completions and a 4,800-score final audit:

```bash
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/eval/submit_hyak.sh --test-only-shapes --four-epoch
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/eval/submit_hyak.sh --submit --four-epoch
```

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

For each year-trained correct/incorrect pair, report the 10 held-in mixed
problems, 20 held-out same-year problems, same-year all-30 overall, and complete
other-year accuracy. Include incorrect-minus-correct deltas, the base reference,
Monte Carlo standard errors, valid-box and cap-hit rates, and length diagnostics.

### Hyak evaluator

The Hyak implementation lives in [`eval`](eval) and consumes the transferred
`handoff/eval_contract.json` directly as its authoritative policy. It renders
each JSONL `query` byte-for-byte as one user message. The Qwen2.5 template adds
`You are a helpful assistant.` and the assistant transition; the Nous `/think`
message and stored source answer are not included.

Each of the five targets has a 16-element H200 array. Element `d` evaluates all
60 prompts with `temperature=0.7`, `top_p=0.8`, `top_k=-1`, `n=1`, and seed
`1234 + 60*d + global_eval_index`. The total context is 32,768 tokens and the
per-prompt generation allowance is `32768 - rendered_prompt_tokens`. Generation
may stop on Qwen `<|endoftext|>` (`151643`) or `<|im_end|>` (`151645`); neither
control token is included in the decoded assistant response.

Every completion, including a length-cap completion, is scored using the
contracted whole-response scorer at its pinned SHA-256. It chooses the last
complete, nonempty, brace-balanced `\boxed{...}`. After fixed presentation
wrapper removal, the content must be exactly one integer from 0 through 999.
There is no unboxed, heuristic, majority-vote, best-of-16, or pass@16 fallback.

```bash
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/eval/submit_hyak.sh --test-only-shapes
bash examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/eval/submit_hyak.sh --submit
```

The dependency graph is CPU preflight, one H200 runtime canary, five parallel
16-way generation arrays, five CPU target finalizers, and one final audit. The
final report contains the symmetric AIME 2024/AIME 2025 held-in, held-out,
same-year, and cross-year tables; paired incorrect-minus-correct deltas; Monte
Carlo standard errors; valid-box and cap-hit rates; and token-length summaries.

## Final evaluation results

The completed 4,800-completion evaluation is published under
[`results/aime_mixed_outcome_sampled_v1`](results/aime_mixed_outcome_sampled_v1).
It includes the human-readable report, immutable final audit, per-target
summaries, and all 4,800 compact per-completion score records. The report also
contains at-a-glance correct- and incorrect-trace generalization comparisons
against the matching base slices.

All five targets completed all 960 requested generations. The all-60 accuracy
was 6.25% for base, 9.27% for AIME 2024 correct, 4.69% for AIME 2024 incorrect,
11.98% for AIME 2025 correct, and 10.10% for AIME 2025 incorrect. Trained-model
cap-hit rates were 98.65--99.90%, so the low 6.35--23.33% valid-box rates and
the contracted policy of scoring every cap hit are material interpretation
caveats.
