# Qwen2.5-7B AIME 2009–2024 split-control SFT

This experiment trains four independent full-SFT Qwen2.5-7B-Base models for
exactly two epochs. It uses the audited
`suryadv/qwen3-8b-aime-2009-2024-16x` dataset at immutable revision
`c34143c92856d9b90fdb9c808c7e66070dcab9a1`: 480 problems and 16 Qwen3-8B
generations per problem, or 7,680 source traces.

## Selection and controls

Of the 480 problems, 212 are mixed-outcome: their 16 source generations contain
at least one scorer-correct and at least one scorer-incorrect response. The
remaining 241 all-correct and 27 all-incorrect problems are excluded from both
training and evaluation.

With seed 42, the 212 mixed problem IDs are assigned a SHA-256 random priority
using domain `mixed-problem-split-v1`. The first 106 form `held_in`; the
complementary 106 form `held_out`. This uniform deterministic split realizes
508 incorrect and 1,188 correct traces in held-in, and 591 incorrect and 1,105
correct traces in held-out.

The shared source target is `N=508` distinct traces over the exact same `K=106`
problem IDs in every condition:

1. Held-in incorrect retains all 508 incorrect traces.
2. Held-out incorrect selects one seeded anchor trace from every problem, then
   fills from a seeded ordering of unused incorrect traces until it has 508.
3. Each correct condition likewise selects one seeded anchor trace from every
   problem, then fills from unused correct traces until it has 508.
4. Selection is without replacement, so all four source sets contain 508
   distinct traces and cover all 106 problems. Within a split, correct and
   incorrect use exactly the same problem IDs.
5. Expand each fixed source set to exactly 6,000 rows without unconstrained
   resampling. Every source trace receives `floor(6000/N)` copies, and exactly
   `6000 mod N` seeded traces receive one additional copy. The 6,000 instances
   are deterministically shuffled. Concretely, all 508 traces receive 11 copies
   and exactly 412 receive a twelfth copy.

This creates four variants:

- `held_in_correct_6000`
- `held_in_incorrect_6000`
- `held_out_correct_6000`
- `held_out_incorrect_6000`

All four start independently from the same base model with a fresh AdamW
optimizer. There is no continuation or shared trained parent.

No source trajectory is rejected or truncated because of length, including
cap-hit and unfinished source generations. The hard training ceiling is 32,896
tokens, chosen as capacity for the complete pinned corpus; the preparation job
fails if even one fully rendered source trace does not fit. The 16,384-token
dynamic-batching value is a soft packing budget: a longer complete sample is
scheduled alone, not filtered.

## Exact prompt and training record

The source generation used one user message and **no system message**. Its
verbatim query was:

```text
Please reason step by step, and put your final answer within \boxed{}.

Problem:
{problem}
```

Training consumes the source row's `prompt`, `rendered_prompt`, and `response`
bytes unchanged. The complete rendered record is:

```text
<|im_start|>user
Please reason step by step, and put your final answer within \boxed{}.

Problem:
{problem}<|im_end|>
<|im_start|>assistant
{SOURCE_RESPONSE_VERBATIM}<|im_end|>
```

The prompt is context-only. Every source assistant token and the terminal
`<|im_end|>` receive loss weight 1; the final template newline receives weight
0. No synthetic `<|endoftext|>` is appended. We manually tokenize the pinned
source `rendered_prompt` instead of calling Qwen2.5's default chat template,
because that default would insert `You are a helpful assistant.` and would no
longer match the generation format. Preparation and both phases of every
canary audit the exact token IDs, shifted labels, response boundary, and loss
weights realized by the weighted-SFT adapter.

## Training recipe

- Model: `Qwen/Qwen2.5-7B` base revision
  `d149729398750b98c0af14eb82c78cfe92750796`
- Full-parameter BF16 SFT, four H200s, TP=4, DP=1, CP=1, PP=1, ZeRO=0
- 6,000 rows × 2 epochs; global/rollout batch size 64
- `ceil(12000 / 64) = 188` optimizer updates, iterations 0–187, with 12,032
  realized presentations (32 wrap presentations)
- AdamW: LR `5e-6`, cosine to `1e-6`, linear warmup fraction `0.03` from 0,
  betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `1e-4`, gradient norm 1.0
- Sequence ceiling 32,896; dynamic packing soft budget 16,384 tokens/GPU
- BF16, CPU optimizer offload, recomputed loss, 2,048-token loss chunks
- Synchronous `train.py` and full optimizer plus dataset-state checkpoints every
  24 updates for exact requeue/resume; only the final HF snapshot is handed off

The Slurm DAG is a strict ten-job `afterok` chain: one 1-H200 preparation job,
four serial 4-H200 five-update canaries, four serial 4-H200 full trainings, and
one 1-H200 finalizer. At most four H200s can be allocated at once. Every job
runs from a detached, clean worktree pinned to the pushed implementation commit,
so later provenance commits cannot alter queued training code.

## Future Hyak evaluation and handoff

The finalizer prepares the four trained HF checkpoints, the base comparison
inventory, `held_in_106.jsonl`, `held_out_106.jsonl`, exact prompt/scorer
instructions, all loss histories, and compact provenance. It deliberately does
not transfer selected/expanded training JSONL or optimizer state.

Hyak should evaluate every checkpoint and the base on both 106-problem mixed-only sets,
using each row's exact `rendered_prompt` directly with no system message. The
held-in-trained correct/incorrect pair is compared on held-in (exact problem
distribution) and held-out (unseen problems); the held-out-trained pair is
compared symmetrically. Use SLIME's `aime_last_boxed_integer_scorer_v1`, score
cap hits too, and report accuracy, valid-box rate, cap-hit rate, response-length
statistics, incorrect-minus-correct deltas, and trained-minus-base deltas.
Sampling and decoding settings will be pinned by the later Hyak evaluation
implementation and are intentionally not invented by this training package.

## Commands

After staging the pinned source files and pushing a clean implementation:

```bash
bash examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/submit_training_only.sh --dry-run
bash examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/submit_training_only.sh --submit
```

After finalization:

```bash
bash examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/rsync_to_klone.sh --check
bash examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/rsync_to_klone.sh --transfer
bash examples/qwen2_5_7b_aime_2009_2024_split_sft_6k/rsync_to_klone.sh --verify
```
