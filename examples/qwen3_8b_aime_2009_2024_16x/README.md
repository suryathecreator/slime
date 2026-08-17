# Qwen3-8B AIME 2009–2024, 16x sampled generation

This handoff produces a training-ready dataset from the official post-trained
`Qwen/Qwen3-8B` model (not `Qwen3-8B-Base`) at revision
`b968826d9c46dd6066d109eabc6255188de91218`.

## Exact contract

- Corpus: 2009 through 2024 inclusive, ordered newest to oldest, with AIME I
  problems 1–15 followed by AIME II problems 1–15 in every year.
- Shape: 480 unique problems and 16 completions per problem (7,680 rows).
- Primary source: `gneubig/aime-1983-2024` at revision
  `5d610df981dec508dd93d0a16333a029ac1739d8` (455 rows).
- Supplement: 25 exact problem statements from the legally licensed LIVE by
  Po-Shen Loh contest pages. Every page URL and fetched-content hash is pinned.
- Model mode: `enable_thinking=True`, one user message, no system message.
- Sampling: temperature 0.6, top-p 0.95, top-k 20, min-p 0.0.
- Context: 32,768 total tokens, with the completion budget computed as
  `32768 - rendered_prompt_tokens` for each problem.
- Stops: `<|endoftext|>` (151643) and `<|im_end|>` (151645); stop control tokens
  are excluded from the saved assistant response.
- Compute layout: 16 draws × four contiguous 120-problem shards = 64 H200
  generation tasks, preceded by a one-sample H200 canary. Each completion is
  atomically persisted, so a requeued task resumes without replacing completed
  samples.

The exact user message is:

```text
Please reason step by step, and put your final answer within \boxed{}.

Problem:
{problem}
```

## Scoring

Every saved response—including every length-cap response—is passed through the
pinned `aime_last_boxed_integer_scorer_v1`. The scorer scans the entire visible
assistant text for balanced, complete, nonempty `\boxed{...}` expressions,
selects the last one, removes only the box plus fixed presentation wrappers/math
delimiters, and accepts only an exact optional-`+` integer in 0–999. It has no
unboxed or heuristic fallback. For 2022 AIME II problem 8, the unchanged scorer
is run against both 080 and 081 because both were officially accepted, and a
match to either is correct.

The CPU audit independently re-scores every row, validates prompt/seed/model and
artifact hashes, and writes overall/per-draw/per-year/per-contest statistics.
Only a complete 7,680-row audit can enter the HF publisher. The publisher writes
16 Parquet shards plus the canonical problems, source manifests, exact scorer,
policy, statistics, and dataset card to
`suryadv/qwen3-8b-aime-2009-2024-16x`.

## Submission

```bash
bash examples/qwen3_8b_aime_2009_2024_16x/submit_hyak.sh --test-only
bash examples/qwen3_8b_aime_2009_2024_16x/submit_hyak.sh --submit
```

The submitter requires a clean worktree: push this handoff first, submit the job
chain, then commit and push the generated `SUBMISSION.json` metadata.
