# Hyak evaluation contract

Evaluate the verified Qwen2.5-7B base and all four trained checkpoints on all
60 transferred prompts. Use the exact `query` as one Qwen2.5 user message with
the tokenizer's default `You are a helpful assistant.` system turn and
`add_generation_prompt=True`. Do not include the Nous `/think` system message
or any stored source generation.

For every checkpoint/prompt pair, draw 16 independent completions with
temperature 0.7, top-p 0.8, top-k -1, n=1, and seed
`1234 + 60 * draw_index + global_eval_index` for draw indices 0 through 15.
This estimates expected single-sample accuracy: score every completion and
average the 0/1 indicators. It is not pass@16, best-of-16, or majority vote.

Use `aime_last_boxed_integer_scorer_v1` from the SLIME repository. It scores the
whole completion, including cap hits, using the last complete nonempty balanced
`\boxed{...}` containing one integer from 0 through 999. The total context
ceiling is 32768; the response allowance is 32768 minus rendered prompt tokens.

Observed splits:

- AIME 2024: 10 held-in mixed-outcome problems,
  20 held-out problems.
- AIME 2025: 10 held-in mixed-outcome problems,
  20 held-out problems.

For the AIME 2024-trained pair, report AIME 2024 held-in, held-out, all-30
overall, and complete AIME 2025 accuracy. Report the symmetric four values for
the AIME 2025-trained pair. The two primary tables have correct/incorrect
columns and in-distribution plus two distinct out-of-distribution checks. Also
report incorrect-minus-correct deltas, the base reference, Monte Carlo standard
errors, valid-box rates, cap-hit rates, and generation-length diagnostics.
Same-year held-out and cross-year are different generalization checks; neither
is assumed to be more distribution-aligned.
