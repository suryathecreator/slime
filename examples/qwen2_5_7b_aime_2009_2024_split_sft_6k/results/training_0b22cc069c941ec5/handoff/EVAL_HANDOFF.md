# Hyak evaluation handoff

Evaluate the verified Qwen2.5-7B base and all four trained checkpoints on both
transferred 106-problem mixed-outcome sets. The files are `held_in_106.jsonl`
and `held_out_106.jsonl`; together they are a disjoint, exhaustive partition of
the 212 AIME 2009–2024 problems with both correct and incorrect source traces.

Use each row's exact `rendered_prompt` bytes directly. It has exactly one user
turn and no system turn:

```text
<|im_start|>user
Please reason step by step, and put your final answer within \boxed{}.

Problem:
{problem}<|im_end|>
<|im_start|>assistant
```

Do not call the Qwen2.5 default chat template, because it would insert `You are
a helpful assistant.` and would differ from both source generation and
training. Do not insert a stored response.

Use `aime_last_boxed_integer_scorer_v1` from SLIME. Score every generated
completion, including cap hits, against `accepted_answer_integers`. The scorer
uses the last complete nonempty balanced `\boxed{...}` and requires one
integer in 0–999. Preserve the two-accepted-answer handling represented in the
transferred row. Sampling count and decoding parameters are intentionally not
pinned here; the Hyak evaluation implementation will pin and record those.

Produce two correct-versus-incorrect tables:

1. The two held-in-trained checkpoints evaluated on held-in (exact training
   problem distribution) and held-out (unseen problems).
2. The two held-out-trained checkpoints evaluated on held-out (exact training
   problem distribution) and held-in (unseen problems).

For every cell report accuracy, valid-box rate, cap-hit rate, and response-length
diagnostics. Include the base on both 106-problem sets and report
incorrect-minus-correct plus trained-minus-base deltas. Training datasets and
optimizer states are not part of this handoff.
