# MATH-500 scorer V2 saved-generation audit

All 2,000 immutable generations were rescored. No model inference or continuation was performed.

| Stage | V1 | V2 | Incorrect -> correct | Correct -> incorrect |
|---|---:|---:|---:|---:|
| qwen3_8b_base | 235/500 | 373/500 (74.60%) | 138 | 0 |
| qwen3_32b_teacher | 476/500 | 482/500 (96.40%) | 6 | 0 |
| sft_200000 | 304/500 | 434/500 (86.80%) | 131 | 1 |
| opd_100pct | 429/500 | 440/500 (88.00%) | 11 | 0 |

## Audit findings

- Structural display delimiters no longer replace valid boxed answers.
- Adjacent boxes are one exact-cardinality multipart answer.
- Cap-hit doubt/checking language does not retract a complete answer; only an explicit unreplaced rejection does.
- A terminal strict prefix of a repeated scalar or multipart answer is treated as cap truncation, not a replacement.
- Natural-stop prose cannot supersede a stronger boxed or explicit answer event.
- Weak standalone lines are never accepted for cap-hit rows.

## Targeted audit rows

| Stage / row | V1 | V2 | V2 candidate | Finding |
|---|---:|---:|---|---|
| sft_200000 / 36 | correct | correct | `3, 5, 7` | A hypothetical 1, 2, 3 formatting example is not an answer event; the last eligible final answer remains 3, 5, 7. |
| sft_200000 / 355 | correct | incorrect | `none` | The cap tail contains only incidental calculation text around 100 and no strong answer event. |
| opd_100pct / 462 | incorrect | correct | `59` | The terminal 5 is a strict-prefix truncation of a repeated complete answer 59, so 59 is retained. |
| qwen3_32b_teacher / 26 | incorrect | correct | `144` | Later checking and a distant subproblem negation do not retract the explicit answer 144. |

Review count: 0. Aggregate gate: passed.
