# AIME scorer v1-to-v2 rescore audit

All 510 completed AIME generations were rescored from their immutable
`generated_text` fields. The original v1 prediction and metric files remain
unchanged; v2 artifacts live under each evaluation's
`rescored/aime_answer_v2` directory.

| Artifact | Scorer v1 | Scorer v2 | Change |
|---|---:|---:|---:|
| Qwen3-1.7B base, 30×16 | 183/480 (38.125%) | 184/480 (38.3333%) | +1 correct (+0.2083 pp) |
| Qwen3-8B teacher, one sample | 21/30 (70%) | 21/30 (70%) | no score change |

The only grade flip is AIME problem 21, sample 2 (`flat_index=322`). Its final
candidate is `41 + 9 = 50`: v1 reported an integer parse failure, while v2
validates the complete equality using exact arithmetic and returns 50.

V2 also changes extraction or parse provenance on 83 other base rows and two
teacher rows without changing their grades. In particular, all six capped base
generations containing `</think>` now select only from the post-think region;
all six remain incorrect. Explicit retraction behavior is covered by synthetic
regression tests because no completed generation has an explicit retraction in
its eligible final-answer region.

Running the rescorer a second time produced byte-identical prediction, metric,
and diff files. Full source and output SHA-256 values are recorded in
`scorer_v2_rescore_20260718.json`.
