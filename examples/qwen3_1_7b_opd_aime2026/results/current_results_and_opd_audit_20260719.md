# Current AIME results and OPD objective audit

This snapshot records every completed Qwen3-1.7B/Qwen3-8B AIME evaluation
available before the 16k-rollout rerun. Large prediction files, rollout tensors,
checkpoints, and logs remain immutable in scratch storage; their SHA-256 values
are recorded in the companion JSON file.

## Completed evaluations

| Model/checkpoint | Evaluation | Scorer | Result | Cap hits | Mean generated tokens |
|---|---|---|---:|---:|---:|
| Qwen3-1.7B base | 30 problems x 16 samples | `aime_answer_v1` | 183/480 (38.125%) | 50 | 17,417.742 |
| Qwen3-1.7B base | immutable rescore of the same 480 generations | `aime_answer_v2` | 184/480 (38.3333%) | 50 | 17,417.742 |
| Qwen3-8B teacher | one sample per problem | `aime_answer_v1` | 21/30 (70%) | 1 | 15,410.733 |
| Qwen3-8B teacher | immutable rescore of the same 30 generations | `aime_answer_v2` | 21/30 (70%) | 1 | 15,410.733 |
| Qwen3-1.7B OPD after 2,048 prompts, iter 15 | one sample per problem | `aime_answer_v2` | 11/30 (36.6667%) | 7 | 21,886.1 |

The base v1-to-v2 change is one newly correct generation: AIME problem 21,
sample 2 (`flat_index=322`), whose final candidate is `41 + 9 = 50`. The
teacher score does not change. These are immutable rescoring results; the
generation text was not rerun.

### OPD problem 12 diagnostic

The official scorer-v2 grade for AIME problem 12 remains incorrect. The
31,744-token capped generation contains the correct `\boxed{161}` immediately
before `</think>`. It then starts a new visible solution after `</think>` and
hits the rollout cap before producing an answer in the eligible final
post-think region. Consequently scorer v2 reports
`no_eligible_answer_event`, as designed.

An answer-anywhere diagnostic would count this row and raise the OPD result to
12/30. That number is diagnostic only: the official result remains 11/30 and
the scorer is unchanged.

## OPD stopping-objective audit

The 16 checkpoint-producing rollout tensors cover 2,048 samples and
22,372,735 teacher-forced response target positions.

- 2,015 samples terminate with the configured stop token and 33 are capped at
  31,744 response tokens.
- Every response target, including all 2,015 stop targets, has `loss_mask=1`;
  all stop-token teacher log probabilities are finite.
- Teacher target IDs align exactly with the student response targets for all
  2,048 samples.
- 2,022 samples contain `</think>` and every close-tag target is unmasked.
  Seven capped samples close thinking before the cap; 26 cap before closing it.
- Final-answer tokens after `</think>` receive the ordinary token loss weight.
  The post-close region contributes 11.36% of response target positions.
- No sample contains an artificial second `<|im_start|>assistant` segment.
- A capped boundary adds no synthetic EOS/stop target. Therefore capped
  sequences train on their observed prefix but provide no stopping event.

EOS/stop, `</think>`, and final-answer tokens are present in the objective and
are not masked. The practical issue is weak and censored stopping supervision,
not a missing-token or chat-template bug: each single stop or close-tag target
has only about 0.0156% of the effective sample-averaged token objective, while
capped samples contribute no boundary stop target.

## Failed 3,072-prompt continuation diagnostic

Job `179552` preserved two rollout tensors before failing:

| Rollout | Samples | Mean response tokens | Cap rate | Outcome |
|---:|---:|---:|---:|---|
| 16 | 128 | 12,909.656 | 0.78125% | actor update completed |
| 17 | 128 | 13,401.789 | 6.25% | generated and teacher-scored; actor update OOM |

Across the 256 saved samples, 247 terminate normally and nine are capped. All
response targets are unmasked and exactly teacher-aligned; all nine capped
samples lack `</think>`. The failure occurred in fused cross-entropy backward:
PyTorch requested 580 MiB with 463.56 MiB free while the process held
138.42 GiB. These files are diagnostics only and will not be reused by the
fresh 16k-cap rerun.

Pending descendants `179553` and `179554` were obsolete because their
dependency root failed; they are canceled as part of the rerun submission.
