# Corrected Colocate4 OPD Diagnostics

This note records the diagnostics that motivated resubmitting the corrected
SFT -> OPD colocate4 job with non-fatal sanity checks. The run still logs
sanity violations and writes summary tables, but violations no longer stop
training.

## Averaging Rules

For OPD rollout and train diagnostics, logprob/KL/loss-like values are
response-token-weighted means over the rollout. That means long responses
contribute proportionally more tokens. Response length statistics are sample
statistics over the rollout batch. Cap-hit, completion, and final-answer rates
are fractions of samples in the rollout batch.

## Job 160279 Length Trigger

Job `160279` was the corrected SFT-loaded colocate4 retry with actor/rollout/
teacher `3/3/1`, actor `TP=1`, `CP=3`, `OPD_MAX_RESPONSE_LEN=31744`,
`OPD_MAX_TOKENS_PER_GPU=2048`, and `OPD_LOG_PROBS_CHUNK_SIZE=512`. It completed
actor updates for rollouts `0..3`. Rollout `4` generated successfully but the
old sanity hardstop fired before reward postprocessing/training because average
response length exceeded the diagnostic threshold.

| rollout | sample range | avg response tokens | median response tokens | min response tokens | max response tokens | cap hit rate | completed rate | final answer rate | rollout log probs | actor log probs | ref log probs | teacher log probs | OPD reverse KL | train abs diff | diagnostic KL loss | note |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 0-127 | 15150.1328 |  |  | 31744 | 0.171875 | 0.828125 | 0.9765625 | -1.479008 | -1.480069 | -1.645964 | -1.984223 | 0.504153 | 0.021644 | 0.156793 | completed actor train |
| 1 | 128-255 | 15728.2812 |  |  | 31744 | 0.15625 | 0.84375 | 0.9765625 | -1.616234 | -1.617506 | -1.791842 | -2.124253 | 0.506747 | 0.023211 | 0.165473 | completed actor train |
| 2 | 256-383 | 14458.9922 |  |  | 31744 | 0.1484375 | 0.8515625 | 0.984375 | -1.426510 | -1.427689 | -1.603311 | -1.942297 | 0.514609 | 0.022638 | 0.167436 | completed actor train |
| 3 | 384-511 | 14563.1641 |  |  | 31744 | 0.1484375 | 0.8515625 | 0.9765625 | -1.369672 | -1.370745 | -1.541017 | -1.900498 | 0.529753 | 0.021658 | 0.160242 | completed actor train |
| 4 | 512-639 | 16131.8125 | 16579 | 2 | 31744 | 0.1640625 | 0.8359375 | 0.984375 |  |  |  |  |  |  |  | hardstopped before reward/train metrics |

Column definitions:

| column | definition |
| --- | --- |
| rollout | OPD rollout/update id. |
| sample range | OPD data row offsets in the rollout batch. |
| avg response tokens | Mean generated response length over samples. |
| median/min/max response tokens | Median, minimum, and maximum generated response lengths over samples, when available. |
| cap hit rate | Fraction of samples that hit the response cap or were marked truncated. |
| completed rate | Fraction of samples marked completed. |
| final answer rate | Fraction of samples whose response tail contained a conservative final-answer marker. |
| rollout log probs | SGLang rollout-student logprobs, response-token-weighted over the rollout. |
| actor log probs | Megatron actor logprobs recomputed before update, response-token-weighted over the rollout. |
| ref log probs | Megatron reference logprobs, response-token-weighted over the rollout. |
| teacher log probs | Qwen3-32B teacher logprobs, response-token-weighted over the rollout. |
| OPD reverse KL | Mean `actor_log_probs - teacher_log_probs` over response tokens. |
| train abs diff | Mean absolute difference between Megatron actor logprobs and SGLang rollout logprobs over response tokens. |
| diagnostic KL loss | Logged reference-KL loss. Its coefficient is `0.00`, so it is diagnostic only. |
| note | Whether actor training completed or why metrics are missing. |

## Dataset Facts From Length Investigation

These counts were calculated while checking why generated traces are long and
whether SFT trained on reasoning traces.

| field | value | meaning |
| --- | ---: | --- |
| source rows | 1,200,000 | Rows in `open-thoughts/OpenThoughts3-1.2M` source split scanned by the prep job. |
| math rows after filter | 850,000 | Rows matching `domain == math`. |
| SFT rows | 25,000 | Seeded SFT split rows. |
| OPD pool rows | 10,000 | Row-disjoint OPD prompt pool rows. |
| seed | 1234 | Seed used for split membership. |
| SFT rows with `<think>` | 25,000 / 25,000 | Raw assistant messages containing an opening think marker. |
| SFT rows with complete `</think>` | 8,102 / 25,000 | Raw assistant messages with a complete close marker. |
| SFT rendered token count | 25,000 | Number of SFT rows in the token-length stats. |
| SFT rendered token avg | 15,694.5678 | Mean rendered SFT chat-token length per row. |
| SFT rendered token max | 16,817 | Maximum rendered SFT chat-token length. |
| OPD prompt token count | 10,000 | Number of OPD prompt rows in the token-length stats. |
| OPD prompt token avg | 89.8991 | Mean OPD prompt-token length per row. |
| OPD prompt token max | 2,170 | Maximum OPD prompt-token length. |

Dataset table averaging: row counts are counts of examples; token averages are
simple means over rows in the relevant split.

## Accidental Base -> OPD Is Not Fidelity-Preserving

The older `1k_32k` run is diagnostic only. It passed an HF snapshot to
Megatron `--load` without the explicit HF-load path, causing the trainable
Megatron actor path to fall back/misload. SGLang rollout and Megatron reference
agreed, but Megatron actor did not.

| field | value | meaning |
| --- | ---: | --- |
| rollout log probs | -0.856 | SGLang rollout-student logprobs, response-token-weighted over the rollout. |
| ref log probs | -0.857 | Megatron reference logprobs, response-token-weighted over the rollout. |
| actor log probs | -12.723 | Megatron actor logprobs, response-token-weighted over the rollout. |
| abs diff | about 11.9 | Mean `abs(actor logprobs - rollout logprobs)` over response tokens. |

The corrected colocate4 path does not show this failure. In job `160279`, the
Megatron-vs-SGLang abs diff was about `0.02` for completed rollouts `0..3`.

## OPD Pipeline Debugging Notes

This section is optional debugging context.

1. The corrected actor initializes from final SFT HF snapshot `iter_0000096`
   with a fresh OPD optimizer. The TP=2 SFT optimizer checkpoint is not loaded
   because colocate4 uses actor `TP=1`, `CP=3`.
2. The Megatron reference model now also loads final SFT HF snapshot
   `iter_0000096`, so logged `train/kl_loss` measures drift from final SFT.
   `kl_loss_coef=0.00`, so this diagnostic does not affect gradients.
3. SGLang rollout engines generate student responses and record
   `rollout_log_probs`.
4. The Qwen3-32B SGLang teacher server computes `teacher_log_probs` for the
   generated sequences.
5. Megatron recomputes pre-update actor `log_probs` on the same response
   tokens, computes OPD advantages from `actor_log_probs - teacher_log_probs`,
   and backpropagates through the actor.
6. The active OPD learning signal is `opd_reverse_kl` with `opd_kl_coef=1.0`;
   the reference KL is logged only.
