# Qwen3 8B -> 32B reproduction results

- qwen3_8b_base: 235/500 (47.00%)
- qwen3_32b_teacher: 476/500 (95.20%)
- sft_200000: 304/500 (60.80%)
- opd_100pct: 429/500 (85.80%)

## OPD learning dynamics

| Rollouts | Mean tokens | Cap hit | Think closed | Eligible final | `<|im_end|>` | Repetition | Reverse-KL |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-22 | 13309.4 | 54.14% | 32.95% | 84.10% | 30.50% | 16.44% | 0.6422 |
| 23-45 | 13605.5 | 56.45% | 42.87% | 86.41% | 37.50% | 19.50% | 0.2164 |
| 46-68 | 13285.5 | 47.69% | 47.69% | 84.99% | 51.90% | 10.05% | 0.2109 |
| 69-91 | 12493.2 | 40.15% | 51.43% | 86.41% | 59.24% | 9.17% | 0.2013 |

These are descriptive windows over different OpenThoughts prompts, not a controlled within-prompt estimate or a correctness measurement. From the first to final window, think closure changed by +18.48 points, natural stopping by +13.99 points, `<|im_end|>` termination by +28.74 points, and cap hits by -13.99 points. The 16K cutoff bounded compute; it was not an explicit terminal target.

## Abandoned 32K context attempt

Job `183039` selected all 58 OPD cap-hit rows. All 58 generation attempts reached engine completion, but exact-prefix replay failed and 0 rows were published. No 32K score exists.

A seeded stochastic rerun is not guaranteed to reproduce an earlier sampled prefix. Batch scheduling, kernel and floating-point paths, RNG consumption, engine version, and the requested generation budget can all change token sampling.

Continuing from the saved 16K prefix would condition on an old trajectory while restarting the sampler state. It is not the same experiment as drawing the response from the original prompt with a 32K budget, so it is not reported as a fidelity-preserving evaluation.

Regenerate all 500 prompts from their original inputs under one pinned 32K protocol. Context length alone would still not perfectly reproduce an external result unless the checkpoint, prompt template, tokenizer, stop rules, sampler implementation, seeds, scorer, engine/runtime, batching, and hardware behavior are also aligned.

A fresh 32K evaluation can be run later as an explicit ablation, alongside other protocol ablations, without changing the completed fixed-16K primary result.

Remaining jobs were cancelled: `183040` (sft_200000_32k_proxy), `183041` (qwen3_32b_teacher_32k_proxy), `183042` (qwen3_8b_base_32k_proxy).

Observed OPD gain: 25.00 points. Public targets were 76% SFT and 94% OPD; exact observed values are reported without seed or scorer selection.

Primary reproduction criteria use only the completed fixed-16K-output evaluations above. No incomplete 32K attempt is included in any metric.
