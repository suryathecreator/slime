# Corrected SFT -> OPD Colocate4 1k/32k Results

Recorded from jobs `160424`-`160427`, completed on 2026-07-05.

## Run Summary

| field | value |
| --- | --- |
| OPD run label | `1k_32k_sft_colocate4` |
| OPD train job | `160424`, completed, `05:40:59` |
| OPD eval job | `160425`, completed, `04:33:04` |
| maybe-base eval job | `160426`, completed, `00:00:05` |
| report job | `160427`, completed, `00:00:03` |
| OPD initial actor weights | final SFT HF snapshot `iter_0000096` |
| OPD reference for logged KL | final SFT HF snapshot `iter_0000096` |
| OPD effective samples | `1024`, rollout ids `0..7` |
| actor / rollout / teacher GPUs | `3 / 3 / 1`, actor+rollout colocated |
| actor TP / CP | `1 / 3` |
| response cap | `31744` |
| max tokens per GPU | `2048` |
| logprob chunk size | `512` |
| train memory margin bytes | `0` |
| final full OPD checkpoint | `qwen3_8b_sft_25k_opd_1k_32k_sft_colocate4_full_optim/iter_0000007` |
| final OPD HF snapshot | `qwen3_8b_sft_25k_opd_1k_32k_sft_colocate4_eval_snapshots/iter_0000007` |

## MATH-500

Parse failures are counted wrong in `accuracy`. `accuracy_on_parseable` is
diagnostic only.

| phase | stage | effective train samples | accuracy | accuracy on parseable | parse failure rate | cap hit rate | avg generated tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| base | `base` | 0 | 0.638 | 0.7595238095 | 0.160 | 0.036 | 1686.402 |
| SFT | `sft_024832` | 24832 | 0.750 | 0.78125 | 0.040 | 0.576 | 18574.302 |
| SFT + OPD | `opd_001024` | 25856 total (`SFT+1024`) | 0.724 | 0.7685774947 | 0.058 | 0.832 | 26567.282 |

## OPD Rollout Diagnostics

Averaging: logprob/KL/loss diagnostics are response-token-weighted rollout
means. Response lengths are sample statistics over each 128-sample rollout.
Cap/completion/final-answer rates are sample fractions.

| rollout | avg response tokens | cap hit rate | completed rate | final answer rate | rollout logprobs | actor logprobs | ref logprobs | teacher logprobs | OPD reverse KL | abs diff actor vs rollout | logged SFT-ref KL loss | grad norm | sanity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 0 | 15409.0859 | 0.171875 | 0.828125 | 0.984375 | -1.362256 | -1.363400 | -1.363400 | -1.864063 | 0.500663 | 0.021403 | 0.000000 | 14.243451 | passed |
| 1 | 14848.8125 | 0.179688 | 0.820312 | 0.984375 | -1.561668 | -1.562774 | -1.562792 | -2.069808 | 0.507034 | 0.023460 | 0.001163 | 14.951931 | passed |
| 2 | 14218.9375 | 0.171875 | 0.828125 | 1.000000 | -1.434321 | -1.435466 | -1.435382 | -1.924017 | 0.488550 | 0.021888 | 0.001173 | 13.952065 | passed |
| 3 | 15013.6719 | 0.117188 | 0.882812 | 0.992188 | -1.333715 | -1.335050 | -1.334924 | -1.844314 | 0.509264 | 0.022715 | 0.001363 | 14.530953 | passed |
| 4 | 15976.6797 | 0.164062 | 0.835938 | 0.976562 | -1.361095 | -1.362380 | -1.362747 | -1.858828 | 0.496449 | 0.022434 | 0.001650 | 13.351168 | passed |
| 5 | 18577.8125 | 0.257812 | 0.742188 | 0.992188 | -1.295447 | -1.296449 | -1.297052 | -1.749019 | 0.452570 | 0.021632 | 0.001809 | 13.133417 | violated avg length, nonfatal |
| 6 | 15737.1797 | 0.171875 | 0.828125 | 0.945312 | -1.488746 | -1.489854 | -1.490554 | -1.986171 | 0.496317 | 0.022789 | 0.001971 | 12.746342 | passed |
| 7 | 16984.9531 | 0.171875 | 0.828125 | 1.000000 | -1.360778 | -1.361966 | -1.362616 | -1.791660 | 0.429695 | 0.021573 | 0.002039 | 12.128090 | violated avg length, nonfatal |

Aggregate rollout diagnostics:

| metric | mean | min | max |
| --- | ---: | ---: | ---: |
| avg response tokens | 15845.8916 | 14218.9375 | 18577.8125 |
| cap hit rate | 0.175781 | 0.117188 | 0.257812 |
| completed rate | 0.824219 | 0.742188 | 0.882812 |
| final answer rate | 0.984375 | 0.945312 | 1.000000 |
| OPD reverse KL | 0.485068 | 0.429695 | 0.509264 |
| actor-vs-rollout abs diff | 0.022237 | 0.021403 | 0.023460 |
| logged SFT-reference KL loss | 0.001396 | 0.000000 | 0.002039 |
| grad norm | 13.629677 | 12.128090 | 14.951931 |

Rollouts `5` and `7` exceeded the diagnostic average-response threshold
(`16000`) but trained because `OPD_SANITY_FAIL_ON_COLLAPSE=0`.

## Tracked Artifacts

- `summary_all.json`: combined base -> SFT -> OPD MATH-500 points.
- `combined_accuracy_curve.csv`: curve data used for the combined figure.
- `combined_accuracy_curve.svg`: generated combined curve.
- `opd_001024_summary.json`: final OPD MATH-500 summary.
- `opd_sanity_summary.{json,csv,md}`: rollout reward/logprob/sanity table.
