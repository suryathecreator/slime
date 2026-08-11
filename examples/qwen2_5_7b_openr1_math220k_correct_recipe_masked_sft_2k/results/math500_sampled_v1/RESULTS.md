# Correct-recipe masked Qwen2.5-7B MATH-500 (math500_last_boxed_symmetric_scorer_v6)

| Target | Source | Sampled mean ± SD | Repeat accuracies | Valid box | Cap hit | Correct cap hits | Parse failures | Mean tokens |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| base_qwen2_5_7b | reused | 55.67% ± 0.12% | 55.60%, 55.60%, 55.80% | 86.47% ± 1.10% | 1.53% ± 0.42% | 0/23 | 17/1500 | 1065.3 |
| qwen2_5_7b_8k | reused | 72.00% ± 1.40% | 70.40%, 72.60%, 73.00% | 79.40% ± 1.25% | 21.07% ± 1.33% | 3/316 | 29/1500 | 9120.4 |
| unmasked | generated | 71.87% ± 0.23% | 72.00%, 72.00%, 71.60% | 80.20% ± 1.40% | 20.13% ± 1.29% | 2/302 | 26/1500 | 8858.6 |
| random_mask_70 | generated | 71.73% ± 0.46% | 72.00%, 71.20%, 72.00% | 79.60% ± 0.20% | 20.73% ± 0.31% | 4/311 | 27/1500 | 9013.1 |
| inverse_tau_0p20 | generated | 70.80% ± 1.56% | 72.60%, 69.80%, 70.00% | 78.67% ± 1.22% | 21.73% ± 1.55% | 3/326 | 29/1500 | 9284.1 |
| inverse_tau_0p05 | generated | 70.73% ± 1.81% | 69.40%, 72.80%, 70.00% | 78.27% ± 2.73% | 21.87% ± 2.72% | 0/328 | 26/1500 | 9347.9 |
| random_mask_25 | generated | 73.33% ± 1.67% | 72.80%, 72.00%, 75.20% | 80.53% ± 0.92% | 19.60% ± 1.04% | 2/294 | 28/1500 | 8772.8 |
| random_mask_50 | generated | 70.93% ± 1.33% | 71.80%, 71.60%, 69.40% | 79.00% ± 0.69% | 21.13% ± 0.64% | 2/317 | 25/1500 | 9177.7 |
| random_mask_80 | generated | 69.60% ± 2.16% | 71.40%, 70.20%, 67.20% | 77.73% ± 2.00% | 22.53% ± 2.23% | 2/338 | 30/1500 | 9432.7 |
| random_mask_90 | generated | 70.27% ± 0.23% | 70.00%, 70.40%, 70.40% | 77.47% ± 0.58% | 22.80% ± 0.35% | 0/342 | 30/1500 | 9407.9 |
| margin_mask | generated | 68.80% ± 2.60% | 71.80%, 67.20%, 67.40% | 76.27% ± 2.47% | 24.27% ± 2.48% | 3/364 | 28/1500 | 10092.8 |
| prob_ratio_mask | generated | 58.27% ± 2.32% | 55.60%, 59.80%, 59.40% | 64.73% ± 2.34% | 36.60% ± 2.55% | 6/549 | 27/1500 | 13492.3 |

| Comparison | Sampled paired Δ mean ± SD |
|---|---:|
| unmasked minus base_qwen2_5_7b | +16.20% ± 0.35% |
| unmasked minus qwen2_5_7b_8k | -0.13% ± 1.55% |
| random_mask_70 minus base_qwen2_5_7b | +16.07% ± 0.42% |
| random_mask_70 minus qwen2_5_7b_8k | -0.27% ± 1.63% |
| inverse_tau_0p20 minus base_qwen2_5_7b | +15.13% ± 1.62% |
| inverse_tau_0p20 minus qwen2_5_7b_8k | -1.20% ± 2.95% |
| inverse_tau_0p05 minus base_qwen2_5_7b | +15.07% ± 1.86% |
| inverse_tau_0p05 minus qwen2_5_7b_8k | -1.27% ± 1.62% |
| random_mask_25 minus base_qwen2_5_7b | +17.67% ± 1.55% |
| random_mask_25 minus qwen2_5_7b_8k | +1.33% ± 1.68% |
| random_mask_50 minus base_qwen2_5_7b | +15.27% ± 1.45% |
| random_mask_50 minus qwen2_5_7b_8k | -1.07% ± 2.50% |
| random_mask_80 minus base_qwen2_5_7b | +13.93% ± 2.27% |
| random_mask_80 minus qwen2_5_7b_8k | -2.40% ± 3.40% |
| random_mask_90 minus base_qwen2_5_7b | +14.60% ± 0.20% |
| random_mask_90 minus qwen2_5_7b_8k | -1.73% ± 1.17% |
| margin_mask minus base_qwen2_5_7b | +13.13% ± 2.66% |
| margin_mask minus qwen2_5_7b_8k | -3.20% ± 3.98% |
| prob_ratio_mask minus base_qwen2_5_7b | +2.60% ± 2.27% |
| prob_ratio_mask minus qwen2_5_7b_8k | -13.73% ± 1.01% |
