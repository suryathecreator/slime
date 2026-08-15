# Qwen2.5-7B AIME distribution generalization — interim (aime_last_boxed_integer_scorer_v1)

| Target | Held-in accuracy | Held-out accuracy | Held-out − held-in | Δ held-in vs base | Δ held-out vs base | Δ held-in vs 3K | Δ held-out vs 3K |
|---|---:|---:|---:|---:|---:|---:|---:|
| base_qwen2_5_7b | 22.17% ± 0.63% | 21.42% ± 2.50% | -0.75% ± 2.29% | +0.00% ± 0.00% | +0.00% ± 0.00% | -16.50% ± 0.25% | -9.08% ± 2.90% |
| aime_correct_3000 | 38.67% ± 0.38% | 30.50% ± 1.09% | -8.17% ± 1.46% | +16.50% ± 0.25% | +9.08% ± 2.90% | +0.00% ± 0.00% | +0.00% ± 0.00% |
| continue_wrong_unmasked | 22.92% ± 2.98% | 23.00% ± 0.90% | +0.08% ± 3.79% | +0.75% ± 3.61% | +1.58% ± 1.94% | -15.75% ± 3.36% | -7.50% ± 1.89% |
| continue_correct_only_1500 | 24.50% ± 1.56% | 26.17% ± 1.38% | +1.67% ± 1.18% | +2.33% ± 1.26% | +4.75% ± 2.17% | -14.17% ± 1.38% | -4.33% ± 2.43% |
| continue_random_mask_50 | 23.58% ± 2.57% | 23.08% ± 0.80% | -0.50% ± 2.63% | +1.42% ± 2.53% | +1.67% ± 2.79% | -15.08% ± 2.55% | -7.42% ± 0.29% |

| Target | Split | Valid box | Cap hit | Correct cap hits | Parse failures | Mean generated tokens |
|---|---|---:|---:|---:|---:|---:|
| base_qwen2_5_7b | held_in | 86.42% ± 1.84% | 4.58% ± 1.13% | 0/55 | 139/1200 | 2328.7 |
| base_qwen2_5_7b | held_out_problem | 85.75% ± 1.25% | 5.25% ± 0.66% | 0/63 | 130/1200 | 2579.1 |
| aime_correct_3000 | held_in | 69.67% ± 1.84% | 31.50% ± 1.98% | 1/378 | 70/1200 | 19984.8 |
| aime_correct_3000 | held_out_problem | 68.58% ± 0.88% | 31.67% ± 1.28% | 0/380 | 78/1200 | 20588.0 |
| continue_wrong_unmasked | held_in | 35.33% ± 4.26% | 65.08% ± 3.97% | 3/781 | 20/1200 | 24100.6 |
| continue_wrong_unmasked | held_out_problem | 34.08% ± 1.66% | 66.33% ± 1.63% | 3/796 | 17/1200 | 24342.3 |
| continue_correct_only_1500 | held_in | 39.50% ± 2.95% | 60.75% ± 2.70% | 1/729 | 27/1200 | 23100.1 |
| continue_correct_only_1500 | held_out_problem | 39.92% ± 1.81% | 60.00% ± 1.95% | 0/720 | 27/1200 | 22705.4 |
| continue_random_mask_50 | held_in | 37.00% ± 1.75% | 63.50% ± 1.80% | 2/762 | 19/1200 | 23582.8 |
| continue_random_mask_50 | held_out_problem | 36.25% ± 2.00% | 64.17% ± 1.77% | 1/770 | 19/1200 | 23687.7 |
