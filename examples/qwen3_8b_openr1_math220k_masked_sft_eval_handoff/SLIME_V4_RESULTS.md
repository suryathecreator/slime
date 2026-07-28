# MATH-500 results with the shared SLIME V4 scorer

All rows below use `math500_strict_boxed_units_scorer_v4`. Existing 2K generations were rescored without inference; base and 40K generations use the same generation policy as the completed 2K evaluations.

| Family | Stage | Native | SLIME V4 | Base wrong→correct | Base correct→wrong |
|---|---|---:|---:|---:|---:|
| base | `base_8b` | 332/500 | 385/500 (77.00%) | 0 | 0 |
| 2k | `2k_correct_only` | 329/500 | 378/500 (75.60%) | 19 | 26 |
| 2k | `2k_unmasked` | 332/500 | 381/500 (76.20%) | 24 | 28 |
| 2k | `2k_random_mask_25` | 334/500 | 384/500 (76.80%) | 24 | 25 |
| 2k | `2k_random_mask_50` | 334/500 | 384/500 (76.80%) | 27 | 28 |
| 2k | `2k_random_mask_70` | 335/500 | 384/500 (76.80%) | 27 | 28 |
| 2k | `2k_random_mask_80` | 339/500 | 388/500 (77.60%) | 28 | 25 |
| 2k | `2k_random_mask_90` | 334/500 | 384/500 (76.80%) | 26 | 27 |
| 2k | `2k_inverse_tau_0p20` | 328/500 | 379/500 (75.80%) | 17 | 23 |
| 2k | `2k_inverse_tau_0p05` | 334/500 | 381/500 (76.20%) | 23 | 27 |
| 2k | `2k_margin_mask` | 332/500 | 379/500 (75.80%) | 26 | 32 |
| 2k | `2k_prob_ratio_mask` | 336/500 | 377/500 (75.40%) | 30 | 38 |
| 40k | `40k_correct_only` | 292/500 | 10/500 (2.00%) | 2 | 377 |
| 40k | `40k_weighted_tau_0p20` | 281/500 | 3/500 (0.60%) | 0 | 382 |
| 40k | `40k_unmasked` | 288/500 | 4/500 (0.80%) | 1 | 382 |
