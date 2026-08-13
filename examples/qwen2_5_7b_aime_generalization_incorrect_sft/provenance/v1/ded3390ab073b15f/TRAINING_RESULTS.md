# AIME distribution-generalization training results

Contract: `ded3390ab073b15f`

Training commit: `59e8e146d4ca22fc0f841acb33fb7e251bb4bf0d`

Finalized: `2026-08-13T04:49:34.513478+00:00`

All 12 full-parameter SFT checkpoints completed and passed the final checkpoint-manifest gate.

| Variant | Job | Updates | First loss | Last loss | Change |
|---|---:|---:|---:|---:|---:|
| `aime_correct_3000` | 223964 | 188 | 0.72369651 | 0.04420446 | -0.67949204 |
| `continue_wrong_unmasked` | 223966 | 94 | 0.71364281 | 0.42633386 | -0.28730895 |
| `continue_random_mask_10` | 223967 | 94 | 0.71368779 | 0.42253325 | -0.29115454 |
| `continue_random_mask_20` | 223968 | 94 | 0.71265042 | 0.41710960 | -0.29554082 |
| `continue_random_mask_30` | 223969 | 94 | 0.71289116 | 0.41215054 | -0.30074062 |
| `continue_random_mask_40` | 223970 | 94 | 0.71269652 | 0.40386149 | -0.30883503 |
| `continue_random_mask_50` | 223971 | 94 | 0.71373756 | 0.39406823 | -0.31966934 |
| `continue_random_mask_60` | 223972 | 94 | 0.71331561 | 0.38251155 | -0.33080407 |
| `continue_random_mask_70` | 223973 | 94 | 0.71058963 | 0.36483628 | -0.34575336 |
| `continue_random_mask_80` | 223974 | 94 | 0.70948890 | 0.32223868 | -0.38725021 |
| `continue_random_mask_90` | 223975 | 94 | 0.71013675 | 0.23757009 | -0.47256667 |
| `continue_correct_only_1500` | 223976 | 94 | 0.60612446 | 0.39729307 | -0.20883139 |

Losses are the realized training objectives. Absolute values across masking rates are not strictly comparable because each rate retains a different token set.

`training_metrics.json` contains every recorded step's loss, gradient norm, learning rate, global batch size, and the SHA-256 of its source Slurm log.
