# Qwen2.5-7B AIME split-control evaluation (aime_split_control_sampled_v1)

Four sampled generations per problem at temperature 0.6, top-p 0.95, top-k 20, min-p 0.0, reused verbatim from the Qwen3-8B 16x source-generation run.

These numbers sit on the same decoding distribution as the traces the checkpoints were
trained on. Earlier Qwen2.5 AIME bundles in this repository used temperature 0.7 /
top-p 0.8 / top-k -1 and are **not** directly comparable to this table.

Every completion is scored, including cap hits. Accuracy is the mean over all
848 completions per target; ± is the standard error across the 4 draws.

## Trained on held_in

| Slice | Base | Correct-trained | Incorrect-trained | Correct − base | Incorrect − base | Incorrect − correct |
|---|---:|---:|---:|---:|---:|---:|
| held_in (trained distribution) | 7.08% ± 1.47% | 21.70% ± 1.76% | 11.79% ± 1.36% | +14.62% ± 2.23% | +4.72% ± 1.90% | -9.91% ± 2.21% |
| held_out (unseen problems) | 1.89% ± 0.67% | 4.72% ± 0.86% | 5.66% ± 0.77% | +2.83% ± 1.24% | +3.77% ± 1.32% | +0.94% ± 1.38% |
| all 212 | 4.48% ± 0.45% | 13.21% ± 1.25% | 8.73% ± 0.99% | +8.73% ± 1.29% | +4.25% ± 1.16% | -4.48% ± 1.32% |

## Trained on held_out

| Slice | Base | Correct-trained | Incorrect-trained | Correct − base | Incorrect − base | Incorrect − correct |
|---|---:|---:|---:|---:|---:|---:|
| held_out (trained distribution) | 1.89% ± 0.67% | 15.57% ± 0.98% | 7.55% ± 0.54% | +13.68% ± 1.89% | +5.66% ± 1.43% | -8.02% ± 1.97% |
| held_in (unseen problems) | 7.08% ± 1.47% | 8.49% ± 1.28% | 8.49% ± 0.39% | +1.42% ± 1.70% | +1.42% ± 1.70% | +0.00% ± 1.42% |
| all 212 | 4.48% ± 0.45% | 12.03% ± 0.73% | 8.02% ± 0.43% | +7.55% ± 1.29% | +3.54% ± 1.11% | -4.01% ± 1.22% |

## All-212 diagnostics

Valid box means the contracted scorer parsed a complete boxed AIME integer.

| Target | Accuracy | Valid box | Cap hit | Mean tokens | Std | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_qwen2_5_7b | 4.48% ± 0.45% | 65.92% | 7.19% | 3412.7 | 8213.7 | 889.5 | 3549.6 | 32567.2 | 32671.6 | 32711 |
| held_in_correct_6000 | 13.21% ± 1.25% | 37.50% | 60.50% | 27947.8 | 7389.3 | 32570.0 | 32675.0 | 32684.0 | 32698.8 | 32711 |
| held_in_incorrect_6000 | 8.73% ± 0.99% | 36.20% | 63.44% | 28744.0 | 6728.0 | 32585.0 | 32675.0 | 32686.0 | 32703.0 | 32711 |
| held_out_correct_6000 | 12.03% ± 0.73% | 37.38% | 61.67% | 28421.5 | 6873.4 | 32571.0 | 32673.0 | 32684.0 | 32706.2 | 32711 |
| held_out_incorrect_6000 | 8.02% ± 0.43% | 37.26% | 63.56% | 28989.6 | 6281.6 | 32581.0 | 32675.0 | 32686.0 | 32709.0 | 32711 |
