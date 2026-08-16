# Nous AIME mixed-outcome evaluation

Every completion, including cap hits, is scored independently.

## AIME 2024-trained pair

| Slice | Base | Correct-trained | Incorrect-trained | Incorrect − correct |
|---|---:|---:|---:|---:|
| AIME 2024 held-in | 0.00% ± 0.00% | 7.50% ± 2.09% | 0.00% ± 0.00% | -7.50% ± 2.09% |
| AIME 2024 held-out | 12.19% ± 1.83% | 12.50% ± 1.85% | 8.12% ± 1.53% | -4.38% ± 1.91% |
| AIME 2024 all-30 | 8.12% ± 1.25% | 10.83% ± 1.42% | 5.42% ± 1.03% | -5.42% ± 1.45% |
| AIME 2025 cross-year | 4.38% ± 0.93% | 7.71% ± 1.22% | 3.96% ± 0.89% | -3.75% ± 1.17% |

## AIME 2025-trained pair

| Slice | Base | Correct-trained | Incorrect-trained | Incorrect − correct |
|---|---:|---:|---:|---:|
| AIME 2025 held-in | 1.25% ± 0.88% | 15.00% ± 2.83% | 0.62% ± 0.62% | -14.37% ± 2.78% |
| AIME 2025 held-out | 5.94% ± 1.32% | 15.00% ± 2.00% | 11.25% ± 1.77% | -3.75% ± 2.07% |
| AIME 2025 all-30 | 4.38% ± 0.93% | 15.00% ± 1.63% | 7.71% ± 1.22% | -7.29% ± 1.67% |
| AIME 2024 cross-year | 8.12% ± 1.25% | 8.96% ± 1.30% | 12.50% ± 1.51% | 3.54% ± 1.51% |

## All-60 diagnostics

Valid box means the contracted scorer parsed a complete boxed AIME integer.

| Target | Valid box | Cap hit | Mean tokens | Std | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_qwen2_5_7b | 68.44% | 5.62% | 2647.0 | 7336.8 | 706.0 | 2076.0 | 32537.0 | 32602.0 | 32629 |
| aime24_correct_3000 | 15.31% | 99.90% | 32522.7 | 701.8 | 32582.0 | 32623.1 | 32629.1 | 32647.0 | 32647 |
| aime24_incorrect_3000 | 6.35% | 99.58% | 32445.3 | 1581.1 | 32582.0 | 32623.0 | 32629.0 | 32647.0 | 32647 |
| aime25_correct_3000 | 23.33% | 99.48% | 32433.5 | 1578.7 | 32582.0 | 32623.0 | 32629.0 | 32647.0 | 32647 |
| aime25_incorrect_3000 | 21.67% | 98.65% | 32252.6 | 2548.2 | 32582.0 | 32623.0 | 32629.0 | 32647.0 | 32647 |

## Generalization at a glance

The base row immediately below each trained condition uses the same evaluation
slices as that condition.

### Correct-trace training

| Condition | Same-year held-out | Cross-year overall | All 60 |
|---|---:|---:|---:|
| AIME 2024 correct | 12.50% | 7.71% on AIME 2025 | 9.27% |
| Base control for AIME 2024 | 12.19% | 4.38% on AIME 2025 | 6.25% |
| AIME 2025 correct | 15.00% | 8.96% on AIME 2024 | 11.98% |
| Base control for AIME 2025 | 5.94% | 8.12% on AIME 2024 | 6.25% |

Correct traces generalized more consistently than incorrect traces. The AIME
2024 condition was essentially tied with base on same-year held-out and gained
3.33 points cross-year. The AIME 2025 condition gained 9.06 points on
same-year held-out and was essentially tied cross-year. Both improved the
all-60 average over base.

### Incorrect-trace training

| Condition | Same-year held-out | Cross-year overall | All 60 |
|---|---:|---:|---:|
| AIME 2024 incorrect | 8.12% | 3.96% on AIME 2025 | 4.69% |
| Base control for AIME 2024 | 12.19% | 4.38% on AIME 2025 | 6.25% |
| AIME 2025 incorrect | 11.25% | 12.50% on AIME 2024 | 10.10% |
| Base control for AIME 2025 | 5.94% | 8.12% on AIME 2024 | 6.25% |

Incorrect-trace generalization depended strongly on training year. The AIME
2024 condition regressed against base on both generalization checks and on all
60 problems. The AIME 2025 condition exceeded base by 5.31 points on
same-year held-out, 4.38 points cross-year, and 3.85 points overall.

These trained-model comparisons require caution: cap-hit rates were
98.65--99.90% and valid-box rates were only 6.35--23.33% over all 60 prompts.
All cap hits were nevertheless scored under the contracted whole-response
last-boxed-integer rule; none were discarded.
