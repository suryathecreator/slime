# Nous AIME mixed-outcome evaluation

Every completion, including cap hits, is scored independently.

## AIME 2024-trained pair

| Slice | Base | Correct-trained | Incorrect-trained | Incorrect − correct |
|---|---:|---:|---:|---:|
| AIME 2024 held-in | 0.00% ± 0.00% | 71.88% ± 3.57% | 8.75% ± 2.24% | -63.12% ± 3.93% |
| AIME 2024 held-out | 12.19% ± 1.83% | 22.81% ± 2.35% | 24.06% ± 2.39% | 1.25% ± 2.03% |
| AIME 2024 all-30 | 8.12% ± 1.25% | 39.17% ± 2.23% | 18.96% ± 1.79% | -20.21% ± 2.34% |
| AIME 2025 cross-year | 4.38% ± 0.93% | 15.21% ± 1.64% | 16.88% ± 1.71% | 1.67% ± 1.35% |

## AIME 2025-trained pair

| Slice | Base | Correct-trained | Incorrect-trained | Incorrect − correct |
|---|---:|---:|---:|---:|
| AIME 2025 held-in | 1.25% ± 0.88% | 60.62% ± 3.87% | 2.50% ± 1.24% | -58.13% ± 4.01% |
| AIME 2025 held-out | 5.94% ± 1.32% | 23.75% ± 2.38% | 25.62% ± 2.44% | 1.88% ± 2.17% |
| AIME 2025 all-30 | 4.38% ± 0.93% | 36.04% ± 2.19% | 17.92% ± 1.75% | -18.12% ± 2.35% |
| AIME 2024 cross-year | 8.12% ± 1.25% | 17.08% ± 1.72% | 16.04% ± 1.68% | -1.04% ± 1.68% |

## All-60 diagnostics

Valid box means the contracted scorer parsed a complete boxed AIME integer.

| Target | Valid box | Cap hit | Mean tokens | Std | P50 | P90 | P95 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base_qwen2_5_7b | 68.44% | 5.62% | 2647.0 | 7336.8 | 706.0 | 2076.0 | 32537.0 | 32602.0 | 32629 |
| aime24_correct_3000 | 83.54% | 10.42% | 20302.9 | 8392.1 | 21145.5 | 32256.2 | 32552.5 | 32608.0 | 32624 |
| aime24_incorrect_3000 | 86.56% | 7.08% | 20607.6 | 6735.1 | 21121.0 | 29447.6 | 32527.0 | 32617.0 | 32647 |
| aime25_correct_3000 | 77.60% | 22.81% | 22115.2 | 8812.0 | 23170.5 | 32592.0 | 32610.0 | 32629.8 | 32645 |
| aime25_incorrect_3000 | 88.75% | 5.42% | 18062.1 | 6691.7 | 17918.5 | 26660.0 | 32479.0 | 32602.0 | 32631 |
