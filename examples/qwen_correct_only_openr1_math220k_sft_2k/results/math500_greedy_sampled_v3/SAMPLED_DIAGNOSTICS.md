# Sampled MATH-500 diagnostics

Each sampled value is the mean and sample standard deviation across three
complete 500-example pass@1 repeats. Each repeat used temperature 0.7 and
top-p 0.8. Cap hits remain in the ordinary 500-example accuracy denominator
and are scored from the saved response exactly like natural stops.

| Target | Greedy | Sampled accuracy | Sampled - greedy | Valid box rate | Cap-hit rate | Accuracy on cap hits | Mean generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B base | 54.20% | 53.07% +/- 0.64 pp | -1.13 pp | 87.60% +/- 1.60 pp | 3.20% +/- 0.20 pp | 0.00% +/- 0.00 pp | 1,607 +/- 82 |
| Qwen2.5-3B 8K | 37.60% | 47.40% +/- 2.11 pp | +9.80 pp | 59.13% +/- 1.21 pp | 78.20% +/- 1.22 pp | 38.17% +/- 2.49 pp | 26,499 +/- 302 |
| Qwen2.5-3B 16K | 32.40% | 43.87% +/- 1.40 pp | +11.47 pp | 54.27% +/- 1.29 pp | 80.47% +/- 2.84 pp | 35.11% +/- 1.76 pp | 27,122 +/- 888 |
| Qwen2.5-7B base | 58.00% | 55.67% +/- 0.12 pp | -2.33 pp | 86.47% +/- 1.10 pp | 1.53% +/- 0.42 pp | 0.00% +/- 0.00 pp | 1,065 +/- 125 |
| Qwen2.5-7B 8K | 64.80% | 72.00% +/- 1.40 pp | +7.20 pp | 79.40% +/- 1.25 pp | 21.07% +/- 1.33 pp | 0.92% +/- 0.89 pp | 9,120 +/- 292 |
| Qwen3-4B base | 72.20% | 69.20% +/- 1.51 pp | -3.00 pp | 95.00% +/- 1.11 pp | 2.40% +/- 0.35 pp | 2.56% +/- 4.44 pp | 1,372 +/- 87 |
| Qwen3-4B 8K | 67.20% | 71.47% +/- 0.83 pp | +4.27 pp | 77.00% +/- 0.53 pp | 48.33% +/- 0.70 pp | 56.26% +/- 1.47 pp | 19,471 +/- 18 |
| Qwen3-8B base | 77.20% | 69.40% +/- 0.87 pp | -7.80 pp | 92.27% +/- 0.83 pp | 2.73% +/- 0.12 pp | 9.52% +/- 10.91 pp | 1,471 +/- 61 |
| Qwen3-8B 8K | 72.20% | 78.33% +/- 0.61 pp | +6.13 pp | 82.67% +/- 1.29 pp | 89.40% +/- 0.72 pp | 80.16% +/- 1.57 pp | 29,725 +/- 143 |

## Sampled-minus-greedy diagnostic deltas

Every entry below is the sampled mean minus the single fixed greedy pass. The
`+/-` value remains the sample standard deviation across sampled repeats.

| Target | Accuracy delta | Valid-box-rate delta | Cap-hit-rate delta | Cap-hit-accuracy delta | Generated-token delta | Parse-failure delta |
|---|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B base | -1.13 +/- 0.64 pp | +1.20 +/- 1.60 pp | -1.20 +/- 0.20 pp | 0.00 +/- 0.00 pp | -359 +/- 82 | +3.00 +/- 1.00 |
| Qwen2.5-3B 8K | +9.80 +/- 2.11 pp | +16.53 +/- 1.21 pp | +20.60 +/- 1.22 pp | +37.82 +/- 2.49 pp | +6,956 +/- 302 | +5.67 +/- 2.08 |
| Qwen2.5-3B 16K | +11.47 +/- 1.40 pp | +17.87 +/- 1.29 pp | +15.07 +/- 2.84 pp | +33.27 +/- 1.76 pp | +5,169 +/- 888 | +5.00 +/- 1.00 |
| Qwen2.5-7B base | -2.33 +/- 0.12 pp | -0.33 +/- 1.10 pp | -0.67 +/- 0.42 pp | 0.00 +/- 0.00 pp | -162 +/- 125 | -0.33 +/- 0.58 |
| Qwen2.5-7B 8K | +7.20 +/- 1.40 pp | +9.80 +/- 1.25 pp | -9.73 +/- 1.33 pp | +0.27 +/- 0.89 pp | -2,289 +/- 292 | +0.67 +/- 1.15 |
| Qwen3-4B base | -3.00 +/- 1.51 pp | -0.60 +/- 1.11 pp | -0.40 +/- 0.35 pp | +2.56 +/- 4.44 pp | -95 +/- 87 | +2.67 +/- 2.52 |
| Qwen3-4B 8K | +4.27 +/- 0.83 pp | +5.80 +/- 0.53 pp | -45.47 +/- 0.70 pp | -8.98 +/- 1.47 pp | -12,680 +/- 18 | +0.67 +/- 2.31 |
| Qwen3-8B base | -7.80 +/- 0.87 pp | -3.73 +/- 0.83 pp | +0.13 +/- 0.12 pp | +9.52 +/- 10.91 pp | +72 +/- 61 | +3.67 +/- 1.53 |
| Qwen3-8B 8K | +6.13 +/- 0.61 pp | +8.07 +/- 1.29 pp | -4.60 +/- 0.72 pp | +9.52 +/- 1.57 pp | -1,229 +/- 143 | +1.67 +/- 2.08 |

The exact three-repeat counts and complete scorer summaries remain in each
target's `RESULTS.json`.
