# Qwen correct-only MATH-500: provisional greedy results

These are the currently available greedy results for submission attempt 3. All
nine targets have four complete 125-example generation shards. Before scoring,
the control code validated exact 500-example coverage, checkpoint and policy
fingerprints, response hashes, and the 32,768-token prompt-plus-response budget.

The table is provisional only at the workflow level: the experiment-wide audit
and most compact per-target result bundles still await the three sampled repeats.
Cap hits are included in the ordinary 500-example accuracy denominator and are
scored from the saved response exactly like natural stops.

| Target | Correct | Accuracy | Delta vs base | Valid boxed answer | Cap hits | Correct cap hits | Parse failures |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-3B base | 271/500 | 54.20% | - | 432 (86.40%) | 22 (4.40%) | 0/22 (0.00%) | 6 |
| Qwen2.5-3B 8K trained | 188/500 | 37.60% | -16.60 pp | 213 (42.60%) | 288 (57.60%) | 1/288 (0.35%) | 2 |
| Qwen2.5-3B 16K trained | 162/500 | 32.40% | -21.80 pp | 182 (36.40%) | 327 (65.40%) | 6/327 (1.83%) | 1 |
| Qwen2.5-7B base | 290/500 | 58.00% | - | 434 (86.80%) | 11 (2.20%) | 0/11 (0.00%) | 6 |
| Qwen2.5-7B 8K trained | 324/500 | 64.80% | +6.80 pp | 348 (69.60%) | 154 (30.80%) | 1/154 (0.65%) | 9 |
| Qwen3-4B base | 361/500 | 72.20% | - | 478 (95.60%) | 14 (2.80%) | 0/14 (0.00%) | 4 |
| Qwen3-4B 8K trained | 336/500 | 67.20% | -5.00 pp | 356 (71.20%) | 469 (93.80%) | 306/469 (65.25%) | 7 |
| Qwen3-8B base | 386/500 | 77.20% | - | 480 (96.00%) | 13 (2.60%) | 0/13 (0.00%) | 6 |
| Qwen3-8B 8K trained | 361/500 | 72.20% | -5.00 pp | 373 (74.60%) | 470 (94.00%) | 332/470 (70.64%) | 7 |

Scorer: `math500_last_boxed_symmetric_scorer_v6` (`c76945273ca08e2a33e9b3ca7437e3da4aa3dde71390e5559062e373caea6df5`).

Two official compact target bundles are also present at this snapshot:

| Target | Greedy | Sampled pass@1 mean +/- sample SD | Sample repeats |
|---|---:|---:|---:|
| Qwen2.5-3B base | 54.20% | 53.07% +/- 0.64 pp | 53.80%, 52.60%, 52.80% |
| Qwen2.5-7B base | 58.00% | 55.67% +/- 0.12 pp | 55.60%, 55.60%, 55.80% |
