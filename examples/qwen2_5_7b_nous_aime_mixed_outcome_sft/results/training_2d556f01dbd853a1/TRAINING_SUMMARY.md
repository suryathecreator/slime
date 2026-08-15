# Completed training provenance

- Contract: `2d556f01dbd853a1`
- Training commit: `418b0a1d3fce75898f4d5f01381da6f8433b18f2`
- Finalized: `2026-08-15T15:39:33.953613+00:00`
- Four independent full-parameter Qwen2.5-7B SFT runs completed at iteration 46.
- The complete 47-update metric history for every variant is in `handoff/training_metrics.json`.

| Variant | First loss | Last loss | Change | Minimum | Min step | First-5 avg | Last-5 avg | Final grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aime24_correct_3000 | 0.905331 | 0.496493 | -0.408838 | 0.496493 | 46 | 0.916130 | 0.537839 | 0.316234 |
| aime24_incorrect_3000 | 1.115909 | 0.711845 | -0.404064 | 0.709011 | 42 | 1.110199 | 0.719449 | 0.311129 |
| aime25_correct_3000 | 0.828161 | 0.495601 | -0.332560 | 0.470174 | 42 | 0.829095 | 0.483669 | 0.330267 |
| aime25_incorrect_3000 | 0.908070 | 0.531701 | -0.376368 | 0.521978 | 43 | 0.886017 | 0.527209 | 0.296468 |

This Git publication intentionally excludes checkpoints, optimizer/full state, pretokenized training data, raw Nous generations, and canary payloads. Checkpoints and the compact evaluation handoff are transferred separately with `rsync_to_klone.sh`.
