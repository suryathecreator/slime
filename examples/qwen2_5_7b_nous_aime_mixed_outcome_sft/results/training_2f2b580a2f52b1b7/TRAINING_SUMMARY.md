# Completed training provenance

- Contract: `2f2b580a2f52b1b7`
- Training commit: `9121068e1e3fe28e9cfe4bc7ec5bd1f5b5380f08`
- Finalized: `2026-08-16T20:34:07.767375+00:00`
- Four independent full-parameter Qwen2.5-7B SFT runs completed at iteration 187.
- The complete 188-update metric history for every variant is in `handoff/training_metrics.json`.

| Variant | First loss | Last loss | Change | Minimum | Min step | First-5 avg | Last-5 avg | Final grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| aime24_correct_3000 | 0.905331 | 0.001900 | -0.903431 | 0.001900 | 187 | 0.927281 | 0.002434 | 0.086628 |
| aime24_incorrect_3000 | 1.115909 | 0.031962 | -1.083947 | 0.027622 | 186 | 1.122269 | 0.031918 | 0.340532 |
| aime25_correct_3000 | 0.828161 | 0.001568 | -0.826593 | 0.001504 | 177 | 0.840521 | 0.002288 | 0.142774 |
| aime25_incorrect_3000 | 0.908070 | 0.005501 | -0.902569 | 0.005501 | 187 | 0.897420 | 0.008095 | 0.156173 |

This Git publication intentionally excludes checkpoints, optimizer/full state, pretokenized training data, raw Nous generations, and canary payloads. Checkpoints and the compact evaluation handoff are transferred separately with `rsync_to_klone.sh`.
