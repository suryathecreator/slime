# Completed training provenance

- Contract: `0b22cc069c941ec5`
- Training commit: `70cd3222ca37e6a545b990cd65e2f12b8b24a2cd`
- Finalized: `2026-08-19T16:32:59.196169+00:00`
- Four independent full-parameter Qwen2.5-7B SFT runs completed at iteration 187.
- Each variant trained 2 epochs over 6000 rows at global batch 64, giving 188 optimizer updates.
- The complete 188-update metric history for every variant is in `handoff/training_metrics.json`.

| Variant | Job | First loss | Last loss | Change | Minimum | Min step | First-5 avg | Last-5 avg | Final grad norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| held_in_correct_6000 | 246946 | 0.546738 | 0.110675 | -0.436063 | 0.104116 | 181 | 0.558574 | 0.112981 | 0.366275 |
| held_in_incorrect_6000 | 246947 | 0.519833 | 0.141047 | -0.378786 | 0.131431 | 183 | 0.513500 | 0.137859 | 0.352724 |
| held_out_correct_6000 | 246948 | 0.562428 | 0.111164 | -0.451264 | 0.105361 | 180 | 0.558108 | 0.115323 | 0.273172 |
| held_out_incorrect_6000 | 246949 | 0.509741 | 0.134787 | -0.374954 | 0.134787 | 187 | 0.517344 | 0.149528 | 0.247408 |

The two correct conditions converge to a visibly lower loss than the two incorrect conditions. That is a fit statistic on different target distributions, not evidence about generalization; the held-in/held-out comparison is decided by the Hyak evaluation described in `handoff/EVAL_HANDOFF.md`.

This Git publication intentionally excludes checkpoints, optimizer/full state, selected and expanded training data, raw source generations, and canary payloads. Checkpoints, the two 106-problem evaluation sets, and the compact handoff are transferred separately with `rsync_to_klone.sh`.
