# OPD configuration

The authoritative values are in `config/opd_config.json` and `env.sh`.

Four samples, 16K response cutoff, temperature 1.0/top-p 1.0 in the public
script, batch 16/global 64, LR 1e-6 constant, and the pure OPD coefficients are
public reference settings. Temperature 0.8/top-k -1 and the four-H200
TP1/CP3/offload geometry are the explicitly requested local contract; they are
reported as deviations rather than attributed to the public script.

- Final 200K SFT student, Qwen3-32B thinking teacher, 1,472 seeded reserve raw
  rows, four samples per row, 5,888 trajectories, prompt batch 16, global batch
  64, and 92 updates.
- Temperature 0.8, top-p 1.0, top-k -1, 16,384 configured response cap bounded
  by the remaining 16,380-token CP3-compatible context.
- Pure public OPD: LR 1e-6 constant, OPD coefficient 1.0, base KL 0.0, entropy
  0.0, and scalar reward 0.0.
- Four-H200 colocated layout: actor and rollout share GPUs 0–2, teacher uses GPU
  3, TP1/CP3, packing 4,096, log-probability chunks 1,024, zero train margin,
  train/rollout offload, CPU optimizer offload, and recomputed loss.
- Weights are retained at 368/736/1,104/1,472 prompt rows. Only the newest
  quarter retains full training state; the final 100% save remains full.
- Stopping, cap, formatting, and repetition metrics are nonfatal diagnostics and
  do not abort this faithful run.
