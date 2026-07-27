# SFT configuration

The authoritative values are in `config/sft_config.json` and `env.sh`.

Publicly sourced choices are the Qwen3-8B-Base initialization, approximately
200K cleaned examples, one epoch, and the contributor's recalled LR near 1e-6.
Batch 200, AdamW details, constant schedule, gradient clip, and TP2/DP2 are our
pinned hardware-aware choices and are not attributed to SLIME.

- Qwen3-8B-Base, BF16, full parameters, one epoch over exactly 200,000 seeded
  cleaned raw rows, LR 1e-6, constant schedule, AdamW 0.9/0.95, weight decay
  0.1, global batch 200, and 1,000 updates.
- Four H200s with TP2/DP2/PP1/CP1, distributed optimizer, CPU optimizer
  offload, full recomputation, and 32,768-token native context. Dynamic batches
  target at most 16,384 packed tokens per GPU, matching the previously
  successful four-H200 TP2/DP2 configuration and avoiding a 32K FP32 logits
  allocation. This packing target is not a context or row-length cap: an
  individual row longer than 16,384 tokens runs alone. Overlength rows above
  the native 32,768-token context are rejected before selection and never
  truncated.
- Qwen3 assistant-only loss includes thinking, `</think>`, final answer,
  `<|im_end|>`, and genuine EOS. A preflight test blocks training on any mask or
  terminal mismatch.
- Weight snapshots are written at every 25K rows. The newest completed cadence
  retains full optimizer/scheduler/RNG/loader/trainer state; when the next save
  is atomically validated, the older full state is pruned but its weights stay.
- The job requests the normal-QoS maximum of 24 hours and self-requeues ten
  minutes before timeout, resuming from the newest validated full state.
