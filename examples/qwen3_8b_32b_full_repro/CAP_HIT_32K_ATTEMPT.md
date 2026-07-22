# Abandoned 32K context replay attempt

- Job: `183039` (FAILED)
- Source cap-hit rows selected: 58
- Generation attempts reaching engine completion: 58
- Saved/published rows: 0/0
- Score published: no

Each shard finished generating its assigned attempts, but the fail-closed exact-prefix check failed on the first checked row. Examples:

- Index 9: first mismatch at token 287 (`12460` versus `5510`).
- Index 10: first mismatch at token 173 (`320` versus `11`).
- Index 36: rerun stopped at 5159 tokens before the saved 16384-token prefix.
- Index 43: first mismatch at token 95 (`1084` versus `10696`).

## Fidelity decision

A seeded stochastic rerun is not guaranteed to reproduce an earlier sampled prefix. Batch scheduling, kernel and floating-point paths, RNG consumption, engine version, and the requested generation budget can all change token sampling.

Continuing from the saved 16K prefix would condition on an old trajectory while restarting the sampler state. It is not the same experiment as drawing the response from the original prompt with a 32K budget, so it is not reported as a fidelity-preserving evaluation.

Regenerate all 500 prompts from their original inputs under one pinned 32K protocol. Context length alone would still not perfectly reproduce an external result unless the checkpoint, prompt template, tokenizer, stop rules, sampler implementation, seeds, scorer, engine/runtime, batching, and hardware behavior are also aligned.

A fresh 32K evaluation can be run later as an explicit ablation, alongside other protocol ablations, without changing the completed fixed-16K primary result.

## Cancelled downstream jobs

- `183040`: sft_200000_32k_proxy (CANCELLED)
- `183041`: qwen3_32b_teacher_32k_proxy (CANCELLED)
- `183042`: qwen3_8b_base_32k_proxy (CANCELLED)
