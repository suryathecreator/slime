# OpenR1-Math-220k Reproduction

## Why This Direction

The OpenThoughts paths and all their scratch artifacts remain preserved, but
new training is paused. The first cleaner retained 429,713 rows after reporting
743,814 incomplete-thinking and 94,920 mixed-language rows. The stricter
English cleaner later timed out after 390,000 source rows because its Lingua
microspan rule rejected every row. At the same time, low-data OpenThoughts SFT
runs showed inconsistent formatting, repeated reasoning, and poor termination.
Those symptoms may improve with substantially more clean SFT data, but cleaning
is expensive and the resulting policy may not generalize from only 25k rows.

This replacement isolates the data question by using the curated `default`
subset of `open-r1/OpenR1-Math-220k`, which has 93,733 prompts and supplied
reasoning-completeness/correctness annotations.

## Deterministic Data Contract

- Dataset: `open-r1/OpenR1-Math-220k`, config `default`, split `train`.
- Revision: `dc748648036c1ed619b020e056dc4b603eb39817`.
- Seed: `1234`.
- Prompt rows are uniformly shuffled without replacement.
- The first 50,000 eligible unique prompts form SFT; the next 5,120 form OPD.
- An eligible trace has `is_reasoning_complete=true`, either supplied
  `correctness_math_verify=true` or `correctness_llama=true`, and an ordered
  `<think>...</think>` pair.
- When several traces qualify, one generation index is selected uniformly and
  deterministically from a seed derived from `1234` and the row UUID.
- The SFT assistant trace is retained verbatim. OPD retains that trace only as
  provenance and trains from the disjoint prompt.
- Metadata proves zero intersection by source row ID, UUID, and prompt hash.

No new Math Verify pass, language detector, or quality cleaner is applied.

## Training And Evaluation

| Stage | Samples | GPUs | Parallelism | Updates | LR | Walltime |
|---|---:|---:|---|---:|---:|---:|
| Data preparation | 50k + 5,120 | 1 | CPU-heavy download/select | - | - | 6h |
| SFT | 50,000 | 4 H200 | TP2 / CP1 / DP2 | 200 | 1e-6 | 24h |
| SFT MATH-500 | 500 | 4 H200 | 4 independent vLLM workers | - | - | 24h |
| OPD-1k | 1,024 | 4 H200 | actor/rollout/teacher 3/3/1, TP1/CP3 | 8 | 1e-6 | 24h |
| OPD-1k MATH-500 | 500 | 4 H200 | 4 independent vLLM workers | - | - | 24h |
| OPD continuation | 4 x 1,024 | 4 H200 | same colocated layout | 32 | 1e-6 | 24h each |
| OPD-5,120 MATH-500 | 500 | 4 H200 | 4 independent vLLM workers | - | - | 24h |
| Final report | - | 1 H200 | report-only | - | - | 6h |

The chain is strict `afterok`, so only one job can run at a time. SFT uses the
Qwen3 loss mask and Megatron distributed optimizer. Qwen3 thinking is explicit,
and generation stops on either `<|im_end|>` (`151645`) or `<|endoftext|>`
(`151643`). Eval uses per-prompt context caps and resumable schema-2 chunks that
preserve special-token text plus raw prefill/generated IDs.

SFT writes HF snapshots every 5,000 samples. OPD writes an HF snapshot every
1,024 samples. A completed new endpoint becomes the sole full optimizer
checkpoint; older endpoints remain as model-only HF snapshots. The OPD-1k full
checkpoint is retained until the 2,048 continuation endpoint is fully written
and validated.

Expected serialized compute time is approximately 34-53 hours plus queue delay.
The larger walltimes are failure headroom, not runtime estimates.
