# Cleaned OpenThoughts SFT + OPD vLLM Plan

This note records the new cleaned-data experiment before resubmission. Existing
result snapshots under this `results/` directory are preserved and pushed with
the implementation so the earlier accidental base->OPD and corrected 1k
SFT->OPD diagnostics remain available.

## What Changes

| Area | Setting |
| --- | --- |
| Data cleaning | Keep OpenThoughts3 rows only when the assistant contains a complete `<think>...</think>` trace and the row is not mixed-language under the script-count heuristic. |
| Expected cleaning scale | Around 300k valid thinking traces. A previous implementation saw 749,380 incomplete-think removals, 397,642 mixed-language removals, and 332,843 valid rows. Exact counts may differ slightly with tokenizer/dataset revision. |
| Split | Deterministically shuffle valid source rows with seed `1234`, then split first `2/3` into SFT reserve and final `1/3` into OPD reserve. |
| SFT selection | First 25,000 rows from the shuffled SFT reserve. |
| OPD-1k selection | First 1,024 rows from the shuffled OPD reserve. |
| OPD-5k continuation selection | Next 4,096 rows from the same shuffled OPD reserve. |
| Disjointness proof | Metadata records selected `source_row_id`s and verifies no overlap between selected SFT rows and used OPD rows. |
| Learning rate | SFT and OPD both use `1e-6`. |
| Epochs | SFT runs exactly one epoch over 25,000 selected traces. OPD runs one pass over 1,024 traces, then one continuation pass over the next 4,096 traces. |

## Training Shape

| Stage | GPUs | Parallelism | Main memory knobs |
| --- | ---: | --- | --- |
| SFT 25k | 4 H200 | `TP=2`, `CP=1`, `DP=2`, Megatron distributed optimizer (`SFT_ZERO_STAGE=1` legacy wrapper value) | `SFT_MAX_TOKENS_PER_GPU=16384`, full latest optimizer checkpoint, model snapshots every 5k samples. |
| OPD 1k | 4 H200 | actor/rollout/teacher `3/3/1`, `TP=1`, `CP=3` | `OPD_SEQ_LENGTH=32766`, `OPD_MAX_RESPONSE_LEN=31744`, `OPD_MAX_TOKENS_PER_GPU=2048`, logprob chunk `512`, train margin `0`, CPU/offload/recompute enabled. |
| OPD +4k | 4 H200 | resumes from OPD-1k full optimizer checkpoint, fresh 4k data pool | Same OPD knobs; skips old rollout dataset state and starts the new data pool at offset 0. |

The cleaned run no longer uses the Megatron-FSDP/ZeRO-2/3 path. Smoke debugging
showed repeated integration issues, with the latest stage-2 attempt completing
backward but failing to save `fsdp_dtensor` because Megatron expected a
`DTensor` and received a local `Tensor`. The submitted chain therefore uses
Megatron's distributed optimizer over DP=2 and runs a tiny smoke job for that
path only.

## Eval

Only four full MATH-500 evals run:

| Eval | Backend | GPUs | Output |
| --- | --- | ---: | --- |
| Base | Tillicum-local vLLM | 4 H200 | `math500_eval_cleaned_base_vllm/base` |
| SFT-25k | Tillicum-local vLLM | 4 H200 | `math500_eval_cleaned_sft_25k_vllm/sft_025000` |
| OPD-1k | Tillicum-local vLLM | 4 H200 | `math500_eval_cleaned_opd_1k_5k_vllm/opd_001024` |
| OPD-5k | Tillicum-local vLLM | 4 H200 | `math500_eval_cleaned_opd_1k_5k_vllm/opd_005120` |

vLLM eval uses four independent 1-GPU workers with greedy decoding,
`max_model_len=32768`, `gpu_memory_utilization=0.92`, `max_num_seqs=16`, and
`max_num_batched_tokens=131072`. The existing `summarize_eval.py` scorer is
reused so metrics remain comparable.

## Dynamic Generation Budget

Training and eval both use per-prompt caps:

| Context | Rule |
| --- | --- |
| OPD training | `min(31744, OPD_ROLLOUT_MAX_CONTEXT_LEN - prompt_tokens)`, with `OPD_ROLLOUT_MAX_CONTEXT_LEN=32766`. |
| vLLM eval | `min(31744, EVAL_MAX_CONTEXT_LEN - prompt_tokens)`, with `EVAL_MAX_CONTEXT_LEN=32768`. |

Each generated sample records prompt length, requested cap, effective cap, and
whether clamping occurred. This keeps the near-32k generation budget for short
prompts while avoiding context overflow for long prompts.

## Checkpoint Policy

The run is resume-friendly. At each phase, the newest checkpoint keeps full
optimizer state. Once a newer full checkpoint is complete and validated, older
checkpoints may be retained only as model snapshots. The checkpoint pruner must
never remove an in-progress checkpoint directory.

## Estimates

| Stage | Expected | Reserved |
| --- | ---: | ---: |
| vLLM setup + smoke | 20-45m; falls back to scratch `pip --target` if container venv support is missing | 2h |
| Clean/split data | 3.5-5h | 8h |
| Base vLLM eval | 35-75m, cap-heavy 2-3h | 4h |
| SFT 25k | 10-12h | 16h |
| SFT vLLM eval | 35-75m, cap-heavy 2-3h | 4h |
| OPD 1k | 4.5-5.5h | 8h |
| OPD-1k vLLM eval | 35-75m, cap-heavy 2-3h | 4h |
| OPD +4k | 16-18h | 24h |
| OPD-5k vLLM eval | 35-75m, cap-heavy 2-3h | 4h |
| Report | minutes | 30m |
