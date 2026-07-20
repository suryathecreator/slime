# Previous fixes carried forward

## Carried over

| Fix | Prior commit and file | Full-run treatment |
|---|---|---|
| Versioned answer state machine, raw generations, rescore hashes, and regression tests | `c5b590e`, `25f64f3`; `examples/qwen3_1_7b_opd_aime2026/aime_answer_v2.py`, `evaluate_aime_once_vllm.py`, `tests/test_aime_answer_v2.py` | Carried into immutable `math500_event_scorer_v1`, atomic per-problem JSON, and new tests. |
| Qwen3 assistant-only masking and terminal supervision | `e98c321`, `cf960ae`; `slime/utils/mask_utils.py`, `slime/rollout/sft_rollout.py`, `examples/qwen3_8b_opd_tillicum/check_sft_loss_mask.py` | Carried and strengthened with explicit `<think>`, `</think>`, final, box, `<|im_end|>`, and genuine-EOS checks. |
| EOS plus `<|im_end|>` stopping and auditable generated terminals | `cf960ae`; `validate_qwen3_generation.py`, `eval_math500_vllm.py` | Carried into rollout/eval stop-ID contracts and per-generation stop fields. |
| Resumable, no-overwrite evaluation | `b645954`; `eval_math500_vllm.py`, `06_eval_math500_vllm.sbatch` | Carried into policy-hashed atomic MATH-500 problem files and four shards. |
| Rolling atomic full-state retention | `2f80f38`; `checkpoint_utils.sh`, SFT/OPD Slurm scripts | Carried: newest cadence full state, older durable HF weights, final full state. |
| Zero-coefficient entropy bypass | `94d8090`; `slime/backends/megatron_utils/loss.py`, `tests/test_policy_loss_entropy.py` | Carried unchanged; faithful entropy coefficient remains 0.0. |
| Rollout sanity artifacts and nonfatal behavior diagnostics | `4e8f653`, made nonfatal for colocate4 in `91233ff`; `slime/rollout/on_policy_distillation.py`, `summarize_opd_sanity.py` | Carried with `OPD_SANITY_FAIL_ON_COLLAPSE=0`; behavior never aborts the faithful run. |
| Validated colocated four-H200 actor/rollout/teacher layout | `7852d74`, result `504c65a`; `05_run_opd_50k_8xh200.sbatch` | Carried and adapted to 16K context: actor/rollout GPUs 0–2, teacher GPU 3, TP1/CP3. |
| CP-compatible sequence geometry | `7c2c850`; `05_run_opd_50k_8xh200.sbatch` | Carried as 16,380, divisible by `2*CP`. |
| Actor packing/OOM guards | `050a961`, `8e0b414`; `env.sh`, OPD Slurm | Carried as explicit 4,096 max tokens/GPU. |
| Chunked teacher/student log-probability computation | `9d89b0d`, retry record `a82b73e`; `env.sh`, OPD Slurm | Carried as the hardware-validated 1,024-token chunk at 16K. |
| Zero train-memory margin and train/rollout offload | `ac6663d`, `7852d74`; `env.sh`, OPD Slurm | Carried for the validated colocated stack. |
| Native RoPE/SGLang compatibility fallback | `7c357c1`, `9ab04c1`, `ddf36d8`; `sglang_launch_native_rope.py`, `container_exec.sh` | Conditionally active on the pinned cluster stack. |
| Correct HF SFT initialization and reference alignment | `3e9c216`; `05_run_opd_50k_8xh200.sbatch` | Carried; actor and reference resolve to the same final 200K HF snapshot before update 1. |
| Megatron DP optimizer and precision-aware CPU offload | `3b42f27`, upstream support `9b40c38`; SFT/OPD Slurm scripts | Carried because it is required by the current four-H200 memory geometry. |
| Global continuation provenance | `c5b590e`; `prepare_remaining_2k_split.py`, continuation manifests | Carried as IDs derived from root rollout ID plus within-rollout order, never a resetting local index alone. |

## Conditional cluster fixes

Optimizer CPU offload, training/rollout offload, memory saver, native RoPE,
packing 4,096, log-probability chunks 1,024, TP/CP geometry, and HF-to-Megatron
initialization are enabled because the current four-H200 colocated stack needs
them. Compatibility fallbacks activate only on the pinned stack.

## Intentionally excluded from the faithful arm

Stable-OPD/Veto, correctness or GRPO rewards, length/truncation penalties,
nonzero base KL, reduced OPD coefficient, terminal-token reweighting, dropping
capped trajectories, synthetic EOS, a lower OPD cutoff, and a 32K OPD cutoff
are not used.
