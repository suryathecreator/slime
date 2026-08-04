# Qwen3-0.6B-Base OpenR1 masked full-SFT rerun

This is a train-only rerun of the same frozen 2,000-example correct/wrong/everything suite on `Qwen/Qwen3-0.6B-Base` at revision `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`. It produces eleven checkpoints and does no evaluation.

## Frozen training contract

- Full-parameter SFT, BF16, one epoch over exactly 2,000 rows.
- Global batch size 16, DP=4, TP=1, PP=1, CP=1.
- Maximum sequence length **32,768** and dynamic packing cap 32,768 tokens per GPU.
- Effective gradient accumulation steps **1**. Data preparation replays Slime's actual DP packer after the seed-42 shuffle and fails unless every one of the 125 updates has one microbatch per DP rank.
- AdamW: LR `1e-5`, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `1e-4`, global gradient norm cap `1.0`.
- Linear warmup from LR 0 for 5% of the 2,000-sample schedule (100 samples, or 6.25 global-batch equivalents), then cosine decay to minimum LR 0. `--lr-decay-iters=125` pins the schedule length.
- Dropout is zero. Loss is normalized by retained token mass, matching the earlier suite; sparse masks therefore retain full update scale.
- Final HF and full-state checkpoint tag: `iter_0000124`.

The variants are, in launch order: `correct_only`, `unmasked`, `random_mask_70`, `inverse_tau_0p20`, `inverse_tau_0p05`, `random_mask_25`, `random_mask_50`, `random_mask_80`, `random_mask_90`, `margin_mask`, and `prob_ratio_mask`. Margins are recomputed under the pinned 0.6B base model.

## Boundary canaries

Full training is gated by two independent 32-example, two-update canaries:

1. The custom pretokenized weighted-SFT path used by all eleven variants.
2. Slime's stock message-based Qwen3 SFT path over the exact same ordered traces.

For both paths, the job saves every real DP-rank runtime batch and audits all 64 examples. It emits a JSONL row for every input position containing the decoded input token, the next-token causal label, and the shifted loss weight. The gate checks exact source-token identity, assistant-entry boundaries, `<|im_end|>`, absence of synthetic EOS, global batch 16, and one microbatch per rank. It then greedily compares eight prompts under base, custom-after-two-updates, and stock-after-two-updates; either trained path is rejected if prompt/role-copy starts increase over base.

Key artifacts live under `${EXPERIMENT_ROOT}`:

- `data/dynamic_batch_schedule_audit.json`
- `outputs/canary/runtime_batch_audit.json`
- `outputs/canary/runtime_shifted_labels.jsonl`
- `outputs/canary/generation_diagnostic.json`
- `handoff/checkpoints/*.json`
- `manifests/training_chain.json`

## Validation and launch

Run the repository-only tests with:

```bash
pytest -q tests/test_qwen3_0_6b_masked_sft_2k.py
python3 examples/qwen3_0_6b_openr1_math220k_masked_sft_2k/validate_config.py \
  --example-dir examples/qwen3_0_6b_openr1_math220k_masked_sft_2k
```

After committing and pushing the dedicated branch, submit the strict `afterok` chain with:

```bash
bash examples/qwen3_0_6b_openr1_math220k_masked_sft_2k/submit_training_only.sh
```

The chain is model preparation -> exact-token data preparation -> both canaries and audits -> 0.6B margin scoring and variant construction -> eleven serial training jobs. A failed gate prevents every downstream training job from starting.
