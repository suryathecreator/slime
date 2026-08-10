# Qwen2.5-7B correct-recipe masked full-SFT

This training-only experiment extends the successful Qwen2.5-7B 8K
`correct_only` run without modifying or retraining it. Every new run starts from
the same pinned Qwen2.5-7B base and copies its realized four-epoch, 125-update
`memory_r3` recipe exactly.

The realized recipe is full-parameter BF16 SFT on four H200s with TP=2, DP=2,
CP=1, PP=1, ZeRO-1, optimizer CPU offload, recomputed loss, 10,240-token
sequences, and a 16,384-token per-GPU dynamic-batch ceiling. Global and rollout
batch size are both 64. Four epochs over 2,000 rows produce exactly 125 updates
(final iteration 124). AdamW uses LR `5e-6`, cosine decay to `1e-6`, 3% warmup
from zero, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `1e-4`, and gradient
clip `1.0`. The only necessary runtime-adapter difference from correct-only is
the pre-existing weighted-SFT adapter that passes the per-token weights through
unchanged.

The frozen correct-only selection is the source of the first 1,000 correct
traces. A deterministic seed-42 reservoir selects 1,000 eligible wrong traces
from the same pinned OpenR1-Math-220k revision, excluding all 2,000 frozen
correct-only problem IDs. The resulting ordered 1,000-correct plus 1,000-wrong
set is shared verbatim across all ten variants.

The variants are unmasked; random wrong-token masking at 25%, 50%, 70%, 80%,
and 90%; inverse-margin weighting at tau 0.20 and 0.05; positive-margin
masking; and probability-ratio masking. Correct tokens, literal think markup,
added control tokens, and the terminal `<|im_end|>` remain at weight 1. The
template newline remains at weight 0 and no synthetic EOS is appended.

The chain is strictly serial and contains no correct-only or evaluation job:
data preparation, Qwen2.5-7B margin scoring and variant construction, a
five-update fractional-weight canary, ten full-SFT jobs, and final handoff
verification. Every Slurm stage has a strict two-hour wall-time cap.

```bash
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k/submit_training_only.sh --dry-run
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k/submit_training_only.sh --submit
```

After training, transfer only the ten new HF checkpoints. The handoff comparison
manifest references the already-transferred Qwen2.5-7B base and correct-only
checkpoint instead of duplicating them.

```bash
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k/rsync_to_klone.sh --check
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k/rsync_to_klone.sh --transfer
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k/rsync_to_klone.sh --verify
```
