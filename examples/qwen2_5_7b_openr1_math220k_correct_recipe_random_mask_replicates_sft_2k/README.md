# Qwen2.5-7B low-rate random-mask replicates

This extension trains six new full-SFT checkpoints on the exact ordered 2,000
traces used by the completed Qwen2.5-7B correct-recipe masked experiment. It
does not repeat selection, tokenization, margin scoring, correct-only training,
or any previously trained variant.

The new variants mask 5% and 15% of ordinary assistant tokens on incorrect
traces at seeds 42, 43, and 44. Correct traces and protected added-control and
literal think-markup positions retain weight 1. For a fixed seed, the same
stable draw is used at both rates, so the 5% masked positions are an exact
subset of the 15% positions. All six datasets retain the source trace order.

Training copies the successful recipe verbatim: Qwen2.5-7B base, full SFT,
10,240-token sequences, global batch 64, four epochs/125 updates, AdamW at
`5e-6`, cosine decay to `1e-6`, 3% warmup, betas `(0.9, 0.95)`, epsilon
`1e-8`, weight decay `1e-4`, BF16, and gradient clipping at 1.0. Each Slurm
stage has a two-hour cap. The chain is strict serial `afterok`: build and
validate data, run a five-update prefix canary, train six variants, then
finalize checkpoint and comparison manifests. It submits no evaluation or
transfer jobs.

```bash
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k/submit_training_only.sh --dry-run
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k/submit_training_only.sh --submit
```

After training, the handoff inventory contains only the six new checkpoints;
the comparison inventory references the prior 12 targets and appends the six
new targets. Transfer and verification are explicit:

```bash
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k/rsync_to_klone.sh --check
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k/rsync_to_klone.sh --transfer
bash examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_2k/rsync_to_klone.sh --verify
```
