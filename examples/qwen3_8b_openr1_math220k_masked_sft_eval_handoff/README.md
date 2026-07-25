# Masked full-SFT checkpoint and MATH-500 handoff

The Tillicum launch trains models only. The current Klone evaluation covers the
eleven completed 2K checkpoints that are already present locally. Base and 40K
evaluation are deferred until those checkpoint copies exist. All models were
trained from exactly
`Qwen/Qwen3-8B-Base@49e3418fbbbca6ecbdf9608b4d22e5a407081db4`;
no floating base-model revision is permitted.

`checkpoint_sources.json` remains the authoritative cross-system inventory.
`available_eval_manifest.json` is the authoritative list for this local run.
Every listed checkpoint has a content-addressed manifest.

## Transfer

After a training job finishes, copy the entire final `source_checkpoint`
directory from `checkpoint_sources.json`, plus its `source_manifest`, with a
resumable transfer tool such as `rsync --partial --append-verify`. On the
destination, verify the copy before evaluation:

```bash
python3 examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py \
  verify \
  --manifest /path/to/checkpoint_manifest.json \
  --checkpoint /path/to/transferred/checkpoint
```

The manifest identity hashes every file, but not the machine-specific source
path. A verified copy therefore has the same model identity on either system.

## Fast Axolotl evaluator used for this run

This run directly calls
`../Axolotl-Masked-SFT/scripts/openr1_axolotl/evaluate_math_vllm.py` at
Axolotl commit `6b8f0e3314e3d162260cdc35d84741c3da163f30`.
That is Axolotl's MATH-500 counterpart to its completed fast HARP evaluator.
It reuses `evaluate_harp_vllm.build_llm`: one model load per job, compiled
vLLM continuous batching, BF16, prefix caching, greedy decoding, and Qwen EOS
plus `<|im_end|>` stopping. It renders Axolotl's established OpenR1 prompt,
extracts the final answer with `common.extract_final_answer`, and grades it
with `math_verify`.

Every transferred checkpoint contains the same three tokenizer artifacts from
the pinned base revision, and their hashes are frozen in
`available_eval_manifest.json`. Transformers 4.57.6 cannot initialize their
list-form `extra_special_tokens` field, so preparation creates a verified
runtime overlay: `tokenizer.json` and `chat_template.jinja` remain byte-exact,
while only that metadata field is converted to the empty mapping that produces
the same tokenizer as the tested `extra_special_tokens={}` load override.
`TOKENIZER_PROVENANCE.json` records both source and runtime hashes. The direct
Axolotl evaluator receives this overlay as `--model_name`; model weights still
load from the independently verified checkpoint directory.

Each checkpoint is split into four contiguous 125-problem jobs. The evaluator
flushes every returned row to `predictions.jsonl` and
`raw_generations.jsonl`; on requeue it reads existing problem IDs and skips
them. Generation chunks contain 32 problems, so completed chunks survive
preemption while vLLM still batches continuously. `run_axolotl_h200.sbatch`
adds Slurm requeue and a ten-minute `USR1` handler. Exact settings are frozen
in `axolotl_eval_policy.json`.

Submit only after validation:

```bash
bash examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/submit_available_axolotl.sh \
  --test-only-shapes
bash examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/submit_available_axolotl.sh \
  --submit
```

The real submission is 11 unthrottled arrays of `0-3`: 44 one-H200 tasks on
`raivn-ckpt` / `ckpt-all`, followed by one merge per checkpoint and a final
all-results audit.

## Exact local paths

The local checkpoint root is
`/mmfs1/gscratch/scrubbed/suryadv/repos/SLIME/checkpoints`. The shared prefix
below is:

```text
/mmfs1/gscratch/scrubbed/suryadv/repos/SLIME/checkpoints/qwen3_8b_openr1_math220k_masked_sft_2k/v1/5c3230a293f7acb2
```

| Variant | Model directory | Manifest | Raw/merged output | Compact result |
|---|---|---|---|---|
| `correct_only` | `outputs/training/correct_only/weights/iter_0000009` | `handoff/checkpoints/correct_only.json` | `outputs/math500/correct_only` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/correct_only` |
| `unmasked` | `outputs/training/unmasked/weights/iter_0000009` | `handoff/checkpoints/unmasked.json` | `outputs/math500/unmasked` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/unmasked` |
| `random_mask_25` | `outputs/training/random_mask_25/weights/iter_0000009` | `handoff/checkpoints/random_mask_25.json` | `outputs/math500/random_mask_25` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/random_mask_25` |
| `random_mask_50` | `outputs/training/random_mask_50/weights/iter_0000009` | `handoff/checkpoints/random_mask_50.json` | `outputs/math500/random_mask_50` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/random_mask_50` |
| `random_mask_70` | `outputs/training/random_mask_70/weights/iter_0000009` | `handoff/checkpoints/random_mask_70.json` | `outputs/math500/random_mask_70` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/random_mask_70` |
| `random_mask_80` | `outputs/training/random_mask_80/weights/iter_0000009` | `handoff/checkpoints/random_mask_80.json` | `outputs/math500/random_mask_80` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/random_mask_80` |
| `random_mask_90` | `outputs/training/random_mask_90/weights/iter_0000009` | `handoff/checkpoints/random_mask_90.json` | `outputs/math500/random_mask_90` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/random_mask_90` |
| `inverse_tau_0p20` | `outputs/training/inverse_tau_0p20/weights/iter_0000009` | `handoff/checkpoints/inverse_tau_0p20.json` | `outputs/math500/inverse_tau_0p20` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/inverse_tau_0p20` |
| `inverse_tau_0p05` | `outputs/training/inverse_tau_0p05/weights/iter_0000009` | `handoff/checkpoints/inverse_tau_0p05.json` | `outputs/math500/inverse_tau_0p05` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/inverse_tau_0p05` |
| `margin_mask` | `outputs/training/margin_mask/weights/iter_0000009` | `handoff/checkpoints/margin_mask.json` | `outputs/math500/margin_mask` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/margin_mask` |
| `prob_ratio_mask` | `outputs/training/prob_ratio_mask/weights/iter_0000009` | `handoff/checkpoints/prob_ratio_mask.json` | `outputs/math500/prob_ratio_mask` | `examples/qwen3_8b_openr1_math220k_masked_sft_2k/results/prob_ratio_mask` |

Model, manifest, and raw-output columns are relative to the shared checkpoint
prefix. Compact-result paths are relative to the SLIME repository root.

Klone's checkpoint partition behavior is documented at
<https://hyak.uw.edu/docs/compute/checkpoint/>.

## Return and publish results

After all four shards complete, the dependent merge requires all 500 unique
problem IDs in benchmark order, checks its recomputed totals against the
native Axolotl shard metrics, and writes merged predictions, raw generations,
summary, and provenance. Compact reviewable results are written to the result
paths above. Raw generations remain under the ignored checkpoint output tree.

Review local commits normally. This workflow never pushes or invokes the older
auto-publish job.
