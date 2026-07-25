# Masked full-SFT checkpoint and MATH-500 handoff

The Tillicum launch trains models only. It does not schedule MATH-500. Each
training job writes a content-addressed Hugging Face checkpoint manifest next
to the scratch outputs, and those final checkpoints are intended to be copied
to another system for evaluation. `checkpoint_sources.json` is the
authoritative inventory of the base model and all 14 trained models.

The model was trained by Slime's speedy OPD-adjacent pipeline as full-parameter
SFT from `Qwen/Qwen3-8B-Base`; Axolotl supplies the frozen trace identities and
masking semantics, not the trainer.

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

## One-H200 resumable evaluation

`run_one_h200.sbatch` is designed for one H200 on Klone's `ckpt-all`
partition. GPU checkpoint jobs may be interrupted/requeued, so the evaluator
uses one shard, continuous vLLM batching, and an atomic, fsynced JSON commit
immediately after each of the 500 examples finishes. Re-running the same job
with `--resume` skips every compatible completed example. Only in-flight
examples can be repeated after interruption.

Before submission, provide a Python environment containing the same compatible
versions of vLLM, Transformers, Datasets, and the math-verification stack used
by this repository. Then:

```bash
sbatch --account=YOUR_KLONE_ACCOUNT \
  --export=ALL,MODEL_DIR=/gscratch/scrubbed/suryadv/Axolotl-Masked-SFT/checkpoints/2k/random_mask_70,MODEL_MANIFEST=/gscratch/scrubbed/suryadv/Axolotl-Masked-SFT/manifests/2k/random_mask_70.json,EVAL_STAGE=2k_random_mask_70,EVAL_OUTPUT_DIR=/gscratch/scrubbed/suryadv/Axolotl-Masked-SFT/evals/2k_random_mask_70,EVAL_PYTHON=/path/to/python \
  examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/run_one_h200.sbatch
```

The exact scientific and engine settings are frozen in `eval_policy.json`:
MATH-500 revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`,
original row order, the existing Qwen3 thinking prompt, greedy decoding,
per-row seed `1234 + eval_index`, response budget
`32768 - rendered_prompt_tokens - 64`, both stop IDs 151643 and 151645, and
`math500_strict_boxed_scorer_v3`. Think tags are diagnostic only; the scorer
uses the last complete balanced boxed answer across the whole response and
marks cap hits incorrect.

Evaluate `base_8b` and each of the 14 trained checkpoints once. The 2K
`base_8b`, `correct_only`, and `unmasked` values are shared verbatim across all
2K comparison tables.

Klone's checkpoint partition behavior is documented at
<https://hyak.uw.edu/docs/compute/checkpoint/>.

## Return and publish results

When all 500 examples finish, the Slurm wrapper aggregates them, writes
`summary.json` and `summary.sha256`, and builds a self-validating result bundle
in the evaluation output directory. The bundle contains:

- `bundle_manifest.json`
- `eval_policy.json`
- `checkpoint_manifest.json`
- `summary.json` and `summary.sha256`
- all 500 `problems/problem_XXXX.json` artifacts

Copy that bundle back beside a clean pull of this branch and import it:

```bash
python3 examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/import_result_bundle.py \
  --bundle /path/to/eval/bundle \
  --repo-root "$PWD"
```

The importer fails closed on the policy, checkpoint identity, scorer, stage,
row order, hashes, and count. It writes compact, reviewable results to the
`results/` directory for the matching 40K or 2K example. Raw generations stay
in the external bundle. Review the diff, then commit and push normally; do not
resolve a divergence implicitly.
