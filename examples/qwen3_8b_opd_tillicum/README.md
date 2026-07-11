# Qwen3-8B SFT + OPD on Tillicum

This directory contains thin Tillicum wrappers for a Qwen3-8B SFT plus
on-policy distillation experiment using OpenThoughts3-1.2M math data.

The wrappers intentionally reuse slime's existing paths:

- `docs/en/get_started/quick_start.md` for the container and Megatron
  conversion workflow.
- `scripts/models/qwen3-8B.sh` for the student Megatron model args.
- `examples/on_policy_distillation/run-qwen3-8B-opd.sh` for the SGLang
  teacher OPD shape.
- `scripts/run-qwen3-4B-base-sft.sh` for the SFT rollout/training shape.
- `examples/eval_multi_task/multi_task.sh` and `multi_task.yaml` for eval.

The reproduction branch also carries small runtime fixes needed for this run,
including skipping entropy allocation when `--entropy-coef 0.00` and handling
non-scalar rollout rewards in logging.

The isolated strict-English v2 SFT -> OPD run is specified in
[`results/strict_en_v2_experiment.md`](results/strict_en_v2_experiment.md).
It saves a reusable all-domain corpus, gates cleaning counts against the
comparison implementation, derives a high-quality math subset, and dynamically
dispatches the batch-aligned training chain.

## Required environment

Source `env.sh` before running commands:

```bash
cd /gpfs/scrubbed/suryadv/repos/slime
source examples/qwen3_8b_opd_tillicum/env.sh
```

Important variables:

- `ACCOUNT`: Slurm account. Default: `raivn`.
- `PARTITION`: Slurm partition. Default: `gpu-h200`.
- `QOS`: Slurm QOS. Default: `normal`.
- `SCRATCH_ROOT`: root for all generated data, checkpoints, caches, logs, and
  the Apptainer image. Default:
  `/gpfs/scrubbed/suryadv/slime-qwen3-8b-opd`.
- `DATA_ROOT`: prepared datasets. Default: `$SCRATCH_ROOT/data`.
- `MODEL_ROOT`: HF model snapshots and Megatron torch_dist conversion.
  Default: `$SCRATCH_ROOT/models`.
- `OUTPUT_ROOT`: training/eval outputs. Default: `$SCRATCH_ROOT/outputs`.
- `HF_HOME`: Hugging Face cache under scratch. Default: `$SCRATCH_ROOT/hf_home`.
- `WANDB_MODE`: default `offline`.
- `SLIME_CONTAINER_FORMAT`: Apptainer image format. Default: `sandbox`,
  because Tillicum's Apptainer produced invalid SquashFS SIFs for this large
  Docker image during testing. Set to `sif` to force SIF output.
- `SLIME_SIF`: Apptainer image/sandbox path. Default:
  `$SCRATCH_ROOT/containers/slime_latest.sandbox`.

The scripts avoid writing caches/checkpoints/data under home.

## Dry checks

Dry checks do not pull the container, download data/models, install packages, or
submit real jobs.

```bash
cd /gpfs/scrubbed/suryadv/repos/slime
source examples/qwen3_8b_opd_tillicum/env.sh
bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh
```

The dry check runs `bash -n`, Python bytecode compilation, and
`sbatch --test-only` for the four Slurm scripts. If `$SLIME_SIF` already
exists and `RUN_CONTAINER_CHECKS=1` is set, it also verifies imports inside the
container.

## Setup and launch after approval

Run the setup steps manually, in this order:

```bash
bash examples/qwen3_8b_opd_tillicum/00_pull_or_load_container.sh
bash examples/qwen3_8b_opd_tillicum/01_prepare_env.sh
bash examples/qwen3_8b_opd_tillicum/run_all_dry_check.sh
```

Then launch only after explicit approval. The conservative 8-GPU downstream
chain is:

```bash
bash examples/qwen3_8b_opd_tillicum/submit_opd_1k_32k_chain.sh
```

The corrected 4-GPU SFT-loaded chain is:

```bash
bash examples/qwen3_8b_opd_tillicum/submit_opd_1k_32k_sft_colocate4_chain.sh
```

This chain preserves the completed SFT checkpoint/eval and runs only the
downstream 1k OPD experiment:

- OPD data pool: 10,000 row-disjoint OpenThoughts3 math prompts.
- OPD training horizon: 1,024 effective samples, `8` rollouts of `128`
  prompts each.
- OPD response cap: `31744` generated tokens.
- OPD actor/rollout/teacher GPUs: `3/3/1` on one 4xH200 allocation, with actor
  and rollout colocated on GPUs `0,1,2` and the teacher on physical GPU 3.
- OPD actor parallelism: tensor parallel `1`, context parallel `3`,
  `OPD_MAX_TOKENS_PER_GPU=2048`.
- OPD actor logprob chunking: `OPD_LOG_PROBS_CHUNK_SIZE=512`.
- OPD eval: final checkpoint only, stage `opd_001024`.
- Final report: base -> final SFT -> final OPD.

Important: run label `1k_32k` from job `156276` is an accidental but useful
base -> OPD test. That run attempted to pass the final SFT HF snapshot to
Megatron `--load` without the explicit HF-load path, so the actor fell back to
the base torch_dist checkpoint. Do not interpret `1k_32k` as SFT -> OPD.

The corrected 4-GPU chain uses `OPD_INITIAL_LOAD_MODE=hf` and initializes the
actor from the final SFT HF weights snapshot `$SFT_FINAL_HF_DIR`. This is
intentional for the 4-GPU topology: the completed SFT full optimizer checkpoint
was saved with tensor parallel `2`, while the corrected colocated OPD actor uses
tensor parallel `1` and context parallel `3`. Megatron cannot optimizer-resume
that SFT checkpoint across the TP mismatch. After OPD starts, its own full
optimizer checkpoint under `$OPD_SAVE_DIR` is the fidelity-safe continuation
point for more OPD.

The corrected chain also sets `OPD_REF_LOAD_DIR=$SFT_FINAL_HF_DIR`, so logged
`train/kl_loss` is a diagnostic KL against the final SFT policy rather than the
base model. The coefficient remains `--kl-loss-coef 0.00`, so this does not
change gradients or the OPD objective.

The corrected 4-GPU default run label is `1k_32k_sft_colocate4`. It keeps the
4-H200 ceiling by colocating actor training and student rollout engines on Ray
GPUs `0,1,2`, with the Qwen3-32B teacher logprob server on physical GPU `3`.
The actor uses `TP=1`, `CP=3`, `OPD_SEQ_LENGTH=32766`,
`OPD_MAX_RESPONSE_LEN=31744`, `OPD_MAX_TOKENS_PER_GPU=2048`, and
`OPD_LOG_PROBS_CHUNK_SIZE=512`, with `--colocate`, `--offload-train`,
`--offload-rollout`, optimizer CPU offload, and `--recompute-loss-function`
enabled.

The `2048` actor packing default and `512` logprob chunk size are
memory-scheduling guards for colocated 4-GPU retries that reached actor
training and OOMed in fused cross-entropy/logprob computation. They do not
reduce the 31,744-token generation cap or the 1,024-sample OPD training
horizon.

The colocate4 submitter also sets `OPD_TRAIN_MEMORY_MARGIN_BYTES=0` so
torch-memory-saver does not reserve its default 1 GiB train allocation margin.
Other OPD entrypoints keep Megatron's default margin of `1073741824` bytes.

The corrected 4-GPU chain also enables OPD rollout sanity logging before actor
updates. The guard logs `OPD_SANITY` metrics and writes JSON under
`$OPD_SANITY_REPORT_DIR`, but corrected colocate4 sets
`OPD_SANITY_FAIL_ON_COLLAPSE=0`, so threshold violations are diagnostic and do
not stop training. The wrapper writes `opd_sanity_summary.{json,csv,md}` under
`$OPD_SANITY_SUMMARY_DIR`; logprob/KL values are response-token-weighted rollout
means, response lengths are sample statistics, and cap/completion/final-answer
rates are sample fractions. MATH-500 scoring remains the source of accuracy
numbers.

## Reproducing On Another Slurm Cluster

These wrappers are Tillicum-shaped, but the core workflow is portable to a
Slurm cluster with Apptainer or Singularity, one 8-GPU node per job, outbound
access to Hugging Face and Docker Hub, and enough scratch space for large model
checkpoints.

Clone the fork and switch to the reproduction branch:

```bash
git clone https://github.com/suryathecreator/slime.git
cd slime
git checkout opd-reproduction
```

Choose cluster-local paths and Slurm settings. Keep all generated state on
scratch or project storage, not home:

```bash
export ACCOUNT="<your-account>"
export PARTITION="<your-gpu-partition>"
export QOS="<your-qos>"
export SCRATCH_ROOT="/path/to/scratch/${USER}/slime-qwen3-8b-opd"
export CONTAINER_BIND_ROOTS="$(pwd),${SCRATCH_ROOT},/tmp"

# Match your site's GPU gres. Examples: gpu:8, gpu:a100:8, gpu:h100:8.
export GPU_GRES="gpu:8"

# Use sif if your Apptainer can build a normal SIF for slimerl/slime:latest.
export SLIME_CONTAINER_FORMAT="sandbox"

source examples/qwen3_8b_opd_tillicum/env.sh
```

Run the same preparation commands:

```bash
bash examples/qwen3_8b_opd_tillicum/00_pull_or_load_container.sh
bash examples/qwen3_8b_opd_tillicum/01_prepare_env.sh
bash examples/qwen3_8b_opd_tillicum/container_exec.sh \
  python examples/qwen3_8b_opd_tillicum/02_prepare_openthoughts3_math_sample.py
```

Submit the dependency chain, overriding the Tillicum `h200` gres embedded in
the sbatch files:

```bash
GPU_GRES=gpu:h200:8 bash examples/qwen3_8b_opd_tillicum/submit_opd_1k_32k_chain.sh
```

If your cluster uses `--gpus-per-node` instead of `--gres`, replace the
`--gres "$GPU_GRES"` arguments above with your site's GPU request flag. If your
cluster uses Docker rather than Apptainer/Singularity, run the same Python and
Slurm entrypoints inside `slimerl/slime:latest` and bind the repo plus
`$SCRATCH_ROOT` into the container; `container_exec.sh` is the only
Apptainer-specific layer.

## Outputs

- SFT split: `$SFT_PARQUET` (JSONL by default despite the legacy variable
  name).
- OPD prompt split: `$OPD_JSONL`
- Data metadata: `$SPLIT_METADATA`
- Student HF snapshot: `$STUDENT_HF_DIR`
- Teacher HF snapshot: `$TEACHER_HF_DIR`
- Student Megatron torch_dist: `$STUDENT_TORCH_DIST_DIR`
- SFT full optimizer checkpoint: `$SFT_SAVE_DIR`
- OPD full optimizer checkpoint: `$OPD_SAVE_DIR`
- SFT model-only eval snapshots: `$SFT_HF_SNAPSHOT_DIR`
- OPD model-only eval snapshots: `$OPD_HF_SNAPSHOT_DIR`
- OPD trained-data manifest: `$OPD_TRAINED_MANIFEST`, copied into the final
  OPD HF snapshot as `opd_trained_manifest.json`
- Eval summaries and curves: `$BASE_EVAL_OUTPUT_DIR`, `$SFT_EVAL_OUTPUT_DIR`,
  `$OPD_EVAL_OUTPUT_DIR`, `$COMBINED_EVAL_OUTPUT_DIR`
- Checkpoint storage reports: `$CHECKPOINT_REPORT_DIR`

## Slurm resources

Each job requests one node with `--gres=gpu:h200:8`, `--ntasks=1`,
`--cpus-per-task=64`, and all node memory. The account, partition, and QOS are
passed at submit time from the environment variables above.

The intended wall-clock budget after model/data/container preparation is:

- SFT 25k: 8 hours.
- OPD 1k/32k-cap: 18 hours.
- MATH-500 greedy eval: final OPD checkpoint 3 hours, base 3 hours.
- Report aggregation job: 30 minutes on the cluster-minimum `gpu:h200:1`.

The main runtime risk is the Qwen3-32B teacher logprob server throughput during
OPD. The corrected offload chain starts from the completed final SFT HF weights
snapshot, runs 1k OPD with the near-32k response cap, evaluates the final OPD
checkpoint with 4 one-GPU SGLang engines and concurrency 4, reuses or runs the
fixed base eval, then generates the combined final figure.

For the corrected 4-GPU SFT-loaded OPD chain, keep the generation/eval response
cap at `31744` and use the colocated `3/3/1` actor/rollout/teacher layout
described above. Job `158041` OOMed during the first actor train step with only
about 68 MiB free on one H200 in the older separated `2/1/1` layout. The
colocated CP=3 layout spreads actor long-sequence activation memory across
three actor GPUs while preserving the 4-GPU ceiling.

The older `1k_32k` OPD run is an accidental base -> OPD experiment because the
OPD job was not loaded from the full SFT Megatron checkpoint. If its final eval
times out before Slime writes `debug_eval_0.pt`, use
`submit_cleanup_base_opd_2gpu.sh`. That cleanup job requests 2 H200s for 18
hours, skips any completed accidental OPD summary, reruns only missing eval
stages, and writes a base -> OPD-only combined report with SFT omitted from the
comparison. The cleanup path is artifact-idempotent: if a nested eval exits
nonzero after writing the expected `summary.json`, the wrapper treats that
stage as salvaged and continues to the remaining missing pieces.

For future OPD runs, do not treat the current 8-GPU allocation as the preferred
long-term configuration. Use optimizer CPU offload and retune the actor/rollout/
teacher split so training can run with fewer GPUs, targeting the 4-GPU total
training job used by `submit_opd_1k_32k_sft_colocate4_chain.sh` if the offload
path is stable enough. This will likely trade wall clock time for lower GPU
occupancy, but it is the right direction for follow-up runs once the
reproduction path is validated.

MATH-500 summaries report `accuracy` with parse failures counted wrong,
`accuracy_on_parseable` as a diagnostic over parseable responses only, and
`parse_failure_rate` separately. Combined reports include a labeled SVG/PNG
curve with light-blue SFT shading and light-purple OPD shading.

Tracked result snapshots live under `results/`. The accidental base -> OPD
salvage eval is recorded in `results/accidental_base_opd_salvage_summary.json`
and `results/accidental_base_opd_salvage_summary.csv`; it is diagnostic only
and not the corrected SFT -> OPD experiment. The corrected colocate4 diagnostic
note is recorded in `results/corrected_colocate4_diagnostics.md`; it defines the
sanity/KL table columns, averaging rules, dataset facts, and the accidental
base -> OPD loader/ref-sync issue.

The completed corrected SFT -> OPD colocate4 1k/32k result is recorded under
`results/corrected_sft_opd_colocate4_1k_32k/`. It includes the combined
base -> SFT -> OPD MATH-500 summaries, the generated curve data, and the OPD
rollout reward/logprob/sanity tables.

## Cleaned OpenThoughts SFT + OPD With vLLM Evals

The next experiment is the cleaned-data run described in
`results/cleaned_openthoughts_sft_opd_vllm_plan.md`. Launch it with:

```bash
bash examples/qwen3_8b_opd_tillicum/submit_cleaned_sft_opd_vllm_chain.sh
```

The corrected cleaned-SFT eval special-token incident and the auditable
thinking/dual-stop fix are recorded in
`results/cleaned_sft_eval_special_token_incident.md`. Qwen3 generation now
passes `enable_thinking=True` explicitly, accepts both `<|im_end|>` and
`<|endoftext|>` as stops, and saves special-preserving text plus raw prefill and
generated token ids. The same note records the remaining language-cleaning
gap and the optional future `<think>\n` prefill diagnostic.

This chain first validates/installs a scratch-local vLLM eval environment,
cleans OpenThoughts3 by requiring complete thinking traces and removing
mixed-language rows, deterministically shuffles valid rows with seed `1234`,
splits them into `2/3` SFT reserve and `1/3` OPD reserve, trains SFT on the
first 25,000 SFT-reserve rows, trains OPD on the first 1,024 OPD-reserve rows,
then continues OPD on the next 4,096 OPD-reserve rows. Metadata records
`source_row_id`s and proves the selected SFT and OPD rows are disjoint.

The cleaned SFT run uses 4 H200s with `TP=2`, `CP=1`, `DP=2`, Megatron's
distributed optimizer (`SFT_ZERO_STAGE=1` in the legacy wrapper variable),
`SFT_MAX_TOKENS_PER_GPU=16384`, learning rate `1e-6`, and exactly one epoch.
We pivoted away from the Megatron-FSDP/ZeRO-2/3 path after smoke repeatedly hit
integration issues, latest a `fsdp_dtensor` save failure where Megatron expected
a `DTensor` but received a local `Tensor`. The chain now runs a tiny smoke job
for the Megatron distributed-optimizer path only.

The cleaned OPD runs use the known 4-H200 colocate/offload shape:
actor/rollout/teacher `3/3/1`, `TP=1`, `CP=3`, `OPD_SEQ_LENGTH=32766`,
`OPD_MAX_RESPONSE_LEN=31744`, `OPD_MAX_TOKENS_PER_GPU=2048`,
`OPD_LOG_PROBS_CHUNK_SIZE=512`, train memory margin `0`, learning rate `1e-6`,
and one pass over each selected OPD subset. OPD-5k loads from the OPD-1k full
optimizer checkpoint, starts the fresh 4k data pool at offset 0, and keeps the
OPD-1k rows in the final trained-data manifest.

Only four full MATH-500 evals run for this cleaned experiment: base, SFT-25k,
OPD-1k, and OPD-5k. They use the Tillicum-local vLLM path with four independent
1-GPU workers, greedy decoding, `VLLM_EVAL_MAX_MODEL_LEN=32768`,
`VLLM_EVAL_GPU_MEMORY_UTILIZATION=0.92`, `VLLM_EVAL_MAX_NUM_SEQS=16`, and
`VLLM_EVAL_MAX_NUM_BATCHED_TOKENS=131072`.

Both training and eval use dynamic per-prompt generation caps. OPD training
requests `min(31744, OPD_ROLLOUT_MAX_CONTEXT_LEN - prompt_tokens)` with
`OPD_ROLLOUT_MAX_CONTEXT_LEN=32766`; vLLM eval requests
`min(31744, EVAL_MAX_CONTEXT_LEN - prompt_tokens)` with
`EVAL_MAX_CONTEXT_LEN=32768`. Samples record prompt length, requested cap,
effective cap, and whether a clamp happened.

For the 25k continuation experiment, use
`submit_opd_continue_1k_to_25k_val100_chain.sh`. This continues from the full
optimizer checkpoint of the completed corrected 1k OPD run at
`qwen3_8b_sft_25k_opd_1k_32k_sft_colocate4_full_optim/iter_0000007`, runs a
seeded MATH-500 val100 eval on that current checkpoint first, then alternates
1k-ish OPD continuation segments with serialized 4-GPU val100 eval jobs until
`24,960` total OPD samples. The midpoint full MATH-500 eval also uses 4 GPUs
and waits for the `12,544` val100 job before later training resumes, so the
continuation chain never intentionally runs more than one train/eval/report job
at a time and never requests more than 4 H200s for any job. The continuation
data prep extracts the actual first 1,024
trained OPD `source_row_id`s from rollout debug artifacts and generates new OPD
prompts excluding both those rows and all 25k SFT rows. The first continuation
segment intentionally skips loading the old rollout dataset state, because that
state points into the old 10k OPD pool; subsequent segments resume the new
continuation dataset state normally. If a continuation checkpoint already
exists, the submitter can resume at that endpoint by preserving the completed
train checkpoint and submitting only the missing val100 stage.

Continuation OPD training also passes `OPD_ROLLOUT_MAX_CONTEXT_LEN`, defaulting
to `OPD_SEQ_LENGTH`, into rollout generation. Per-sample generation caps are
clamped to the remaining prompt budget under that context limit, so long prompts
avoid SGLang context overflow without reducing the global `OPD_MAX_RESPONSE_LEN`
for shorter prompts.
