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
bash examples/qwen3_8b_opd_tillicum/submit_opd_1k_32k_sft_offload4_chain.sh
```

This chain preserves the completed SFT checkpoint/eval and runs only the
downstream 1k OPD experiment:

- OPD data pool: 10,000 row-disjoint OpenThoughts3 math prompts.
- OPD training horizon: 1,024 effective samples, `8` rollouts of `128`
  prompts each.
- OPD response cap: `31744` generated tokens.
- OPD actor/rollout/teacher GPUs: `6/1/1` on one 8xH200 node, with teacher on
  physical GPU 7.
- OPD actor parallelism: tensor parallel `2`, context parallel `3`,
  `OPD_MAX_TOKENS_PER_GPU=11264`.
- OPD eval: final checkpoint only, stage `opd_001024`.
- Final report: base -> final SFT -> final OPD.

Important: run label `1k_32k` from job `156276` is an accidental but useful
base -> OPD test. That run attempted to pass the final SFT HF snapshot to
Megatron `--load` without the explicit HF-load path, so the actor fell back to
the base torch_dist checkpoint. Do not interpret `1k_32k` as SFT -> OPD.

The corrected 4-GPU chain uses `OPD_INITIAL_LOAD_MODE=hf` and initializes the
actor from the final SFT HF weights snapshot `$SFT_FINAL_HF_DIR`. This is
intentional for the 4-GPU topology: the completed SFT full optimizer checkpoint
was saved with tensor parallel `2`, while the corrected OPD actor uses tensor
parallel `1` and context parallel `2`. Megatron cannot optimizer-resume that SFT
checkpoint across the TP mismatch. After OPD starts, its own full optimizer
checkpoint under `$OPD_SAVE_DIR` is the fidelity-safe continuation point for
more OPD.

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

The older `1k_32k` OPD run is an accidental base -> OPD experiment because the
OPD job was not loaded from the full SFT Megatron checkpoint. If its final eval
times out before Slime writes `debug_eval_0.pt`, use
`submit_cleanup_base_opd_2gpu.sh`. That cleanup job requests 2 H200s for 18
hours, reruns the accidental OPD final eval to completion, runs base eval only
if the current base summary is missing, and writes a base -> OPD-only combined
report with SFT omitted from the comparison.

For future OPD runs, do not treat the current 8-GPU allocation as the preferred
long-term configuration. Use optimizer CPU offload and retune the actor/rollout/
teacher split so training can run with fewer GPUs, targeting the 4-GPU total
training job used by `submit_opd_1k_32k_sft_offload4_chain.sh` if the offload
path is stable enough. This will likely trade wall clock time for lower GPU
occupancy, but it is the right direction for follow-up runs once the
reproduction path is validated.

MATH-500 summaries report `accuracy` with parse failures counted wrong,
`accuracy_on_parseable` as a diagnostic over parseable responses only, and
`parse_failure_rate` separately. Combined reports include a labeled SVG/PNG
curve with light-blue SFT shading and light-purple OPD shading.
