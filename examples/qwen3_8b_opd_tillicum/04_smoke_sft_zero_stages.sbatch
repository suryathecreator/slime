#!/bin/bash
#SBATCH --job-name=slime-qwen3-sft-zero-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"

mkdir -p "${SLURM_LOG_DIR}" "${OUTPUT_ROOT}/sft_zero_smoke"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Starting SFT ZeRO smoke checks on $(hostname) at $(date)"

for stage in 1 2 3; do
  echo "Running SFT ZeRO stage ${stage} smoke"
  ckpt_format="torch_dist"
  if [[ "${stage}" == "3" ]]; then
    ckpt_format="fsdp_dtensor"
  fi
  env \
    SFT_SIZE=4 \
    SFT_NUM_EPOCH=1 \
    SFT_ROLLOUT_BATCH_SIZE=4 \
    SFT_GLOBAL_BATCH_SIZE=4 \
    SFT_NUM_ROLLOUT=1 \
    SFT_FINAL_ROLLOUT_ID=0 \
    SFT_SAVE_INTERVAL=1 \
    SFT_ACTOR_GPUS=4 \
    SFT_TENSOR_MODEL_PARALLEL_SIZE=2 \
    SFT_CONTEXT_PARALLEL_SIZE=1 \
    SFT_PIPELINE_MODEL_PARALLEL_SIZE=1 \
    SFT_ZERO_STAGE="${stage}" \
    SFT_CKPT_FORMAT="${ckpt_format}" \
    SFT_LR=1e-6 \
    SFT_SAVE_DIR="${OUTPUT_ROOT}/sft_zero_smoke/stage_${stage}_full_optim" \
    SFT_HF_SNAPSHOT_DIR="${OUTPUT_ROOT}/sft_zero_smoke/stage_${stage}_hf" \
    SFT_HF_SNAPSHOT_TEMPLATE="${OUTPUT_ROOT}/sft_zero_smoke/stage_${stage}_hf/iter_{rollout_id:07d}" \
    SFT_FINAL_HF_DIR="${OUTPUT_ROOT}/sft_zero_smoke/stage_${stage}_hf/iter_0000000" \
    SFT_FINAL_FULL_CKPT_DIR="${OUTPUT_ROOT}/sft_zero_smoke/stage_${stage}_full_optim/iter_0000000" \
    SFT_DETAILS_DIR="${OUTPUT_ROOT}/sft_zero_smoke/stage_${stage}_details" \
    CHECKPOINT_REPORT_DIR="${OUTPUT_ROOT}/sft_zero_smoke/stage_${stage}_reports" \
    bash "${SCRIPT_DIR}/04_run_sft_100k_8xh200.sbatch"
done

echo "Finished SFT ZeRO smoke checks at $(date)"
