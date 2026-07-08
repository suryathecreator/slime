#!/bin/bash
#SBATCH --job-name=slime-qwen3-sft-dpopt-smoke
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

echo "Starting SFT Megatron DP optimizer smoke checks on $(hostname) at $(date)"

# Keep the historical directory name so previous smoke artifacts and failure
# logs remain easy to compare. The cleaned chain has pivoted away from the
# Megatron-FSDP/ZeRO-2/3 path after repeated fsdp_dtensor integration failures.
SMOKE_ROOT="${OUTPUT_ROOT}/sft_zero_smoke"
SMOKE_SFT_JSONL="${SMOKE_ROOT}/sft_zero_smoke_000004.jsonl"

if [[ ! -s "${SMOKE_SFT_JSONL}" ]]; then
  if [[ ! -s "${SFT_PARQUET}" ]]; then
    echo "Missing source SFT data for smoke: ${SFT_PARQUET}" >&2
    exit 1
  fi
  head -n 4 "${SFT_PARQUET}" >"${SMOKE_SFT_JSONL}"
fi

smoke_rows="$(wc -l <"${SMOKE_SFT_JSONL}")"
echo "SFT Megatron DP optimizer smoke data: ${SMOKE_SFT_JSONL}"
echo "SFT Megatron DP optimizer smoke data row count: ${smoke_rows}"
if [[ "${smoke_rows}" -ne 4 ]]; then
  echo "Expected exactly 4 smoke rows, got ${smoke_rows}" >&2
  exit 1
fi

# The old stage-2/3 FSDP smoke loop is intentionally disabled for this cleaned
# run. Latest failure reached training, then failed saving fsdp_dtensor with
# AttributeError: 'Tensor' object has no attribute 'to_local'. Stage 1 exercises
# the Megatron distributed optimizer over DP=2 without entering that path.
for stage in 1; do
  echo "Running SFT Megatron distributed optimizer smoke (legacy SFT_ZERO_STAGE=${stage})"
  ckpt_format="torch_dist"

  stage_save_dir="${SMOKE_ROOT}/tiny_stage_${stage}_full_optim"
  stage_hf_dir="${SMOKE_ROOT}/tiny_stage_${stage}_hf"
  stage_final_hf="${stage_hf_dir}/iter_0000000"
  stage_final_ckpt="${stage_save_dir}/iter_0000000"
  if [[ -f "${stage_final_ckpt}/.metadata" && -f "${stage_final_hf}/config.json" ]]; then
    echo "Skipping SFT Megatron DP optimizer smoke; existing complete artifacts found."
    continue
  fi

  env \
    SFT_SIZE=4 \
    SFT_PARQUET="${SMOKE_SFT_JSONL}" \
    SFT_SKIP_LOG_REDIRECT=1 \
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
    SFT_SAVE_DIR="${stage_save_dir}" \
    SFT_HF_SNAPSHOT_DIR="${stage_hf_dir}" \
    SFT_HF_SNAPSHOT_TEMPLATE="${stage_hf_dir}/iter_{rollout_id:07d}" \
    SFT_FINAL_HF_DIR="${stage_final_hf}" \
    SFT_FINAL_FULL_CKPT_DIR="${stage_final_ckpt}" \
    SFT_DETAILS_DIR="${SMOKE_ROOT}/tiny_stage_${stage}_details" \
    CHECKPOINT_REPORT_DIR="${SMOKE_ROOT}/tiny_stage_${stage}_reports" \
    bash "${SCRIPT_DIR}/04_run_sft_100k_8xh200.sbatch"
done

echo "Finished SFT Megatron DP optimizer smoke checks at $(date)"
