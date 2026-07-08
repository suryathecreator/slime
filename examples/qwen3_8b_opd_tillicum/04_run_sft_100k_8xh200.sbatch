#!/bin/bash
#SBATCH --job-name=slime-qwen3-sft25k
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/checkpoint_utils.sh"

mkdir -p "${SLURM_LOG_DIR}" "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${SFT_DETAILS_DIR}" "${CHECKPOINT_REPORT_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export RAY_TMPDIR="/tmp/${USER:-suryadv}/slime-ray-${SLURM_JOB_ID}"
  mkdir -p "${RAY_TMPDIR}"
  if [[ "${SFT_SKIP_LOG_REDIRECT:-0}" != "1" ]]; then
    exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
  fi
fi

for required_path in "${SFT_PARQUET}" "${STUDENT_HF_DIR}" "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path: ${required_path}" >&2
    exit 1
  fi
done

echo "Starting SFT on $(hostname) at $(date)"
echo "SFT data: ${SFT_PARQUET}"
echo "Save dir: ${SFT_SAVE_DIR}"
echo "HF eval snapshots: ${SFT_HF_SNAPSHOT_DIR}"
echo "Expected final SFT rollout id: ${SFT_FINAL_ROLLOUT_ID}"
echo "SFT actor GPUs: ${SFT_ACTOR_GPUS}"
echo "SFT TP/CP/PP: ${SFT_TENSOR_MODEL_PARALLEL_SIZE}/${SFT_CONTEXT_PARALLEL_SIZE}/${SFT_PIPELINE_MODEL_PARALLEL_SIZE}"
echo "SFT ZeRO stage: ${SFT_ZERO_STAGE}"
echo "SFT LR: ${SFT_LR}"
echo "SFT max tokens per GPU: ${SFT_MAX_TOKENS_PER_GPU}"

checkpoint_start_pruner sft "${SFT_SAVE_DIR}" "${CHECKPOINT_PRUNE_INTERVAL_SECONDS}" "${CHECKPOINT_REPORT_DIR}"
cleanup() {
  checkpoint_stop_pruner "${CHECKPOINT_PRUNER_PID:-}"
}
trap cleanup EXIT

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"

ray stop --force >/dev/null 2>&1 || true

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

source scripts/models/qwen3-8B.sh

SFT_DP=$((SFT_ACTOR_GPUS / (SFT_TENSOR_MODEL_PARALLEL_SIZE * SFT_CONTEXT_PARALLEL_SIZE * SFT_PIPELINE_MODEL_PARALLEL_SIZE)))
if [[ "${SFT_DP}" -lt 1 ]]; then
  echo "Invalid SFT DP=${SFT_DP}; check actor GPUs and TP/CP/PP." >&2
  exit 1
fi
if [[ "${SFT_ZERO_STAGE}" != "0" && "${SFT_DP}" -lt 2 ]]; then
  echo "SFT_ZERO_STAGE=${SFT_ZERO_STAGE} requires DP >= 2, got DP=${SFT_DP}" >&2
  exit 1
fi
echo "Resolved SFT DP=${SFT_DP}"

CKPT_ARGS=(
  --hf-checkpoint "${STUDENT_HF_DIR}"
  --ref-load "${STUDENT_TORCH_DIST_DIR}"
  --load "${SFT_SAVE_DIR}"
  --save "${SFT_SAVE_DIR}"
  --save-interval "${SFT_SAVE_INTERVAL}"
  --save-hf "${SFT_HF_SNAPSHOT_TEMPLATE}"
)

ZERO_ARGS=()
case "${SFT_ZERO_STAGE}" in
  0)
    ZERO_ARGS=(--data-parallel-sharding-strategy no_shard)
    ;;
  1)
    ZERO_ARGS=(--use-distributed-optimizer --data-parallel-sharding-strategy optim)
    ;;
  2)
    ZERO_ARGS=(--use-distributed-optimizer --use-megatron-fsdp --data-parallel-sharding-strategy optim_grads)
    ;;
  3)
    ZERO_ARGS=(--use-distributed-optimizer --use-megatron-fsdp --data-parallel-sharding-strategy optim_grads_params)
    ;;
  *)
    echo "SFT_ZERO_STAGE must be 0, 1, 2, or 3; got ${SFT_ZERO_STAGE}" >&2
    exit 1
    ;;
esac
if [[ -n "${SFT_CKPT_FORMAT:-}" ]]; then
  ZERO_ARGS+=(--ckpt-format "${SFT_CKPT_FORMAT}")
fi
echo "SFT ZeRO args: ${ZERO_ARGS[*]}"

SFT_ARGS=(
  --rollout-function-path slime.rollout.sft_rollout.generate_rollout
  --prompt-data "${SFT_PARQUET}"
  --input-key messages
  --rollout-shuffle
  --num-epoch "${SFT_NUM_EPOCH}"
  --rollout-batch-size "${SFT_ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${SFT_GLOBAL_BATCH_SIZE}"
  --loss-type sft_loss
  --calculate-per-token-loss
  --disable-compute-advantages-and-returns
  --debug-train-only
  --dump-details "${SFT_DETAILS_DIR}"
)

PERF_ARGS=(
  --tensor-model-parallel-size "${SFT_TENSOR_MODEL_PARALLEL_SIZE}"
  --sequence-parallel
  --pipeline-model-parallel-size "${SFT_PIPELINE_MODEL_PARALLEL_SIZE}"
  --context-parallel-size "${SFT_CONTEXT_PARALLEL_SIZE}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${SFT_MAX_TOKENS_PER_GPU}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${SFT_LR}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
)

WANDB_ARGS=()
if [[ "${WANDB_MODE}" != "disabled" ]]; then
  WANDB_ARGS+=(--wandb-project slime-tillicum --wandb-group qwen3-8b-sft25k)
fi

MISC_ARGS=(
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"
SLIME_SGLANG_PATCH_SITE=0 ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${SFT_ACTOR_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${SLIME_REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"WANDB_MODE\": \"${WANDB_MODE}\",
    \"HF_HOME\": \"${HF_HOME}\"
  }
}"

trap "ray stop --force >/dev/null 2>&1 || true" EXIT

SLIME_SGLANG_PATCH_SITE=0 ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train_async.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node "${SFT_ACTOR_GPUS}" \
  "${MODEL_ARGS[@]}" \
  "${CKPT_ARGS[@]}" \
  "${SFT_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${ZERO_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MISC_ARGS[@]}"
'

checkpoint_stop_pruner "${CHECKPOINT_PRUNER_PID:-}"
checkpoint_prune_old sft "${SFT_SAVE_DIR}" final
checkpoint_report sft "${SFT_SAVE_DIR}" "${CHECKPOINT_REPORT_DIR}"
checkpoint_verify_single_latest sft "${SFT_SAVE_DIR}"

if [[ -f "${SPLIT_METADATA}" ]]; then
  cp -f "${SPLIT_METADATA}" "${SFT_SAVE_DIR}/split_metadata.json"
fi

if [[ ! -d "${SFT_FINAL_HF_DIR}" ]]; then
  echo "Missing final SFT HF eval snapshot: ${SFT_FINAL_HF_DIR}" >&2
  exit 1
fi

echo "Finished SFT at $(date)"
