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

export SFT_LOSS_MASK_TYPE="${SFT_LOSS_MASK_TYPE:-qwen}"
export SFT_LOSS_MASK_PREFLIGHT_ENABLED="${SFT_LOSS_MASK_PREFLIGHT_ENABLED:-0}"
export SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK="${SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK:-0}"
export SFT_LOSS_MASK_PREFLIGHT_SAMPLES="${SFT_LOSS_MASK_PREFLIGHT_SAMPLES:-3}"

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
echo "SFT optimizer mode: ${SFT_OPTIMIZER_MODE:-legacy_stage_${SFT_ZERO_STAGE}}"
echo "SFT legacy sharding stage: ${SFT_ZERO_STAGE}"
echo "SFT loss mask type: ${SFT_LOSS_MASK_TYPE}"
echo "SFT loss mask preflight: enabled=${SFT_LOSS_MASK_PREFLIGHT_ENABLED} require_think=${SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK} samples=${SFT_LOSS_MASK_PREFLIGHT_SAMPLES}"
echo "SFT LR: ${SFT_LR}"
echo "SFT max tokens per GPU: ${SFT_MAX_TOKENS_PER_GPU}"
echo "SFT sequence length / gradient clip: ${SFT_SEQ_LENGTH}/${SFT_GRAD_CLIP}"
echo "SFT log probs chunk size: ${SFT_LOG_PROBS_CHUNK_SIZE}"
echo "SFT optimizer CPU offload: ${SFT_OPTIMIZER_CPU_OFFLOAD}"
echo "SFT recompute loss function: ${SFT_RECOMPUTE_LOSS_FUNCTION}"

checkpoint_start_pruner sft "${SFT_SAVE_DIR}" "${CHECKPOINT_PRUNE_INTERVAL_SECONDS}" "${CHECKPOINT_REPORT_DIR}"
cleanup() {
  checkpoint_stop_pruner "${CHECKPOINT_PRUNER_PID:-}"
}
trap cleanup EXIT

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"

if [[ -n "${SFT_TORCH_COMPILE_CC:-}" ]]; then
  [[ -x "${SFT_TORCH_COMPILE_CC}" ]] || { echo "SFT compiler is not executable: ${SFT_TORCH_COMPILE_CC}" >&2; exit 1; }
  export CC="${SFT_TORCH_COMPILE_CC}"
  export CPATH="${SFT_TORCH_COMPILE_CPATH}"
  echo "SFT_TORCH_COMPILER_READY cc=${CC} cpath=${CPATH}"
fi

ray stop --force >/dev/null 2>&1 || true

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o "NV[0-9][0-9]*" | wc -l || true)
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
echo "HAS_NVLINK=${HAS_NVLINK} (detected ${NVLINK_COUNT} NVLink references)"

SFT_MODEL_ARGS_SCRIPT="${SFT_MODEL_ARGS_SCRIPT:-scripts/models/qwen3-8B.sh}"
[[ -f "${SFT_MODEL_ARGS_SCRIPT}" ]] || {
  echo "Missing SFT model argument script: ${SFT_MODEL_ARGS_SCRIPT}" >&2
  exit 1
}
source "${SFT_MODEL_ARGS_SCRIPT}"
if [[ -z "${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON:-}" ]]; then
  SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON="{\"enable_thinking\": true}"
fi
python3 -c "import json, sys; value = json.loads(sys.argv[1]); assert isinstance(value, dict), \"SFT chat template kwargs must be a JSON object\"" \
  "${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON}"
echo "SFT model args script: ${SFT_MODEL_ARGS_SCRIPT}"
echo "SFT chat template kwargs: ${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON}"

if [[ "${SFT_LOSS_MASK_PREFLIGHT_ENABLED}" == "1" ]]; then
  PREFLIGHT_ARGS=(
    --data "${SFT_PARQUET}"
    --hf-checkpoint "${STUDENT_HF_DIR}"
    --loss-mask-type "${SFT_LOSS_MASK_TYPE}"
    --num-samples "${SFT_LOSS_MASK_PREFLIGHT_SAMPLES}"
    --min-mask-raw-ratio 0.80
    --print-snippets
  )
  if [[ "${SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK}" == "1" ]]; then
    PREFLIGHT_ARGS+=(--require-think)
  fi
  python3 examples/qwen3_8b_opd_tillicum/check_sft_loss_mask.py "${PREFLIGHT_ARGS[@]}"
else
  echo "SFT loss-mask preflight disabled."
fi

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

SFT_LOAD_DIR="${SFT_SAVE_DIR}"
SFT_LOAD_KIND="sft_resume"
if [[ -f "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  echo "Resuming SFT from full optimizer checkpoint ${SFT_LOAD_DIR}"
else
  if [[ "${SFT_ZERO_STAGE}" == "2" || "${SFT_ZERO_STAGE}" == "3" ]]; then
    # Retained only for legacy FSDP/ZeRO smoke debugging. The cleaned chain now
    # uses SFT_ZERO_STAGE=1 because fsdp_dtensor saving failed in Megatron.
    SFT_LOAD_DIR="${STUDENT_HF_DIR}"
    SFT_LOAD_KIND="base_hf_weights_initial_legacy_fsdp"
    echo "Starting legacy FSDP SFT from base HF weights ${SFT_LOAD_DIR}"
  else
    SFT_LOAD_DIR="${STUDENT_TORCH_DIST_DIR}"
    SFT_LOAD_KIND="base_megatron_torch_dist_initial"
    echo "Starting SFT from converted Megatron torch_dist base ${SFT_LOAD_DIR}"
  fi
  echo "Fresh SFT optimizer will be initialized because no SFT checkpoint is present."
fi

CKPT_ARGS=(
  --megatron-to-hf-mode bridge
  --hf-checkpoint "${STUDENT_HF_DIR}"
  --ref-load "${SFT_LOAD_DIR}"
  --load "${SFT_LOAD_DIR}"
  --save "${SFT_SAVE_DIR}"
  --save-interval "${SFT_SAVE_INTERVAL}"
  --save-hf "${SFT_HF_SNAPSHOT_TEMPLATE}"
)
echo "SFT load kind: ${SFT_LOAD_KIND}"
echo "SFT load dir: ${SFT_LOAD_DIR}"

DP_OPT_ARGS=()
case "${SFT_ZERO_STAGE}" in
  0)
    DP_OPT_ARGS=(--data-parallel-sharding-strategy no_shard)
    ;;
  1)
    DP_OPT_ARGS=(--use-distributed-optimizer --data-parallel-sharding-strategy optim)
    ;;
  2)
    # Legacy Megatron-FSDP/ZeRO path retained for debugging, but not used by
    # the cleaned chain after fsdp_dtensor checkpoint integration failures.
    DP_OPT_ARGS=(--use-distributed-optimizer --use-megatron-fsdp --data-parallel-sharding-strategy optim_grads)
    ;;
  3)
    # Legacy Megatron-FSDP/ZeRO path retained for debugging, but not used by
    # the cleaned chain after fsdp_dtensor checkpoint integration failures.
    DP_OPT_ARGS=(--use-distributed-optimizer --use-megatron-fsdp --data-parallel-sharding-strategy optim_grads_params)
    ;;
  *)
    echo "SFT_ZERO_STAGE must be 0, 1, 2, or 3; got ${SFT_ZERO_STAGE}" >&2
    exit 1
    ;;
esac
if [[ -n "${SFT_CKPT_FORMAT:-}" ]]; then
  DP_OPT_ARGS+=(--ckpt-format "${SFT_CKPT_FORMAT}")
fi
echo "SFT DP optimizer args: ${DP_OPT_ARGS[*]}"

SFT_ARGS=(
  --rollout-function-path "${SFT_ROLLOUT_FUNCTION_PATH:-slime.rollout.sft_rollout.generate_rollout}"
  --prompt-data "${SFT_PARQUET}"
  --input-key "${SFT_INPUT_KEY:-messages}"
  --rollout-shuffle
  --rollout-seed "${DATA_SEED}"
  --num-epoch "${SFT_NUM_EPOCH}"
  --rollout-batch-size "${SFT_ROLLOUT_BATCH_SIZE}"
  --global-batch-size "${SFT_GLOBAL_BATCH_SIZE}"
  --loss-type sft_loss
  --loss-mask-type "${SFT_LOSS_MASK_TYPE}"
  --apply-chat-template-kwargs "${SFT_APPLY_CHAT_TEMPLATE_KWARGS_JSON}"
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
  --seq-length "${SFT_SEQ_LENGTH}"
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu "${SFT_MAX_TOKENS_PER_GPU}"
  --log-probs-chunk-size "${SFT_LOG_PROBS_CHUNK_SIZE}"
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr "${SFT_LR}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.95
  --clip-grad "${SFT_GRAD_CLIP}"
)
if [[ "${SFT_OPTIMIZER_CPU_OFFLOAD}" == "1" ]]; then
  OPTIMIZER_ARGS+=(
    --optimizer-cpu-offload
    --overlap-cpu-optimizer-d2h-h2d
    --use-precision-aware-optimizer
  )
fi

MEMORY_ARGS=()
if [[ "${SFT_RECOMPUTE_LOSS_FUNCTION}" == "1" ]]; then
  MEMORY_ARGS+=(--recompute-loss-function)
fi

WANDB_ARGS=()
if [[ "${WANDB_MODE}" != "disabled" ]]; then
  WANDB_ARGS+=(
    --wandb-project "${SFT_WANDB_PROJECT:-slime-tillicum}"
    --wandb-group "${SFT_WANDB_GROUP:-qwen3-8b-sft25k}"
  )
fi

MISC_ARGS=(
  --seed "${DATA_SEED}"
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-softmax-in-fp32
  --attention-backend flash
)
case "${SFT_ZERO_STAGE}" in
  2|3)
    echo "SFT FSDP/ZeRO stage ${SFT_ZERO_STAGE}: using BF16 gradient accumulation buffers"
    MISC_ARGS+=(--grad-reduce-in-bf16)
    ;;
  *)
    MISC_ARGS+=(--accumulate-allreduce-grads-in-fp32)
    ;;
esac

export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export no_proxy="127.0.0.1,localhost,${MASTER_ADDR}"
SLIME_SGLANG_PATCH_SITE=0 ray start --head --node-ip-address "${MASTER_ADDR}" --num-gpus "${SFT_ACTOR_GPUS}" \
  --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

if [[ -z "${SFT_CUDA_DEVICE_MAX_CONNECTIONS:-}" ]]; then
  case "${SFT_ZERO_STAGE}" in
    2|3)
      SFT_CUDA_DEVICE_MAX_CONNECTIONS=8
      ;;
    *)
      SFT_CUDA_DEVICE_MAX_CONNECTIONS=1
      ;;
  esac
fi
echo "SFT CUDA_DEVICE_MAX_CONNECTIONS=${SFT_CUDA_DEVICE_MAX_CONNECTIONS}"

COMPILER_RUNTIME_ENV=""
if [[ -n "${CC:-}" ]]; then
  COMPILER_RUNTIME_ENV=",
    \"CC\": \"${CC}\",
    \"CPATH\": \"${CPATH}\""
fi
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM:${SLIME_REPO_ROOT}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"${SFT_CUDA_DEVICE_MAX_CONNECTIONS}\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\",
    \"WANDB_MODE\": \"${WANDB_MODE}\",
    \"HF_HOME\": \"${HF_HOME}\"${COMPILER_RUNTIME_ENV}
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
  "${DP_OPT_ARGS[@]}" \
  "${WANDB_ARGS[@]}" \
  "${PERF_ARGS[@]}" \
  "${MEMORY_ARGS[@]}" \
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
