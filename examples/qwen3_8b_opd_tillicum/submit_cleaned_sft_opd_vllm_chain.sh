#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export CLEANED_EXPERIMENT_LABEL="${CLEANED_EXPERIMENT_LABEL:-cleaned_think_sft25k_opd5k_vllm}"
export CLEANED_OPD_FIRST_SIZE="${CLEANED_OPD_FIRST_SIZE:-1024}"
export CLEANED_OPD_NEXT_SIZE="${CLEANED_OPD_NEXT_SIZE:-4096}"
export CLEANED_RESUME_AFTER_CONVERT="${CLEANED_RESUME_AFTER_CONVERT:-0}"
export CLEANED_RESUME_AFTER_BASE_EVAL="${CLEANED_RESUME_AFTER_BASE_EVAL:-0}"
export CLEANED_RESUME_AFTER_SFT="${CLEANED_RESUME_AFTER_SFT:-0}"
export CLEANED_OUTPUT_TAG="${CLEANED_OUTPUT_TAG:-qwen3mask}"
export CLEANED_WALLTIME_VLLM_SETUP="${CLEANED_WALLTIME_VLLM_SETUP:-06:00:00}"
export CLEANED_WALLTIME_CLEAN_DATA="${CLEANED_WALLTIME_CLEAN_DATA:-24:00:00}"
export CLEANED_WALLTIME_CONVERT="${CLEANED_WALLTIME_CONVERT:-06:00:00}"
export CLEANED_WALLTIME_SMOKE="${CLEANED_WALLTIME_SMOKE:-06:00:00}"
export CLEANED_WALLTIME_BASE_EVAL="${CLEANED_WALLTIME_BASE_EVAL:-72:00:00}"
export CLEANED_WALLTIME_SFT="${CLEANED_WALLTIME_SFT:-72:00:00}"
export CLEANED_WALLTIME_SFT_EVAL="${CLEANED_WALLTIME_SFT_EVAL:-72:00:00}"
export CLEANED_WALLTIME_OPD1="${CLEANED_WALLTIME_OPD1:-72:00:00}"
export CLEANED_WALLTIME_OPD1_EVAL="${CLEANED_WALLTIME_OPD1_EVAL:-72:00:00}"
export CLEANED_WALLTIME_OPD5="${CLEANED_WALLTIME_OPD5:-72:00:00}"
export CLEANED_WALLTIME_OPD5_EVAL="${CLEANED_WALLTIME_OPD5_EVAL:-72:00:00}"
export CLEANED_WALLTIME_REPORT="${CLEANED_WALLTIME_REPORT:-06:00:00}"
export SFT_SIZE="${SFT_SIZE:-25000}"
export SFT_ROLLOUT_BATCH_SIZE="${SFT_ROLLOUT_BATCH_SIZE:-250}"
export SFT_GLOBAL_BATCH_SIZE="${SFT_GLOBAL_BATCH_SIZE:-250}"
export SFT_NUM_ROLLOUT="${SFT_NUM_ROLLOUT:-100}"
export SFT_FINAL_ROLLOUT_ID="${SFT_FINAL_ROLLOUT_ID:-99}"
export SFT_MILESTONE_ROLLOUT_IDS="${SFT_MILESTONE_ROLLOUT_IDS:-19 39 59 79 99}"
export SFT_SAVE_INTERVAL="${SFT_SAVE_INTERVAL:-20}"
export SFT_ACTOR_GPUS="${SFT_ACTOR_GPUS:-4}"
export SFT_TENSOR_MODEL_PARALLEL_SIZE="${SFT_TENSOR_MODEL_PARALLEL_SIZE:-2}"
export SFT_CONTEXT_PARALLEL_SIZE="${SFT_CONTEXT_PARALLEL_SIZE:-1}"
export SFT_PIPELINE_MODEL_PARALLEL_SIZE="${SFT_PIPELINE_MODEL_PARALLEL_SIZE:-1}"
export SFT_OPTIMIZER_MODE="${SFT_OPTIMIZER_MODE:-megatron_distributed_optimizer}"
# The variable name is kept for compatibility with the shared SFT wrapper.
# Stage 1 here means Megatron's distributed optimizer over DP=2; the cleaned
# chain no longer uses the Megatron-FSDP/ZeRO-2/3 path after repeated
# fsdp_dtensor integration failures during smoke.
export SFT_ZERO_STAGE="${SFT_ZERO_STAGE:-1}"
export SFT_CKPT_FORMAT="${SFT_CKPT_FORMAT:-torch_dist}"
export SFT_LOSS_MASK_TYPE="${SFT_LOSS_MASK_TYPE:-qwen3}"
export SFT_LOSS_MASK_PREFLIGHT_ENABLED="${SFT_LOSS_MASK_PREFLIGHT_ENABLED:-1}"
export SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK="${SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK:-1}"
export SFT_LOSS_MASK_PREFLIGHT_SAMPLES="${SFT_LOSS_MASK_PREFLIGHT_SAMPLES:-3}"
export SFT_MAX_TOKENS_PER_GPU="${SFT_MAX_TOKENS_PER_GPU:-16384}"
export SFT_LR="${SFT_LR:-1e-6}"
export OPD_ACTOR_GPUS="${OPD_ACTOR_GPUS:-3}"
export OPD_ROLLOUT_GPUS="${OPD_ROLLOUT_GPUS:-3}"
export OPD_RAY_GPUS="${OPD_RAY_GPUS:-3}"
export OPD_TEACHER_GPU="${OPD_TEACHER_GPU:-3}"
export OPD_TENSOR_MODEL_PARALLEL_SIZE="${OPD_TENSOR_MODEL_PARALLEL_SIZE:-1}"
export OPD_CONTEXT_PARALLEL_SIZE="${OPD_CONTEXT_PARALLEL_SIZE:-3}"
export OPD_SEQ_LENGTH="${OPD_SEQ_LENGTH:-32766}"
export OPD_ROLLOUT_MAX_CONTEXT_LEN="${OPD_ROLLOUT_MAX_CONTEXT_LEN:-32766}"
export OPD_MAX_RESPONSE_LEN="${OPD_MAX_RESPONSE_LEN:-31744}"
export OPD_MAX_TOKENS_PER_GPU="${OPD_MAX_TOKENS_PER_GPU:-2048}"
export OPD_LOG_PROBS_CHUNK_SIZE="${OPD_LOG_PROBS_CHUNK_SIZE:-512}"
export OPD_TRAIN_MEMORY_MARGIN_BYTES="${OPD_TRAIN_MEMORY_MARGIN_BYTES:-0}"
export OPD_LR="${OPD_LR:-1e-6}"
export OPD_COLOCATE="${OPD_COLOCATE:-1}"
export OPD_OFFLOAD_TRAIN="${OPD_OFFLOAD_TRAIN:-1}"
export OPD_OFFLOAD_ROLLOUT="${OPD_OFFLOAD_ROLLOUT:-1}"
export OPD_OPTIMIZER_CPU_OFFLOAD="${OPD_OPTIMIZER_CPU_OFFLOAD:-1}"
export OPD_RECOMPUTE_LOSS_FUNCTION="${OPD_RECOMPUTE_LOSS_FUNCTION:-1}"
export OPD_SANITY_FAIL_ON_COLLAPSE="${OPD_SANITY_FAIL_ON_COLLAPSE:-0}"
export OPD_MANIFEST_FROM_ROLLOUT_LOGS="${OPD_MANIFEST_FROM_ROLLOUT_LOGS:-0}"
export EVAL_BACKEND="${EVAL_BACKEND:-vllm}"
export EVAL_MAX_RESPONSE_LEN="${EVAL_MAX_RESPONSE_LEN:-31744}"
export EVAL_MAX_CONTEXT_LEN="${EVAL_MAX_CONTEXT_LEN:-32768}"
export EVAL_EXPECTED_SAMPLES="${EVAL_EXPECTED_SAMPLES:-500}"
export VLLM_EVAL_NUM_GPUS="${VLLM_EVAL_NUM_GPUS:-4}"
export VLLM_EVAL_MAX_MODEL_LEN="${VLLM_EVAL_MAX_MODEL_LEN:-32768}"
export VLLM_EVAL_GPU_MEMORY_UTILIZATION="${VLLM_EVAL_GPU_MEMORY_UTILIZATION:-0.92}"
export VLLM_EVAL_MAX_NUM_SEQS="${VLLM_EVAL_MAX_NUM_SEQS:-16}"
export VLLM_EVAL_MAX_NUM_BATCHED_TOKENS="${VLLM_EVAL_MAX_NUM_BATCHED_TOKENS:-131072}"
export VLLM_EVAL_CHUNK_SIZE="${VLLM_EVAL_CHUNK_SIZE:-16}"
export VLLM_EVAL_RESUME_COMPLETED="${VLLM_EVAL_RESUME_COMPLETED:-1}"
export REPORT_SFT_FINAL_ONLY="${REPORT_SFT_FINAL_ONLY:-1}"
export REPORT_OPD_FINAL_ONLY="${REPORT_OPD_FINAL_ONLY:-0}"
export REPORT_INCLUDE_SFT="${REPORT_INCLUDE_SFT:-1}"
export REPORT_EXPERIMENT_NOTE="${REPORT_EXPERIMENT_NOTE:-Cleaned OpenThoughts SFT OPD vLLM final eval run with Qwen3 loss masking}"

source "${SCRIPT_DIR}/env.sh"

export SFT_PARQUET="${DATA_ROOT}/openthoughts3_cleaned_${CLEANED_EXPERIMENT_LABEL}_sft_25000.jsonl"
export CLEANED_METADATA="${DATA_ROOT}/openthoughts3_cleaned_${CLEANED_EXPERIMENT_LABEL}_metadata.json"
export SPLIT_METADATA="${CLEANED_METADATA}"
export CLEANED_OPD_RESERVE_JSONL="${DATA_ROOT}/openthoughts3_cleaned_${CLEANED_EXPERIMENT_LABEL}_opd_reserve.jsonl"
export CLEANED_OPD_1K_JSONL="${DATA_ROOT}/openthoughts3_cleaned_${CLEANED_EXPERIMENT_LABEL}_opd_001024.jsonl"
export CLEANED_OPD_4K_JSONL="${DATA_ROOT}/openthoughts3_cleaned_${CLEANED_EXPERIMENT_LABEL}_opd_next_004096.jsonl"

export SFT_SAVE_DIR="${OUTPUT_ROOT}/qwen3_8b_cleaned_sft_25k_${CLEANED_OUTPUT_TAG}_full_optim"
export SFT_HF_SNAPSHOT_DIR="${OUTPUT_ROOT}/qwen3_8b_cleaned_sft_25k_${CLEANED_OUTPUT_TAG}_eval_snapshots"
export SFT_DETAILS_DIR="${OUTPUT_ROOT}/cleaned_sft_25k_${CLEANED_OUTPUT_TAG}_details"
export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000099"
export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/iter_0000099"
export SFT_EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/math500_eval_cleaned_sft_25k_${CLEANED_OUTPUT_TAG}_vllm"
export BASE_EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/math500_eval_cleaned_base_vllm"
export OPD_EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/math500_eval_cleaned_${CLEANED_OUTPUT_TAG}_opd_1k_5k_vllm"
export COMBINED_EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/math500_eval_cleaned_${CLEANED_OUTPUT_TAG}_combined_vllm"
export CHECKPOINT_REPORT_DIR="${OUTPUT_ROOT}/checkpoint_reports_${CLEANED_EXPERIMENT_LABEL}_${CLEANED_OUTPUT_TAG}"
export REPORT_OPD_X_OFFSET_SAMPLES="25000"
export REPORT_OPD_LABEL_PREFIX="SFT+"

OPD1_RUN_LABEL="cleaned_1k_32k_sft_colocate4"
OPD5_RUN_LABEL="cleaned_5k_32k_sft_colocate4"
OPD1_SAVE_DIR="${OUTPUT_ROOT}/qwen3_8b_cleaned_sft_25k_${CLEANED_OUTPUT_TAG}_opd_1k_32k_full_optim"
OPD1_HF_SNAPSHOT_DIR="${OUTPUT_ROOT}/qwen3_8b_cleaned_sft_25k_${CLEANED_OUTPUT_TAG}_opd_1k_32k_eval_snapshots"
OPD1_ROLLOUT_LOG_DIR="${OUTPUT_ROOT}/opd_${OPD1_RUN_LABEL}_rollout_logs"
OPD1_SANITY_DIR="${OUTPUT_ROOT}/opd_${OPD1_RUN_LABEL}_sanity"
OPD1_FINAL_HF_DIR="${OPD1_HF_SNAPSHOT_DIR}/iter_0000007"
OPD5_SAVE_DIR="${OUTPUT_ROOT}/qwen3_8b_cleaned_sft_25k_${CLEANED_OUTPUT_TAG}_opd_5k_32k_full_optim"
OPD5_HF_SNAPSHOT_DIR="${OUTPUT_ROOT}/qwen3_8b_cleaned_sft_25k_${CLEANED_OUTPUT_TAG}_opd_5k_32k_eval_snapshots"
OPD5_ROLLOUT_LOG_DIR="${OUTPUT_ROOT}/opd_${OPD5_RUN_LABEL}_rollout_logs"
OPD5_SANITY_DIR="${OUTPUT_ROOT}/opd_${OPD5_RUN_LABEL}_sanity"
OPD5_FINAL_HF_DIR="${OPD5_HF_SNAPSHOT_DIR}/iter_0000039"

cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

SBATCH_ONE=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:1)
SBATCH_FOUR=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:4)
SBATCH_REPORT=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:1)

submit_log="${SLURM_LOG_DIR}/submit_cleaned_sft_opd_vllm_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitting cleaned OpenThoughts SFT + OPD vLLM chain"
echo "submit log: ${submit_log}"
echo "cleaned label: ${CLEANED_EXPERIMENT_LABEL}"
echo "resume after convert: ${CLEANED_RESUME_AFTER_CONVERT}"
echo "resume after base eval: ${CLEANED_RESUME_AFTER_BASE_EVAL}"
echo "resume after sft: ${CLEANED_RESUME_AFTER_SFT}"
echo "output tag: ${CLEANED_OUTPUT_TAG}"
echo "SFT data: ${SFT_PARQUET}"
echo "OPD 1k data: ${CLEANED_OPD_1K_JSONL}"
echo "OPD next 4k data: ${CLEANED_OPD_4K_JSONL}"
echo "SFT TP/CP/DP/optimizer/loss-mask: ${SFT_TENSOR_MODEL_PARALLEL_SIZE}/${SFT_CONTEXT_PARALLEL_SIZE}/2/${SFT_OPTIMIZER_MODE}/${SFT_LOSS_MASK_TYPE}"
echo "SFT loss-mask preflight: enabled=${SFT_LOSS_MASK_PREFLIGHT_ENABLED} require_think=${SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK} samples=${SFT_LOSS_MASK_PREFLIGHT_SAMPLES}"
echo "SFT max tokens/GPU LR epochs: ${SFT_MAX_TOKENS_PER_GPU} ${SFT_LR} ${SFT_NUM_EPOCH}"
echo "OPD TP/CP max-response/context max-tokens/GPU chunk LR: ${OPD_TENSOR_MODEL_PARALLEL_SIZE}/${OPD_CONTEXT_PARALLEL_SIZE} ${OPD_MAX_RESPONSE_LEN}/${OPD_ROLLOUT_MAX_CONTEXT_LEN} ${OPD_MAX_TOKENS_PER_GPU} ${OPD_LOG_PROBS_CHUNK_SIZE} ${OPD_LR}"
echo "vLLM eval workers/max-model/max-seqs/batched-tokens: ${VLLM_EVAL_NUM_GPUS}/${VLLM_EVAL_MAX_MODEL_LEN}/${VLLM_EVAL_MAX_NUM_SEQS}/${VLLM_EVAL_MAX_NUM_BATCHED_TOKENS}"
echo "vLLM eval chunk resume: chunk_size=${VLLM_EVAL_CHUNK_SIZE} resume_completed=${VLLM_EVAL_RESUME_COMPLETED}"
echo "walltimes setup/data/convert/smoke/base/sft/sft_eval/opd1/opd1_eval/opd5/opd5_eval/report: ${CLEANED_WALLTIME_VLLM_SETUP} ${CLEANED_WALLTIME_CLEAN_DATA} ${CLEANED_WALLTIME_CONVERT} ${CLEANED_WALLTIME_SMOKE} ${CLEANED_WALLTIME_BASE_EVAL} ${CLEANED_WALLTIME_SFT} ${CLEANED_WALLTIME_SFT_EVAL} ${CLEANED_WALLTIME_OPD1} ${CLEANED_WALLTIME_OPD1_EVAL} ${CLEANED_WALLTIME_OPD5} ${CLEANED_WALLTIME_OPD5_EVAL} ${CLEANED_WALLTIME_REPORT}"

common_required_paths=(
  "${SFT_PARQUET}"
  "${CLEANED_METADATA}"
  "${CLEANED_OPD_RESERVE_JSONL}"
  "${CLEANED_OPD_1K_JSONL}"
  "${CLEANED_OPD_4K_JSONL}"
  "${STUDENT_HF_DIR}"
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt"
)

sft_dependency_args=()
submit_sft=1
if [[ "${CLEANED_RESUME_AFTER_SFT}" == "1" ]]; then
  echo "Reusing completed vLLM setup, cleaned data, model conversion, smoke, base eval, and corrected SFT artifacts."
  for required_path in \
    "${common_required_paths[@]}" \
    "${BASE_EVAL_OUTPUT_DIR}/base/summary.json" \
    "${SFT_FINAL_HF_DIR}/config.json" \
    "${SFT_FINAL_FULL_CKPT_DIR}/.metadata"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Cannot resume after SFT; missing required artifact: ${required_path}" >&2
      exit 1
    fi
  done
  jid_vllm="preserved_163642"
  jid_data="preserved_163643"
  jid_convert="preserved_163644"
  jid_sft_smoke="preserved_165034"
  jid_base_eval="preserved_165035"
  jid_sft="preserved_165695"
  submit_sft=0
elif [[ "${CLEANED_RESUME_AFTER_BASE_EVAL}" == "1" ]]; then
  echo "Reusing completed vLLM setup, cleaned data, model conversion, smoke, and base eval artifacts."
  for required_path in \
    "${common_required_paths[@]}" \
    "${BASE_EVAL_OUTPUT_DIR}/base/summary.json"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Cannot resume after base eval; missing required artifact: ${required_path}" >&2
      exit 1
    fi
  done
  jid_vllm="preserved_163642"
  jid_data="preserved_163643"
  jid_convert="preserved_163644"
  jid_sft_smoke="preserved_165034"
  jid_base_eval="preserved_165035"
else
  smoke_dependency_args=()
  if [[ "${CLEANED_RESUME_AFTER_CONVERT}" == "1" ]]; then
    echo "Reusing completed vLLM setup, cleaned data, and model conversion artifacts."
    for required_path in "${common_required_paths[@]}"; do
      if [[ ! -e "${required_path}" ]]; then
        echo "Cannot resume after convert; missing required artifact: ${required_path}" >&2
        exit 1
      fi
    done
    jid_vllm="preserved_163642"
    jid_data="preserved_163643"
    jid_convert="preserved_163644"
  elif [[ "${CLEANED_RESUME_AFTER_CONVERT}" == "0" ]]; then
    jid_vllm="$(
      sbatch --parsable "${SBATCH_ONE[@]}" \
        --time="${CLEANED_WALLTIME_VLLM_SETUP}" \
        --job-name=slime-qwen3-clean-vllm-setup \
        --cpus-per-task=8 \
        --export=ALL \
        examples/qwen3_8b_opd_tillicum/10_setup_vllm_eval_env.sbatch
    )"
    jid_data="$(
      sbatch --parsable "${SBATCH_ONE[@]}" \
        --dependency=afterok:${jid_vllm} \
        --time="${CLEANED_WALLTIME_CLEAN_DATA}" \
        --job-name=slime-qwen3-clean-data \
        --cpus-per-task=8 \
        --export=ALL \
        examples/qwen3_8b_opd_tillicum/02_prepare_cleaned_data.sbatch
    )"
    jid_convert="$(
      sbatch --parsable "${SBATCH_FOUR[@]}" \
        --dependency=afterok:${jid_data} \
        --time="${CLEANED_WALLTIME_CONVERT}" \
        --job-name=slime-qwen3-clean-convert \
        --cpus-per-task=32 \
        --export=ALL,CONVERT_NPROC=4 \
        examples/qwen3_8b_opd_tillicum/03_convert_models_if_needed.sbatch
    )"
    smoke_dependency_args=(--dependency=afterok:${jid_convert})
  else
    echo "CLEANED_RESUME_AFTER_CONVERT must be 0 or 1; got ${CLEANED_RESUME_AFTER_CONVERT}" >&2
    exit 1
  fi
  jid_sft_smoke="$(
    sbatch --parsable "${SBATCH_FOUR[@]}" \
      "${smoke_dependency_args[@]}" \
      --time="${CLEANED_WALLTIME_SMOKE}" \
      --job-name=slime-qwen3-sft-dpopt-smoke \
      --cpus-per-task=32 \
      --export=ALL \
      examples/qwen3_8b_opd_tillicum/04_smoke_sft_zero_stages.sbatch
  )"
  jid_base_eval="$(
    sbatch --parsable "${SBATCH_FOUR[@]}" \
      --dependency=afterok:${jid_sft_smoke} \
      --time="${CLEANED_WALLTIME_BASE_EVAL}" \
      --job-name=slime-qwen3-clean-base-vllm \
      --cpus-per-task=32 \
      --export=ALL,EVAL_TARGETS=base,EVAL_SKIP_COMPLETED=1 \
      examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch
  )"
  sft_dependency_args=(--dependency=afterok:${jid_base_eval})
fi
if [[ "${submit_sft}" == "1" ]]; then
  jid_sft="$(
    sbatch --parsable "${SBATCH_FOUR[@]}" \
      "${sft_dependency_args[@]}" \
      --time="${CLEANED_WALLTIME_SFT}" \
      --job-name=slime-qwen3-clean-sft25k \
      --cpus-per-task=32 \
      --export=ALL \
      examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch
  )"
  sft_eval_dependency_args=(--dependency=afterok:${jid_sft})
else
  sft_eval_dependency_args=()
fi
jid_sft_eval="$(
  sbatch --parsable "${SBATCH_FOUR[@]}" \
    "${sft_eval_dependency_args[@]}" \
    --time="${CLEANED_WALLTIME_SFT_EVAL}" \
    --job-name=slime-qwen3-clean-sft-vllm \
    --cpus-per-task=32 \
    --export=ALL,EVAL_TARGETS=sft,EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch
)"

export OPD_RUN_LABEL="${OPD1_RUN_LABEL}"
export OPD_JSONL="${CLEANED_OPD_1K_JSONL}"
export OPD_POOL_SIZE="${CLEANED_OPD_FIRST_SIZE}"
export OPD_SIZE="${CLEANED_OPD_FIRST_SIZE}"
export OPD_TRAIN_SIZE="${CLEANED_OPD_FIRST_SIZE}"
export OPD_NUM_ROLLOUT="8"
export OPD_FINAL_ROLLOUT_ID="7"
export OPD_EFFECTIVE_TRAIN_SAMPLES="${CLEANED_OPD_FIRST_SIZE}"
export OPD_MILESTONE_ROLLOUT_IDS="7"
export OPD_SAVE_INTERVAL="8"
export OPD_SAVE_DIR="${OPD1_SAVE_DIR}"
export OPD_HF_SNAPSHOT_DIR="${OPD1_HF_SNAPSHOT_DIR}"
export OPD_HF_SNAPSHOT_TEMPLATE="${OPD1_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export OPD_FINAL_HF_DIR="${OPD1_FINAL_HF_DIR}"
export OPD_ROLLOUT_LOG_DIR="${OPD1_ROLLOUT_LOG_DIR}"
export OPD_SANITY_REPORT_DIR="${OPD1_SANITY_DIR}"
export OPD_SANITY_SUMMARY_DIR="${OPD1_SANITY_DIR}"
export OPD_TRAINED_MANIFEST="${OPD1_SAVE_DIR}/opd_trained_manifest.json"
export OPD_INITIAL_LOAD_MODE="hf"
export OPD_INITIAL_LOAD_DIR="${SFT_FINAL_HF_DIR}"
export OPD_REF_LOAD_DIR="${SFT_FINAL_HF_DIR}"
export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD="0"
export OPD_ALREADY_TRAINED_SAMPLES="0"
export OPD_START_ROLLOUT_ID="0"
export OPD_MANIFEST_FROM_ROLLOUT_LOGS="0"
export OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH="0"

jid_opd1="$(
  sbatch --parsable "${SBATCH_FOUR[@]}" \
    --dependency=afterok:${jid_sft_eval} \
    --time="${CLEANED_WALLTIME_OPD1}" \
    --job-name=slime-qwen3-clean-opd1k \
    --cpus-per-task=32 \
    --export=ALL \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
)"
jid_opd1_eval="$(
  sbatch --parsable "${SBATCH_FOUR[@]}" \
    --dependency=afterok:${jid_opd1} \
    --time="${CLEANED_WALLTIME_OPD1_EVAL}" \
    --job-name=slime-qwen3-clean-opd1k-vllm \
    --cpus-per-task=32 \
    --export=ALL,EVAL_TARGETS=opd,EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch
)"

export OPD_RUN_LABEL="${OPD5_RUN_LABEL}"
export OPD_JSONL="${CLEANED_OPD_4K_JSONL}"
export OPD_POOL_SIZE="${CLEANED_OPD_NEXT_SIZE}"
export OPD_SIZE="${CLEANED_OPD_NEXT_SIZE}"
export OPD_TRAIN_SIZE="5120"
export OPD_NUM_ROLLOUT="40"
export OPD_FINAL_ROLLOUT_ID="39"
export OPD_EFFECTIVE_TRAIN_SAMPLES="5120"
export OPD_MILESTONE_ROLLOUT_IDS="39"
export OPD_SAVE_INTERVAL="8"
export OPD_SAVE_DIR="${OPD5_SAVE_DIR}"
export OPD_HF_SNAPSHOT_DIR="${OPD5_HF_SNAPSHOT_DIR}"
export OPD_HF_SNAPSHOT_TEMPLATE="${OPD5_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export OPD_FINAL_HF_DIR="${OPD5_FINAL_HF_DIR}"
export OPD_ROLLOUT_LOG_DIR="${OPD5_ROLLOUT_LOG_DIR}"
export OPD_SANITY_REPORT_DIR="${OPD5_SANITY_DIR}"
export OPD_SANITY_SUMMARY_DIR="${OPD5_SANITY_DIR}"
export OPD_TRAINED_MANIFEST="${OPD5_SAVE_DIR}/opd_trained_manifest.json"
export OPD_INITIAL_LOAD_MODE="megatron"
export OPD_INITIAL_LOAD_DIR="${OPD1_SAVE_DIR}"
export OPD_REF_LOAD_DIR="${SFT_FINAL_HF_DIR}"
export OPD_PREVIOUS_RUN_LABEL="${OPD1_RUN_LABEL}"
export OPD_PREVIOUS_SAVE_DIR="${OPD1_SAVE_DIR}"
export OPD_PREVIOUS_HF_SNAPSHOT_DIR="${OPD1_HF_SNAPSHOT_DIR}"
export OPD_PREVIOUS_ROLLOUT_LOG_DIR="${OPD1_ROLLOUT_LOG_DIR}"
export OPD_PREVIOUS_TRAINED_MANIFEST="${OPD1_SAVE_DIR}/opd_trained_manifest.json"
export OPD_CONTINUATION_METADATA="${CLEANED_METADATA}"
export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD="1"
export OPD_ALREADY_TRAINED_SAMPLES="${CLEANED_OPD_FIRST_SIZE}"
export OPD_START_ROLLOUT_ID="8"
export OPD_MANIFEST_FROM_ROLLOUT_LOGS="1"
export OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH="1"

jid_opd5="$(
  sbatch --parsable "${SBATCH_FOUR[@]}" \
    --dependency=afterok:${jid_opd1_eval} \
    --time="${CLEANED_WALLTIME_OPD5}" \
    --job-name=slime-qwen3-clean-opd5k \
    --cpus-per-task=32 \
    --export=ALL \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
)"
jid_opd5_eval="$(
  sbatch --parsable "${SBATCH_FOUR[@]}" \
    --dependency=afterok:${jid_opd5} \
    --time="${CLEANED_WALLTIME_OPD5_EVAL}" \
    --job-name=slime-qwen3-clean-opd5k-vllm \
    --cpus-per-task=32 \
    --export=ALL,EVAL_TARGETS=opd,EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch
)"
jid_report="$(
  sbatch --parsable "${SBATCH_REPORT[@]}" \
    --dependency=afterok:${jid_opd5_eval} \
    --time="${CLEANED_WALLTIME_REPORT}" \
    --job-name=slime-qwen3-clean-report \
    --cpus-per-task=8 \
    --export=ALL,EVAL_TARGETS=report \
    examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)"

{
  echo "cleaned_experiment_label=${CLEANED_EXPERIMENT_LABEL}"
  echo "submit_time=$(date --iso-8601=seconds)"
  echo "resume_after_convert=${CLEANED_RESUME_AFTER_CONVERT}"
  echo "resume_after_base_eval=${CLEANED_RESUME_AFTER_BASE_EVAL}"
  echo "resume_after_sft=${CLEANED_RESUME_AFTER_SFT}"
  echo "output_tag=${CLEANED_OUTPUT_TAG}"
  echo "vllm_setup=${jid_vllm}"
  echo "clean_data=${jid_data}"
  echo "convert=${jid_convert}"
  echo "sft_optimizer_smoke=${jid_sft_smoke}"
  echo "base_eval=${jid_base_eval}"
  echo "sft_train=${jid_sft}"
  echo "sft_eval=${jid_sft_eval}"
  echo "opd1_train=${jid_opd1}"
  echo "opd1_eval=${jid_opd1_eval}"
  echo "opd5_train=${jid_opd5}"
  echo "opd5_eval=${jid_opd5_eval}"
  echo "report=${jid_report}"
  echo "max_gpu_per_job=4"
  echo "dependency_policy=strict_afterok_chain"
  echo "sft_data=${SFT_PARQUET}"
  echo "opd_1k_data=${CLEANED_OPD_1K_JSONL}"
  echo "opd_next_4k_data=${CLEANED_OPD_4K_JSONL}"
  echo "cleaned_metadata=${CLEANED_METADATA}"
  echo "sft_save_dir=${SFT_SAVE_DIR}"
  echo "sft_hf_snapshot_dir=${SFT_HF_SNAPSHOT_DIR}"
  echo "opd1_save_dir=${OPD1_SAVE_DIR}"
  echo "opd1_hf_snapshot_dir=${OPD1_HF_SNAPSHOT_DIR}"
  echo "opd5_save_dir=${OPD5_SAVE_DIR}"
  echo "opd5_hf_snapshot_dir=${OPD5_HF_SNAPSHOT_DIR}"
  echo "base_eval_output=${BASE_EVAL_OUTPUT_DIR}"
  echo "sft_eval_output=${SFT_EVAL_OUTPUT_DIR}"
  echo "opd_eval_output=${OPD_EVAL_OUTPUT_DIR}"
  echo "combined_report_output=${COMBINED_EVAL_OUTPUT_DIR}"
  echo "sft_lr=${SFT_LR}"
  echo "opd_lr=${OPD_LR}"
  echo "sft_optimizer_mode=${SFT_OPTIMIZER_MODE}"
  echo "sft_legacy_zero_stage=${SFT_ZERO_STAGE}"
  echo "sft_ckpt_format=${SFT_CKPT_FORMAT}"
  echo "sft_loss_mask_type=${SFT_LOSS_MASK_TYPE}"
  echo "sft_loss_mask_preflight_enabled=${SFT_LOSS_MASK_PREFLIGHT_ENABLED}"
  echo "sft_loss_mask_preflight_require_think=${SFT_LOSS_MASK_PREFLIGHT_REQUIRE_THINK}"
  echo "sft_loss_mask_preflight_samples=${SFT_LOSS_MASK_PREFLIGHT_SAMPLES}"
  echo "sft_max_tokens_per_gpu=${SFT_MAX_TOKENS_PER_GPU}"
  echo "opd_max_tokens_per_gpu=${OPD_MAX_TOKENS_PER_GPU}"
  echo "opd_log_probs_chunk_size=${OPD_LOG_PROBS_CHUNK_SIZE}"
  echo "vllm_eval_num_gpus=${VLLM_EVAL_NUM_GPUS}"
  echo "vllm_eval_chunk_size=${VLLM_EVAL_CHUNK_SIZE}"
  echo "vllm_eval_resume_completed=${VLLM_EVAL_RESUME_COMPLETED}"
  echo "walltime_vllm_setup=${CLEANED_WALLTIME_VLLM_SETUP}"
  echo "walltime_clean_data=${CLEANED_WALLTIME_CLEAN_DATA}"
  echo "walltime_convert=${CLEANED_WALLTIME_CONVERT}"
  echo "walltime_smoke=${CLEANED_WALLTIME_SMOKE}"
  echo "walltime_base_eval=${CLEANED_WALLTIME_BASE_EVAL}"
  echo "walltime_sft=${CLEANED_WALLTIME_SFT}"
  echo "walltime_sft_eval=${CLEANED_WALLTIME_SFT_EVAL}"
  echo "walltime_opd1=${CLEANED_WALLTIME_OPD1}"
  echo "walltime_opd1_eval=${CLEANED_WALLTIME_OPD1_EVAL}"
  echo "walltime_opd5=${CLEANED_WALLTIME_OPD5}"
  echo "walltime_opd5_eval=${CLEANED_WALLTIME_OPD5_EVAL}"
  echo "walltime_report=${CLEANED_WALLTIME_REPORT}"
} | tee "${submit_log}"

echo "Submitted cleaned chain. Job IDs are recorded in ${submit_log}."
