#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export OPD_RUN_LABEL="${OPD_RUN_LABEL:-25k_32k_sft_colocate4_continue}"
export OPD_TRAIN_SIZE="${OPD_TRAIN_SIZE:-24960}"
export OPD_POOL_SIZE="${OPD_POOL_SIZE:-23936}"
export OPD_ALREADY_TRAINED_SAMPLES="${OPD_ALREADY_TRAINED_SAMPLES:-1024}"
export OPD_ACTOR_GPUS="${OPD_ACTOR_GPUS:-3}"
export OPD_ROLLOUT_GPUS="${OPD_ROLLOUT_GPUS:-3}"
export OPD_RAY_GPUS="${OPD_RAY_GPUS:-3}"
export OPD_TEACHER_GPU="${OPD_TEACHER_GPU:-3}"
export OPD_TENSOR_MODEL_PARALLEL_SIZE="${OPD_TENSOR_MODEL_PARALLEL_SIZE:-1}"
export OPD_CONTEXT_PARALLEL_SIZE="${OPD_CONTEXT_PARALLEL_SIZE:-3}"
export OPD_SEQ_LENGTH="${OPD_SEQ_LENGTH:-32766}"
export OPD_MAX_RESPONSE_LEN="${OPD_MAX_RESPONSE_LEN:-31744}"
export OPD_MAX_TOKENS_PER_GPU="${OPD_MAX_TOKENS_PER_GPU:-2048}"
export OPD_LOG_PROBS_CHUNK_SIZE="${OPD_LOG_PROBS_CHUNK_SIZE:-512}"
export OPD_TRAIN_MEMORY_MARGIN_BYTES="${OPD_TRAIN_MEMORY_MARGIN_BYTES:-0}"
export OPD_COLOCATE="${OPD_COLOCATE:-1}"
export OPD_OFFLOAD_TRAIN="${OPD_OFFLOAD_TRAIN:-1}"
export OPD_OFFLOAD_ROLLOUT="${OPD_OFFLOAD_ROLLOUT:-1}"
export OPD_OPTIMIZER_CPU_OFFLOAD="${OPD_OPTIMIZER_CPU_OFFLOAD:-1}"
export OPD_RECOMPUTE_LOSS_FUNCTION="${OPD_RECOMPUTE_LOSS_FUNCTION:-1}"
export OPD_SANITY_FAIL_ON_COLLAPSE="${OPD_SANITY_FAIL_ON_COLLAPSE:-0}"
export OPD_MANIFEST_FROM_ROLLOUT_LOGS="${OPD_MANIFEST_FROM_ROLLOUT_LOGS:-1}"
export OPD_INITIAL_LOAD_MODE="${OPD_INITIAL_LOAD_MODE:-megatron}"
export OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH="${OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH:-1}"
export OPD_CONTINUE_SKIP_PREP_AND_CURRENT_VAL="${OPD_CONTINUE_SKIP_PREP_AND_CURRENT_VAL:-auto}"
export OPD_CONTINUE_RESUME_ENDPOINT="${OPD_CONTINUE_RESUME_ENDPOINT:-auto}"
export GPU_GRES="${GPU_GRES:-gpu:h200:4}"
export VAL_GPU_GRES="${VAL_GPU_GRES:-gpu:h200:4}"
export REPORT_GPU_GRES="${REPORT_GPU_GRES:-gpu:h200:1}"
export EVAL_ROLLOUT_NUM_GPUS="${EVAL_ROLLOUT_NUM_GPUS:-4}"
export EVAL_ROLLOUT_BATCH_SIZE="${EVAL_ROLLOUT_BATCH_SIZE:-64}"
export EVAL_TENSOR_MODEL_PARALLEL_SIZE="${EVAL_TENSOR_MODEL_PARALLEL_SIZE:-1}"
export EVAL_CONTEXT_PARALLEL_SIZE="${EVAL_CONTEXT_PARALLEL_SIZE:-1}"
export EVAL_SGLANG_SERVER_CONCURRENCY="${EVAL_SGLANG_SERVER_CONCURRENCY:-2}"
export REPORT_EXPERIMENT_NOTE="${REPORT_EXPERIMENT_NOTE:-Continuation of corrected SFT -> OPD colocate4 from the completed 1k full optimizer checkpoint to 24,960 total OPD samples. Val100 uses a seeded uniform MATH-500 subset and all continuation OPD prompts exclude both SFT rows and the first 1,024 OPD rows.}"

source "${SCRIPT_DIR}/env.sh"

SFT_BASE_METADATA="${SFT_BASE_METADATA:-${DATA_ROOT}/openthoughts3_math_sft_${SFT_SIZE}_opd_10000_metadata.json}"
OPD_CONT_JSONL="${OPD_CONT_JSONL:-${DATA_ROOT}/openthoughts3_math_opd_continue_1k_to_25k_seed${DATA_SEED}.jsonl}"
OPD_CONT_METADATA="${OPD_CONT_METADATA:-${DATA_ROOT}/openthoughts3_math_sft_${SFT_SIZE}_opd_continue_1k_to_25k_seed${DATA_SEED}_metadata.json}"
OPD_VAL100_EVAL_OUTPUT_DIR="${OPD_VAL100_EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/math500_eval_opd_${OPD_RUN_LABEL}_val100}"
OPD_MID_FULL_EVAL_OUTPUT_DIR="${OPD_MID_FULL_EVAL_OUTPUT_DIR:-${OUTPUT_ROOT}/math500_eval_opd_${OPD_RUN_LABEL}_mid_full500}"
OPD_VAL100_COMBINED_OUTPUT_DIR="${OPD_VAL100_COMBINED_OUTPUT_DIR:-${OUTPUT_ROOT}/math500_eval_combined_${OPD_RUN_LABEL}_val100}"

export OPD_JSONL="${OPD_CONT_JSONL}"
export OPD_CONTINUATION_METADATA="${OPD_CONT_METADATA}"
export OPD_INITIAL_LOAD_DIR="${OPD_PREVIOUS_SAVE_DIR}"
export OPD_REF_LOAD_DIR="${SFT_FINAL_HF_DIR}"
export OPD_TRAINED_MANIFEST="${OPD_SAVE_DIR}/opd_trained_manifest.json"

cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

for required_path in \
  "${SFT_BASE_METADATA}" \
  "${SFT_FINAL_HF_DIR}" \
  "${OPD_PREVIOUS_SAVE_DIR}/latest_checkpointed_iteration.txt" \
  "${OPD_PREVIOUS_SAVE_DIR}/iter_0000007/.metadata" \
  "${OPD_PREVIOUS_SAVE_DIR}/rollout/global_dataset_state_dict_7.pt" \
  "${OPD_PREVIOUS_HF_SNAPSHOT_DIR}/iter_0000007" \
  "${OPD_PREVIOUS_ROLLOUT_LOG_DIR}" \
  "${MATH500_JSONL}" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" \
  "${TEACHER_HF_DIR}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path before submit: ${required_path}" >&2
    exit 1
  fi
done

previous_latest="$(tr -d '[:space:]' <"${OPD_PREVIOUS_SAVE_DIR}/latest_checkpointed_iteration.txt")"
if [[ "${previous_latest}" != "7" ]]; then
  echo "Expected previous OPD latest checkpoint iteration 7, got ${previous_latest}" >&2
  exit 1
fi
if [[ "${OPD_EFFECTIVE_TRAIN_SAMPLES}" -ne 24960 ]]; then
  echo "Expected final effective OPD samples 24960, got ${OPD_EFFECTIVE_TRAIN_SAMPLES}" >&2
  exit 1
fi
if [[ "${OPD_ALREADY_TRAINED_SAMPLES}" -ne 1024 ]]; then
  echo "Expected OPD_ALREADY_TRAINED_SAMPLES=1024, got ${OPD_ALREADY_TRAINED_SAMPLES}" >&2
  exit 1
fi
if [[ -f "${OPD_SAVE_DIR}/latest_checkpointed_iteration.txt" && "${OPD_CONTINUE_ALLOW_EXISTING_SAVE:-0}" != "1" ]]; then
  echo "New continuation save dir already has a checkpoint marker: ${OPD_SAVE_DIR}" >&2
  echo "Set OPD_CONTINUE_ALLOW_EXISTING_SAVE=1 only if intentionally resuming this continuation run." >&2
  exit 1
fi
if [[ "${OPD_CONTEXT_PARALLEL_SIZE}" -gt 1 ]]; then
  divisor=$((2 * OPD_CONTEXT_PARALLEL_SIZE))
  if (( OPD_SEQ_LENGTH % divisor != 0 )); then
    echo "OPD_SEQ_LENGTH=${OPD_SEQ_LENGTH} must be divisible by ${divisor} for CP=${OPD_CONTEXT_PARALLEL_SIZE}" >&2
    exit 1
  fi
fi

SBATCH_DATA=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres "${VAL_GPU_GRES}" --cpus-per-task=8)
SBATCH_TRAIN=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres "${GPU_GRES}" --cpus-per-task=32)
SBATCH_EVAL=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres "${VAL_GPU_GRES}" --cpus-per-task=8)
SBATCH_REPORT=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres "${REPORT_GPU_GRES}" --cpus-per-task=4)

submit_log="${SLURM_LOG_DIR}/submit_opd_continue_1k_to_25k_val100_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitting OPD continuation 1k -> 25k with val100 checkpoints"
echo "submit log: ${submit_log}"
echo "OPD save dir: ${OPD_SAVE_DIR}"
echo "OPD continuation data: ${OPD_JSONL}"
echo "OPD continuation metadata: ${OPD_CONTINUATION_METADATA}"
echo "Previous OPD checkpoint: ${OPD_PREVIOUS_SAVE_DIR}/iter_0000007"
echo "Val100 output: ${OPD_VAL100_EVAL_OUTPUT_DIR}"
echo "Midpoint full500 output: ${OPD_MID_FULL_EVAL_OUTPUT_DIR}"
echo "Train/eval/report GRES: ${GPU_GRES}/${VAL_GPU_GRES}/${REPORT_GPU_GRES}"
echo "Eval GPUs/batch/concurrency: ${EVAL_ROLLOUT_NUM_GPUS}/${EVAL_ROLLOUT_BATCH_SIZE}/${EVAL_SGLANG_SERVER_CONCURRENCY}"

format_iter_dir() {
  local base_dir="$1"
  local rollout_id="$2"
  printf "%s/iter_%07d" "${base_dir}" "${rollout_id}"
}

segment_export_vars() {
  local endpoint="$1"
  local rollout_id="$2"
  local hf_snapshot_dir="$3"
  local num_rollout=$((rollout_id + 1))
  local final_hf_dir
  final_hf_dir="$(format_iter_dir "${hf_snapshot_dir}" "${rollout_id}")"
  printf "OPD_TRAIN_SIZE=%s,OPD_NUM_ROLLOUT=%s,OPD_FINAL_ROLLOUT_ID=%s,OPD_EFFECTIVE_TRAIN_SAMPLES=%s,OPD_FINAL_HF_DIR=%s,OPD_MILESTONE_ROLLOUT_IDS=%s,OPD_SANITY_MAX_ROLLOUT_ID=%s" \
    "${endpoint}" "${num_rollout}" "${rollout_id}" "${endpoint}" "${final_hf_dir}" "${rollout_id}" "${rollout_id}"
}

stage_name_for_endpoint() {
  local endpoint="$1"
  printf "opd_%06d" "${endpoint}"
}

checkpoint_ready_for_endpoint() {
  local endpoint="$1"
  local rollout_id=$((endpoint / OPD_ROLLOUT_BATCH_SIZE - 1))
  [[ -f "$(format_iter_dir "${OPD_SAVE_DIR}" "${rollout_id}")/.metadata" && -d "$(format_iter_dir "${OPD_HF_SNAPSHOT_DIR}" "${rollout_id}")" ]]
}

val_summary_for_endpoint() {
  local endpoint="$1"
  printf "%s/%s/summary.json" "${OPD_VAL100_EVAL_OUTPUT_DIR}" "$(stage_name_for_endpoint "${endpoint}")"
}

current_val_summary="${OPD_VAL100_EVAL_OUTPUT_DIR}/opd_001024/summary.json"
skip_initial=0
case "${OPD_CONTINUE_SKIP_PREP_AND_CURRENT_VAL}" in
  1|true|yes)
    skip_initial=1
    ;;
  0|false|no)
    skip_initial=0
    ;;
  auto)
    if [[ -s "${OPD_JSONL}" && -f "${OPD_CONTINUATION_METADATA}" && -f "${MATH500_VAL100_JSONL}" && -f "${MATH500_VAL100_CONFIG}" && -f "${current_val_summary}" ]]; then
      skip_initial=1
    fi
    ;;
  *)
    echo "OPD_CONTINUE_SKIP_PREP_AND_CURRENT_VAL must be auto, 0, or 1; got ${OPD_CONTINUE_SKIP_PREP_AND_CURRENT_VAL}" >&2
    exit 1
    ;;
esac

ENDPOINTS=()
for samples in 2048 3072 4096 5120 6144 7168 8192 9216 10240 11264 12288 12544 13312 14336 15360 16384 17408 18432 19456 20480 21504 22528 23552 24576 24960; do
  ENDPOINTS+=("${samples}")
done

resume_endpoint="0"
case "${OPD_CONTINUE_RESUME_ENDPOINT}" in
  none|0)
    resume_endpoint="0"
    ;;
  auto)
    for endpoint in "${ENDPOINTS[@]}"; do
      if checkpoint_ready_for_endpoint "${endpoint}" && [[ ! -f "$(val_summary_for_endpoint "${endpoint}")" ]]; then
        resume_endpoint="${endpoint}"
        break
      fi
    done
    ;;
  ''|*[!0-9]*)
    echo "OPD_CONTINUE_RESUME_ENDPOINT must be auto, none, 0, or an endpoint sample count; got ${OPD_CONTINUE_RESUME_ENDPOINT}" >&2
    exit 1
    ;;
  *)
    resume_endpoint="${OPD_CONTINUE_RESUME_ENDPOINT}"
    found_resume_endpoint=0
    for endpoint in "${ENDPOINTS[@]}"; do
      if [[ "${endpoint}" -eq "${resume_endpoint}" ]]; then
        found_resume_endpoint=1
        break
      fi
    done
    if [[ "${found_resume_endpoint}" != "1" ]]; then
      echo "OPD_CONTINUE_RESUME_ENDPOINT=${resume_endpoint} is not one of the configured endpoints." >&2
      exit 1
    fi
    if ! checkpoint_ready_for_endpoint "${resume_endpoint}"; then
      echo "Cannot resume from endpoint ${resume_endpoint}; missing full checkpoint or HF snapshot." >&2
      echo "  full checkpoint: $(format_iter_dir "${OPD_SAVE_DIR}" "$((resume_endpoint / OPD_ROLLOUT_BATCH_SIZE - 1))")/.metadata" >&2
      echo "  HF snapshot: $(format_iter_dir "${OPD_HF_SNAPSHOT_DIR}" "$((resume_endpoint / OPD_ROLLOUT_BATCH_SIZE - 1))")" >&2
      exit 1
    fi
    ;;
esac

if [[ "${skip_initial}" == "1" ]]; then
  for preserved_path in \
    "${OPD_JSONL}" \
    "${OPD_CONTINUATION_METADATA}" \
    "${MATH500_VAL100_JSONL}" \
    "${MATH500_VAL100_CONFIG}" \
    "${current_val_summary}"; do
    if [[ ! -e "${preserved_path}" ]]; then
      echo "Cannot skip data/current-val stage; missing preserved artifact: ${preserved_path}" >&2
      exit 1
    fi
  done
  echo "Reusing completed continuation data and current 1k val100 summary: ${current_val_summary}"
  jid_data="preserved_existing_data"
  jid_current_val="preserved_existing_opd_001024_val100"
  prior_dep=""
else
  jid_data="$(
    sbatch --parsable "${SBATCH_DATA[@]}" \
      --time=03:00:00 \
      --job-name=slime-qwen3-opd25k-cont-data \
      --export=ALL,SPLIT_METADATA="${SFT_BASE_METADATA}",OPD_JSONL="${OPD_JSONL}",OPD_CONTINUATION_METADATA="${OPD_CONTINUATION_METADATA}",OPD_TRAIN_SIZE=24960,OPD_ALREADY_TRAINED_SAMPLES="${OPD_ALREADY_TRAINED_SAMPLES}" \
      examples/qwen3_8b_opd_tillicum/02_prepare_opd_continuation_25k_val100.sbatch
  )"

  current_val_exports="$(segment_export_vars 1024 7 "${OPD_PREVIOUS_HF_SNAPSHOT_DIR}")"
  jid_current_val="$(
    sbatch --parsable "${SBATCH_EVAL[@]}" \
      --dependency=afterok:${jid_data} \
      --time=05:00:00 \
      --job-name=slime-qwen3-opd1k-val100 \
      --export=ALL,${current_val_exports},OPD_HF_SNAPSHOT_DIR="${OPD_PREVIOUS_HF_SNAPSHOT_DIR}",EVAL_TARGETS=opd,EVAL_OUTPUT_DIR="${OPD_VAL100_EVAL_OUTPUT_DIR}",OPD_EVAL_OUTPUT_DIR="${OPD_VAL100_EVAL_OUTPUT_DIR}",MATH500_JSONL="${MATH500_VAL100_JSONL}",MATH500_CONFIG="${MATH500_VAL100_CONFIG}",EVAL_EXPECTED_SAMPLES=100,EVAL_SKIP_COMPLETED=1 \
      examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
  )"
  prior_dep="${jid_current_val}"
fi

mid_full_jid=""
train_jobs=()
val_jobs=("${jid_current_val}")

for endpoint in "${ENDPOINTS[@]}"; do
  rid=$((endpoint / OPD_ROLLOUT_BATCH_SIZE - 1))
  endpoint_exports="$(segment_export_vars "${endpoint}" "${rid}" "${OPD_HF_SNAPSHOT_DIR}")"
  endpoint_val_summary="$(val_summary_for_endpoint "${endpoint}")"
  if checkpoint_ready_for_endpoint "${endpoint}" && [[ -f "${endpoint_val_summary}" ]]; then
    train_jobs+=("preserved_existing_train_${endpoint}")
    val_jobs+=("preserved_existing_val100_${endpoint}")
    prior_dep=""
    continue
  fi
  if [[ "${resume_endpoint}" != "0" && "${endpoint}" -lt "${resume_endpoint}" ]]; then
    if [[ ! -f "${endpoint_val_summary}" ]]; then
      echo "Cannot skip endpoint ${endpoint}; missing prior val100 summary: ${endpoint_val_summary}" >&2
      exit 1
    fi
    train_jobs+=("preserved_existing_train_${endpoint}")
    val_jobs+=("preserved_existing_val100_${endpoint}")
    continue
  fi
  if [[ "${resume_endpoint}" != "0" && "${endpoint}" -eq "${resume_endpoint}" ]]; then
    train_jobs+=("preserved_existing_train_${endpoint}")
    if [[ -f "${endpoint_val_summary}" ]]; then
      jid_val="preserved_existing_val100_${endpoint}"
      prior_dep=""
    else
      val_dependency_args=()
      if [[ -n "${prior_dep}" ]]; then
        val_dependency_args=(--dependency=afterok:${prior_dep})
      fi
      jid_val="$(
        sbatch --parsable "${SBATCH_EVAL[@]}" \
          "${val_dependency_args[@]}" \
          --time=05:00:00 \
          --job-name="$(printf "slime-qwen3-val100-%05d" "${endpoint}")" \
          --export=ALL,${endpoint_exports},OPD_HF_SNAPSHOT_DIR="${OPD_HF_SNAPSHOT_DIR}",EVAL_TARGETS=opd,EVAL_OUTPUT_DIR="${OPD_VAL100_EVAL_OUTPUT_DIR}",OPD_EVAL_OUTPUT_DIR="${OPD_VAL100_EVAL_OUTPUT_DIR}",MATH500_JSONL="${MATH500_VAL100_JSONL}",MATH500_CONFIG="${MATH500_VAL100_CONFIG}",EVAL_EXPECTED_SAMPLES=100,EVAL_SKIP_COMPLETED=1 \
          examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
      )"
      prior_dep="${jid_val}"
    fi
    val_jobs+=("${jid_val}")
    continue
  fi
  train_time="08:00:00"
  if [[ "${endpoint}" -eq 12544 ]]; then
    train_time="03:00:00"
  elif [[ "${endpoint}" -eq 24960 ]]; then
    train_time="04:00:00"
  fi
  skip_data_state_load=0
  if [[ "${endpoint}" -eq 2048 ]]; then
    skip_data_state_load=1
  fi
  num_rollout=$((rid + 1))
  train_dependency_args=()
  if [[ -n "${prior_dep}" ]]; then
    train_dependency_args=(--dependency=afterok:${prior_dep})
  fi

  jid_train="$(
    sbatch --parsable "${SBATCH_TRAIN[@]}" \
      "${train_dependency_args[@]}" \
      --time="${train_time}" \
      --job-name="$(printf "slime-qwen3-opd-cont-%05d" "${endpoint}")" \
      --export=ALL,SPLIT_METADATA="${OPD_CONTINUATION_METADATA}",OPD_JSONL="${OPD_JSONL}",OPD_CONTINUATION_METADATA="${OPD_CONTINUATION_METADATA}",${endpoint_exports},OPD_SKIP_ROLLOUT_DATA_STATE_LOAD="${skip_data_state_load}",OPD_ALREADY_TRAINED_SAMPLES="${OPD_ALREADY_TRAINED_SAMPLES}",OPD_INITIAL_LOAD_MODE="${OPD_INITIAL_LOAD_MODE}",OPD_INITIAL_LOAD_DIR="${OPD_INITIAL_LOAD_DIR}",OPD_REF_LOAD_DIR="${OPD_REF_LOAD_DIR}",OPD_MANIFEST_FROM_ROLLOUT_LOGS=1,OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH="${OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH}" \
      examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
  )"
  train_jobs+=("${jid_train}")

  jid_val="$(
    sbatch --parsable "${SBATCH_EVAL[@]}" \
      --dependency=afterok:${jid_train} \
      --time=05:00:00 \
      --job-name="$(printf "slime-qwen3-val100-%05d" "${endpoint}")" \
      --export=ALL,${endpoint_exports},OPD_HF_SNAPSHOT_DIR="${OPD_HF_SNAPSHOT_DIR}",EVAL_TARGETS=opd,EVAL_OUTPUT_DIR="${OPD_VAL100_EVAL_OUTPUT_DIR}",OPD_EVAL_OUTPUT_DIR="${OPD_VAL100_EVAL_OUTPUT_DIR}",MATH500_JSONL="${MATH500_VAL100_JSONL}",MATH500_CONFIG="${MATH500_VAL100_CONFIG}",EVAL_EXPECTED_SAMPLES=100,EVAL_SKIP_COMPLETED=1 \
      examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
  )"
  val_jobs+=("${jid_val}")

  if [[ "${endpoint}" -eq 12544 ]]; then
    mid_full_jid="$(
      sbatch --parsable "${SBATCH_EVAL[@]}" \
        --dependency=afterok:${jid_val} \
        --time=20:00:00 \
        --job-name=slime-qwen3-opd12544-full500 \
        --export=ALL,${endpoint_exports},OPD_HF_SNAPSHOT_DIR="${OPD_HF_SNAPSHOT_DIR}",EVAL_TARGETS=opd,EVAL_OUTPUT_DIR="${OPD_MID_FULL_EVAL_OUTPUT_DIR}",OPD_EVAL_OUTPUT_DIR="${OPD_MID_FULL_EVAL_OUTPUT_DIR}",EVAL_EXPECTED_SAMPLES=500,EVAL_SKIP_COMPLETED=1 \
        examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
    )"
    prior_dep="${mid_full_jid}"
  else
    prior_dep="${jid_val}"
  fi
done

report_dependency="afterany:${prior_dep}"
if [[ -n "${mid_full_jid}" ]]; then
  report_dependency+=":${mid_full_jid}"
fi
jid_report="$(
  sbatch --parsable "${SBATCH_REPORT[@]}" \
    --dependency="${report_dependency}" \
    --time=00:30:00 \
    --job-name=slime-qwen3-opd25k-val-report \
    --export=ALL,EVAL_TARGETS=report,EVAL_OUTPUT_DIR="${OPD_VAL100_COMBINED_OUTPUT_DIR}",OPD_EVAL_OUTPUT_DIR="${OPD_VAL100_EVAL_OUTPUT_DIR}",BASE_EVAL_OUTPUT_DIR="${OPD_VAL100_COMBINED_OUTPUT_DIR}/empty_base",COMBINED_EVAL_OUTPUT_DIR="${OPD_VAL100_COMBINED_OUTPUT_DIR}",REPORT_INCLUDE_SFT=0,REPORT_OPD_X_OFFSET_SAMPLES=0,REPORT_OPD_LABEL_PREFIX="" \
    examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)"

{
  echo "data=${jid_data}"
  echo "val100_current_001024=${jid_current_val}"
  for i in "${!ENDPOINTS[@]}"; do
    endpoint="${ENDPOINTS[$i]}"
    echo "train_${endpoint}=${train_jobs[$i]}"
    echo "val100_${endpoint}=${val_jobs[$((i + 1))]}"
  done
  echo "mid_full500_012544=${mid_full_jid}"
  echo "final_report=${jid_report}"
  echo "report_dependency=${report_dependency}"
  echo "OPD_RUN_LABEL=${OPD_RUN_LABEL}"
  echo "OPD_PREVIOUS_SAVE_DIR=${OPD_PREVIOUS_SAVE_DIR}"
  echo "OPD_PREVIOUS_HF_SNAPSHOT_DIR=${OPD_PREVIOUS_HF_SNAPSHOT_DIR}"
  echo "OPD_PREVIOUS_ROLLOUT_LOG_DIR=${OPD_PREVIOUS_ROLLOUT_LOG_DIR}"
  echo "OPD_SAVE_DIR=${OPD_SAVE_DIR}"
  echo "OPD_HF_SNAPSHOT_DIR=${OPD_HF_SNAPSHOT_DIR}"
  echo "OPD_ROLLOUT_LOG_DIR=${OPD_ROLLOUT_LOG_DIR}"
  echo "OPD_JSONL=${OPD_JSONL}"
  echo "SFT_BASE_METADATA=${SFT_BASE_METADATA}"
  echo "OPD_CONTINUATION_METADATA=${OPD_CONTINUATION_METADATA}"
  echo "MATH500_VAL100_JSONL=${MATH500_VAL100_JSONL}"
  echo "MATH500_VAL100_CONFIG=${MATH500_VAL100_CONFIG}"
  echo "MATH500_VAL100_METADATA=${MATH500_VAL100_METADATA}"
  echo "OPD_VAL100_EVAL_OUTPUT_DIR=${OPD_VAL100_EVAL_OUTPUT_DIR}"
  echo "OPD_MID_FULL_EVAL_OUTPUT_DIR=${OPD_MID_FULL_EVAL_OUTPUT_DIR}"
  echo "OPD_VAL100_COMBINED_OUTPUT_DIR=${OPD_VAL100_COMBINED_OUTPUT_DIR}"
  echo "OPD_INITIAL_LOAD_MODE=${OPD_INITIAL_LOAD_MODE}"
  echo "OPD_INITIAL_LOAD_DIR=${OPD_INITIAL_LOAD_DIR}"
  echo "OPD_REF_LOAD_DIR=${OPD_REF_LOAD_DIR}"
  echo "OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=${OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH}"
  echo "OPD_CONTINUE_SKIP_PREP_AND_CURRENT_VAL=${OPD_CONTINUE_SKIP_PREP_AND_CURRENT_VAL}"
  echo "OPD_CONTINUE_RESUME_ENDPOINT=${OPD_CONTINUE_RESUME_ENDPOINT}"
  echo "OPD_CONTINUE_RESUME_ENDPOINT_RESOLVED=${resume_endpoint}"
  echo "OPD_CONTINUE_INITIAL_STAGES_SKIPPED=${skip_initial}"
  echo "OPD_ALREADY_TRAINED_SAMPLES=${OPD_ALREADY_TRAINED_SAMPLES}"
  echo "OPD_EFFECTIVE_TRAIN_SAMPLES=${OPD_EFFECTIVE_TRAIN_SAMPLES}"
  echo "OPD_MAX_RESPONSE_LEN=${OPD_MAX_RESPONSE_LEN}"
  echo "OPD_SEQ_LENGTH=${OPD_SEQ_LENGTH}"
  echo "OPD_MAX_TOKENS_PER_GPU=${OPD_MAX_TOKENS_PER_GPU}"
  echo "OPD_LOG_PROBS_CHUNK_SIZE=${OPD_LOG_PROBS_CHUNK_SIZE}"
  echo "OPD_TRAIN_MEMORY_MARGIN_BYTES=${OPD_TRAIN_MEMORY_MARGIN_BYTES}"
  echo "OPD_COLOCATE=${OPD_COLOCATE}"
  echo "OPD_OFFLOAD_TRAIN=${OPD_OFFLOAD_TRAIN}"
  echo "OPD_OFFLOAD_ROLLOUT=${OPD_OFFLOAD_ROLLOUT}"
  echo "OPD_OPTIMIZER_CPU_OFFLOAD=${OPD_OPTIMIZER_CPU_OFFLOAD}"
  echo "OPD_RECOMPUTE_LOSS_FUNCTION=${OPD_RECOMPUTE_LOSS_FUNCTION}"
  echo "EVAL_ROLLOUT_NUM_GPUS=${EVAL_ROLLOUT_NUM_GPUS}"
  echo "EVAL_ROLLOUT_BATCH_SIZE=${EVAL_ROLLOUT_BATCH_SIZE}"
  echo "EVAL_SGLANG_SERVER_CONCURRENCY=${EVAL_SGLANG_SERVER_CONCURRENCY}"
  echo "VAL_GPU_GRES=${VAL_GPU_GRES}"
  echo "REPORT_GPU_GRES=${REPORT_GPU_GRES}"
} | tee "${submit_log}"
