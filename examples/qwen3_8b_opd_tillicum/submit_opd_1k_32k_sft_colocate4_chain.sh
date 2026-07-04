#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

export OPD_RUN_LABEL="${OPD_RUN_LABEL:-1k_32k_sft_colocate4}"
export OPD_INITIAL_LOAD_MODE="${OPD_INITIAL_LOAD_MODE:-hf}"
export OPD_ACTOR_GPUS="${OPD_ACTOR_GPUS:-3}"
export OPD_ROLLOUT_GPUS="${OPD_ROLLOUT_GPUS:-3}"
export OPD_RAY_GPUS="${OPD_RAY_GPUS:-3}"
export OPD_TEACHER_GPU="${OPD_TEACHER_GPU:-3}"
export OPD_TENSOR_MODEL_PARALLEL_SIZE="${OPD_TENSOR_MODEL_PARALLEL_SIZE:-1}"
export OPD_CONTEXT_PARALLEL_SIZE="${OPD_CONTEXT_PARALLEL_SIZE:-3}"
export OPD_SEQ_LENGTH="${OPD_SEQ_LENGTH:-32766}"
export OPD_MAX_RESPONSE_LEN="${OPD_MAX_RESPONSE_LEN:-31744}"
export OPD_MAX_TOKENS_PER_GPU="${OPD_MAX_TOKENS_PER_GPU:-4096}"
export OPD_TRAIN_MEMORY_MARGIN_BYTES="${OPD_TRAIN_MEMORY_MARGIN_BYTES:-0}"
export OPD_COLOCATE="${OPD_COLOCATE:-1}"
export OPD_OFFLOAD_TRAIN="${OPD_OFFLOAD_TRAIN:-1}"
export OPD_OFFLOAD_ROLLOUT="${OPD_OFFLOAD_ROLLOUT:-1}"
export OPD_OPTIMIZER_CPU_OFFLOAD="${OPD_OPTIMIZER_CPU_OFFLOAD:-1}"
export OPD_RECOMPUTE_LOSS_FUNCTION="${OPD_RECOMPUTE_LOSS_FUNCTION:-1}"
export GPU_GRES="${GPU_GRES:-gpu:h200:4}"
export EVAL_GPU_GRES="${EVAL_GPU_GRES:-gpu:h200:4}"
export EVAL_ROLLOUT_NUM_GPUS="${EVAL_ROLLOUT_NUM_GPUS:-4}"
export EVAL_ROLLOUT_BATCH_SIZE="${EVAL_ROLLOUT_BATCH_SIZE:-128}"
export EVAL_SGLANG_SERVER_CONCURRENCY="${EVAL_SGLANG_SERVER_CONCURRENCY:-4}"
export CORRECTED_AFTERANY_DEPENDENCY="${CORRECTED_AFTERANY_DEPENDENCY:-none}"
export REPORT_EXPERIMENT_NOTE="${REPORT_EXPERIMENT_NOTE:-Corrected SFT -> OPD run: OPD initializes from the final SFT HF weights at rollout 96 with a fresh OPD optimizer. The 4-GPU corrected run colocates actor and student rollout engines on GPUs 0,1,2 with CP=3 and keeps the teacher on GPU 3. The earlier 1k_32k run is an accidental base -> OPD test because it used an HF snapshot as Megatron --load without the explicit HF-load path and fell back to base.}"

source "${SCRIPT_DIR}/env.sh"
export BASE_EVAL_REUSE_DIR="${BASE_EVAL_REUSE_DIR:-${OUTPUT_ROOT}/math500_eval_base_25k_opd_1k_32k}"

cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

SBATCH_TRAIN=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres "${GPU_GRES}" --cpus-per-task=32)
SBATCH_EVAL=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres "${EVAL_GPU_GRES}" --cpus-per-task=32)
SBATCH_REPORT=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}")
TRAIN_DEP_ARGS=()
TRAIN_DEP_LABEL="none"
if [[ -n "${CORRECTED_AFTERANY_DEPENDENCY}" && "${CORRECTED_AFTERANY_DEPENDENCY}" != "none" ]]; then
  TRAIN_DEP_ARGS=(--dependency=afterany:${CORRECTED_AFTERANY_DEPENDENCY})
  TRAIN_DEP_LABEL="afterany:${CORRECTED_AFTERANY_DEPENDENCY}"
fi

printf -v SFT_FINAL_STAGE "sft_%06d" "$((SFT_NUM_ROLLOUT * SFT_ROLLOUT_BATCH_SIZE))"
SFT_FINAL_SUMMARY="${SFT_EVAL_OUTPUT_DIR}/${SFT_FINAL_STAGE}/summary.json"

for required_path in \
  "${OPD_JSONL}" \
  "${SPLIT_METADATA}" \
  "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt" \
  "${SFT_FINAL_FULL_CKPT_DIR}/.metadata" \
  "${SFT_FINAL_FULL_CKPT_DIR}/common.pt" \
  "${SFT_FINAL_HF_DIR}" \
  "${SFT_FINAL_SUMMARY}" \
  "${TEACHER_HF_DIR}" \
  "${STUDENT_HF_DIR}" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required path before submit: ${required_path}" >&2
    exit 1
  fi
done

if [[ "${OPD_EFFECTIVE_TRAIN_SAMPLES}" -ne 1024 ]]; then
  echo "Expected 1024 effective OPD samples, got ${OPD_EFFECTIVE_TRAIN_SAMPLES}" >&2
  exit 1
fi
if [[ "${OPD_CONTEXT_PARALLEL_SIZE}" -gt 1 ]]; then
  divisor=$((2 * OPD_CONTEXT_PARALLEL_SIZE))
  if (( OPD_SEQ_LENGTH % divisor != 0 )); then
    echo "OPD_SEQ_LENGTH=${OPD_SEQ_LENGTH} must be divisible by ${divisor} for CP=${OPD_CONTEXT_PARALLEL_SIZE}" >&2
    exit 1
  fi
fi
case "${OPD_INITIAL_LOAD_MODE}" in
  hf)
    for required_path in \
      "${OPD_INITIAL_LOAD_DIR}" \
      "${OPD_INITIAL_LOAD_DIR}/config.json"; do
      if [[ ! -e "${required_path}" ]]; then
        echo "Missing required HF initial load path before submit: ${required_path}" >&2
        exit 1
      fi
    done
    shopt -s nullglob
    hf_weight_files=("${OPD_INITIAL_LOAD_DIR}"/model*.safetensors "${OPD_INITIAL_LOAD_DIR}"/pytorch_model*.bin)
    shopt -u nullglob
    if [[ ! -f "${OPD_INITIAL_LOAD_DIR}/model.safetensors.index.json" && "${#hf_weight_files[@]}" -eq 0 ]]; then
      echo "HF initial load dir is missing model weights: ${OPD_INITIAL_LOAD_DIR}" >&2
      exit 1
    fi
    ;;
  megatron)
    for required_path in \
      "${OPD_INITIAL_LOAD_DIR}/latest_checkpointed_iteration.txt"; do
      if [[ ! -e "${required_path}" ]]; then
        echo "Missing required Megatron initial load path before submit: ${required_path}" >&2
        exit 1
      fi
    done
    ;;
  *)
    echo "OPD_INITIAL_LOAD_MODE must be 'hf' or 'megatron', got '${OPD_INITIAL_LOAD_MODE}'" >&2
    exit 1
    ;;
esac

submit_log="${SLURM_LOG_DIR}/submit_opd_${OPD_RUN_LABEL}_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitting corrected SFT-loaded OPD ${OPD_RUN_LABEL} chain"
echo "account/partition/qos: ${ACCOUNT}/${PARTITION}/${QOS}"
echo "submit log: ${submit_log}"
echo "wait dependency: ${TRAIN_DEP_LABEL}"
echo "train/eval gres: ${GPU_GRES}/${EVAL_GPU_GRES}"
echo "OPD initial load mode: ${OPD_INITIAL_LOAD_MODE}"
echo "OPD initial load: ${OPD_INITIAL_LOAD_DIR}"
echo "OPD actor/rollout/teacher GPU counts: ${OPD_ACTOR_GPUS}/${OPD_ROLLOUT_GPUS}/1"
echo "OPD teacher physical GPU: ${OPD_TEACHER_GPU}"
echo "OPD Ray GPUs: ${OPD_RAY_GPUS}"
echo "OPD TP/CP/max response/seq length/max tokens per GPU: ${OPD_TENSOR_MODEL_PARALLEL_SIZE}/${OPD_CONTEXT_PARALLEL_SIZE}/${OPD_MAX_RESPONSE_LEN}/${OPD_SEQ_LENGTH}/${OPD_MAX_TOKENS_PER_GPU}"
echo "OPD train memory margin bytes: ${OPD_TRAIN_MEMORY_MARGIN_BYTES}"
echo "OPD colocate/offload train/offload rollout: ${OPD_COLOCATE}/${OPD_OFFLOAD_TRAIN}/${OPD_OFFLOAD_ROLLOUT}"
echo "OPD optimizer CPU offload/recompute loss function: ${OPD_OPTIMIZER_CPU_OFFLOAD}/${OPD_RECOMPUTE_LOSS_FUNCTION}"
echo "OPD/EVAL disable cuda graph: ${OPD_DISABLE_CUDA_GRAPH}/${EVAL_DISABLE_CUDA_GRAPH}"
echo "SLIME SGLang force native RoPE: ${SLIME_SGLANG_FORCE_NATIVE_ROPE}"
echo "OPD/EVAL SGLang rl-on-policy target: ${OPD_SGLANG_RL_ON_POLICY_TARGET:-<none>}/${EVAL_SGLANG_RL_ON_POLICY_TARGET:-<none>}"
echo "OPD sanity guard: enabled=${OPD_SANITY_CHECK_ENABLED} max_rollout=${OPD_SANITY_MAX_ROLLOUT_ID} max_cap_hit=${OPD_SANITY_MAX_CAP_HIT_RATE} max_avg_tokens=${OPD_SANITY_MAX_AVG_RESPONSE_TOKENS} min_final_answer=${OPD_SANITY_MIN_FINAL_ANSWER_RATE}"
echo "Eval GPUs/batch/concurrency: ${EVAL_ROLLOUT_NUM_GPUS}/${EVAL_ROLLOUT_BATCH_SIZE}/${EVAL_SGLANG_SERVER_CONCURRENCY}"

jid_opd="$(
  sbatch --parsable "${SBATCH_TRAIN[@]}" \
    "${TRAIN_DEP_ARGS[@]}" \
    --time=18:00:00 \
    --job-name=slime-qwen3-opd1k-sft4g \
    --export=ALL \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
)"
jid_opd_eval="$(
  sbatch --parsable "${SBATCH_EVAL[@]}" \
    --dependency=afterok:${jid_opd} \
    --time=05:00:00 \
    --job-name=slime-qwen3-opd1k-sft4g-eval \
    --export=ALL,EVAL_TARGETS=opd,EVAL_OUTPUT_DIR="${OPD_EVAL_OUTPUT_DIR}",EVAL_SKIP_COMPLETED=1,OPD_MILESTONE_ROLLOUT_IDS="${OPD_FINAL_ROLLOUT_ID}" \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_greedy_1x.sbatch
)"
jid_base_maybe="$(
  sbatch --parsable "${SBATCH_EVAL[@]}" \
    --dependency=afterok:${jid_opd_eval} \
    --time=05:00:00 \
    --job-name=slime-qwen3-base-math500-maybe \
    --export=ALL,EVAL_TARGETS=base,EVAL_OUTPUT_DIR="${BASE_EVAL_OUTPUT_DIR}",EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/08_maybe_base_eval_math500.sbatch
)"
jid_report="$(
  sbatch --parsable "${SBATCH_REPORT[@]}" \
    --dependency=afterok:${jid_base_maybe} \
    --time=00:30:00 \
    --job-name=slime-qwen3-final-report-sft4g \
    --export=ALL,EVAL_TARGETS=report,EVAL_OUTPUT_DIR="${COMBINED_EVAL_OUTPUT_DIR}" \
    examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)"

{
  echo "opd=${jid_opd}"
  echo "opd_eval_final=${jid_opd_eval}"
  echo "base_maybe=${jid_base_maybe}"
  echo "report=${jid_report}"
  echo "dependency=${TRAIN_DEP_LABEL}"
  echo "SFT_SAVE_DIR=${SFT_SAVE_DIR}"
  echo "OPD_INITIAL_LOAD_MODE=${OPD_INITIAL_LOAD_MODE}"
  echo "OPD_INITIAL_LOAD_DIR=${OPD_INITIAL_LOAD_DIR}"
  echo "OPD_COLOCATE=${OPD_COLOCATE}"
  echo "OPD_OFFLOAD_TRAIN=${OPD_OFFLOAD_TRAIN}"
  echo "OPD_OFFLOAD_ROLLOUT=${OPD_OFFLOAD_ROLLOUT}"
  echo "OPD_OPTIMIZER_CPU_OFFLOAD=${OPD_OPTIMIZER_CPU_OFFLOAD}"
  echo "OPD_RECOMPUTE_LOSS_FUNCTION=${OPD_RECOMPUTE_LOSS_FUNCTION}"
  echo "OPD_TRAIN_MEMORY_MARGIN_BYTES=${OPD_TRAIN_MEMORY_MARGIN_BYTES}"
  echo "OPD_DISABLE_CUDA_GRAPH=${OPD_DISABLE_CUDA_GRAPH}"
  echo "EVAL_DISABLE_CUDA_GRAPH=${EVAL_DISABLE_CUDA_GRAPH}"
  echo "SLIME_SGLANG_FORCE_NATIVE_ROPE=${SLIME_SGLANG_FORCE_NATIVE_ROPE}"
  echo "OPD_SGLANG_RL_ON_POLICY_TARGET=${OPD_SGLANG_RL_ON_POLICY_TARGET}"
  echo "EVAL_SGLANG_RL_ON_POLICY_TARGET=${EVAL_SGLANG_RL_ON_POLICY_TARGET}"
  echo "OPD_SANITY_CHECK_ENABLED=${OPD_SANITY_CHECK_ENABLED}"
  echo "OPD_SANITY_REPORT_DIR=${OPD_SANITY_REPORT_DIR}"
  echo "SFT_FINAL_FULL_CKPT_DIR=${SFT_FINAL_FULL_CKPT_DIR}"
  echo "SFT_FINAL_HF_DIR=${SFT_FINAL_HF_DIR}"
  echo "SFT_FINAL_SUMMARY=${SFT_FINAL_SUMMARY}"
  echo "OPD_JSONL=${OPD_JSONL}"
  echo "SPLIT_METADATA=${SPLIT_METADATA}"
  echo "OPD_SAVE_DIR=${OPD_SAVE_DIR}"
  echo "OPD_HF_SNAPSHOT_DIR=${OPD_HF_SNAPSHOT_DIR}"
  echo "OPD_FINAL_HF_DIR=${OPD_FINAL_HF_DIR}"
  echo "OPD_TRAINED_MANIFEST=${OPD_TRAINED_MANIFEST}"
  echo "OPD_EVAL_OUTPUT_DIR=${OPD_EVAL_OUTPUT_DIR}"
  echo "BASE_EVAL_OUTPUT_DIR=${BASE_EVAL_OUTPUT_DIR}"
  echo "BASE_EVAL_REUSE_DIR=${BASE_EVAL_REUSE_DIR}"
  echo "COMBINED_EVAL_OUTPUT_DIR=${COMBINED_EVAL_OUTPUT_DIR}"
  echo "CHECKPOINT_REPORT_DIR=${CHECKPOINT_REPORT_DIR}"
  echo "EVAL_ROLLOUT_NUM_GPUS=${EVAL_ROLLOUT_NUM_GPUS}"
  echo "EVAL_ROLLOUT_BATCH_SIZE=${EVAL_ROLLOUT_BATCH_SIZE}"
  echo "EVAL_SGLANG_SERVER_CONCURRENCY=${EVAL_SGLANG_SERVER_CONCURRENCY}"
} | tee "${submit_log}"
