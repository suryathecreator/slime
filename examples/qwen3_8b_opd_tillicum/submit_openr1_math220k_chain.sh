#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/openr1_math220k_env.sh"

export OPENR1_WALLTIME_DATA="${OPENR1_WALLTIME_DATA:-06:00:00}"
export OPENR1_WALLTIME_SFT="${OPENR1_WALLTIME_SFT:-24:00:00}"
export OPENR1_WALLTIME_EVAL="${OPENR1_WALLTIME_EVAL:-24:00:00}"
export OPENR1_WALLTIME_OPD="${OPENR1_WALLTIME_OPD:-24:00:00}"
export OPENR1_WALLTIME_REPORT="${OPENR1_WALLTIME_REPORT:-06:00:00}"
export OPENR1_RESUME_FROM_SFT_CHECKPOINT="${OPENR1_RESUME_FROM_SFT_CHECKPOINT:-0}"

cd "${SLIME_REPO_ROOT}"
mkdir -p "${SLURM_LOG_DIR}"

for required_path in \
  "${STUDENT_HF_DIR}/config.json" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" \
  "${TEACHER_HF_DIR}/config.json" \
  "${VLLM_EVAL_SITE}/vllm" \
  "${BASE_EVAL_OUTPUT_DIR}/base/summary.json"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "Missing required OpenR1 prerequisite: ${required_path}" >&2
    exit 1
  fi
done
case "${OPENR1_RESUME_FROM_SFT_CHECKPOINT}" in
  0|1) ;;
  *)
    echo "OPENR1_RESUME_FROM_SFT_CHECKPOINT must be 0 or 1, got ${OPENR1_RESUME_FROM_SFT_CHECKPOINT}" >&2
    exit 1
    ;;
esac
resume_sft_checkpoint_iteration=""
if [[ "${OPENR1_RESUME_FROM_SFT_CHECKPOINT}" == "1" ]]; then
  for required_resume_path in \
    "${OPENR1_SUCCESS_FILE}" \
    "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt"; do
    if [[ ! -e "${required_resume_path}" ]]; then
      echo "Missing required OpenR1 resume path: ${required_resume_path}" >&2
      exit 1
    fi
  done
  resume_sft_checkpoint_iteration="$(<"${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt")"
elif [[ -e "${OPENR1_SUCCESS_FILE}" || -e "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt" ]]; then
  echo "OpenR1 output already exists. Refusing an accidental overwrite/resubmission." >&2
  echo "  data marker: ${OPENR1_SUCCESS_FILE}" >&2
  echo "  SFT save: ${SFT_SAVE_DIR}" >&2
  echo "Set OPENR1_RESUME_FROM_SFT_CHECKPOINT=1 to resume from the existing SFT optimizer checkpoint." >&2
  exit 1
fi
if (( OPD_SEQ_LENGTH % (2 * OPD_CONTEXT_PARALLEL_SIZE) != 0 )); then
  echo "OPD_SEQ_LENGTH=${OPD_SEQ_LENGTH} is not divisible by 2*CP=$((2 * OPD_CONTEXT_PARALLEL_SIZE))" >&2
  exit 1
fi

SBATCH_ONE=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:1)
SBATCH_FOUR=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:4)
submit_log="${SLURM_LOG_DIR}/submit_openr1_math220k_$(date +%Y%m%d_%H%M%S).txt"

echo "Submitting ${OPENR1_EXPERIMENT_LABEL}"
echo "Dataset: ${OPENR1_DATASET}/${OPENR1_CONFIG}@${OPENR1_REVISION} seed=${DATA_SEED}"
echo "SFT: 50000 rows, batch 250, 200 updates, TP/CP/DP 2/1/2, LR ${SFT_LR}, max_tokens_per_gpu ${SFT_MAX_TOKENS_PER_GPU}, log_probs_chunk ${SFT_LOG_PROBS_CHUNK_SIZE}, optimizer_cpu_offload ${SFT_OPTIMIZER_CPU_OFFLOAD}, recompute_loss ${SFT_RECOMPUTE_LOSS_FUNCTION}"
if [[ "${OPENR1_RESUME_FROM_SFT_CHECKPOINT}" == "1" ]]; then
  echo "Resume: existing OpenR1 data and SFT checkpoint iteration ${resume_sft_checkpoint_iteration}"
fi
echo "OPD: 1024 then 4 x 1024 new prompts, actor/rollout/teacher 3/3/1, TP/CP 1/3"
echo "vLLM eval: replicas=${VLLM_EVAL_NUM_GPUS} max_seqs=${VLLM_EVAL_MAX_NUM_SEQS} batch_tokens=${VLLM_EVAL_MAX_NUM_BATCHED_TOKENS} gpu_memory=${VLLM_EVAL_GPU_MEMORY_UTILIZATION} kv=${VLLM_EVAL_KV_CACHE_DTYPE} speculative=${VLLM_EVAL_SPECULATIVE_METHOD}:${VLLM_EVAL_NUM_SPECULATIVE_TOKENS}"
echo "Output data: ${OPENR1_DATA_DIR}"
echo "Submission log: ${submit_log}"

jid_data="existing"
sft_dependency_args=()
sft_job_name=slime-qwen3-openr1-sft50k-resume
if [[ "${OPENR1_RESUME_FROM_SFT_CHECKPOINT}" != "1" ]]; then
  jid_data="$(
    sbatch --parsable "${SBATCH_ONE[@]}" \
      --time="${OPENR1_WALLTIME_DATA}" \
      --cpus-per-task=8 \
      --job-name=slime-qwen3-openr1-data \
      --export=ALL \
      examples/qwen3_8b_opd_tillicum/14_prepare_openr1_math220k.sbatch
  )"
  sft_dependency_args=(--dependency=afterok:${jid_data})
  sft_job_name=slime-qwen3-openr1-sft50k
fi
jid_sft="$(
  sbatch --parsable "${SBATCH_FOUR[@]}" \
    "${sft_dependency_args[@]}" \
    --time="${OPENR1_WALLTIME_SFT}" \
    --cpus-per-task=32 \
    --job-name="${sft_job_name}" \
    --export=ALL \
    examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch
)"
jid_sft_eval="$(
  sbatch --parsable "${SBATCH_FOUR[@]}" \
    --dependency=afterok:${jid_sft} \
    --time="${OPENR1_WALLTIME_EVAL}" \
    --cpus-per-task=32 \
    --job-name=slime-qwen3-openr1-sft-eval \
    --export=ALL,EVAL_TARGETS=sft,EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch
)"

export OPD_RUN_LABEL="${OPENR1_OPD1_RUN_LABEL}"
export OPD_JSONL="${CLEANED_OPD_1K_JSONL}"
export OPD_POOL_SIZE=1024
export OPD_SIZE=1024
export OPD_TRAIN_SIZE=1024
export OPD_NUM_ROLLOUT=8
export OPD_FINAL_ROLLOUT_ID=7
export OPD_EFFECTIVE_TRAIN_SAMPLES=1024
export OPD_MILESTONE_ROLLOUT_IDS=7
export OPD_SAVE_INTERVAL=8
export OPD_SAVE_DIR="${OPENR1_OPD1_SAVE_DIR}"
export OPD_HF_SNAPSHOT_DIR="${OPENR1_OPD1_HF_SNAPSHOT_DIR}"
export OPD_HF_SNAPSHOT_TEMPLATE="${OPD_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export OPD_FINAL_HF_DIR="${OPD_HF_SNAPSHOT_DIR}/iter_0000007"
export OPD_ROLLOUT_LOG_DIR="${OPENR1_OPD1_ROLLOUT_LOG_DIR}"
export OPD_SANITY_REPORT_DIR="${OPENR1_OPD1_SANITY_DIR}"
export OPD_SANITY_SUMMARY_DIR="${OPENR1_OPD1_SANITY_DIR}"
export OPD_TRAINED_MANIFEST="${OPD_SAVE_DIR}/opd_trained_manifest.json"
export OPD_INITIAL_LOAD_MODE=hf
export OPD_INITIAL_LOAD_DIR="${SFT_FINAL_HF_DIR}"
export OPD_REF_LOAD_DIR="${SFT_FINAL_HF_DIR}"
export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=0
export OPD_ALREADY_TRAINED_SAMPLES=0
export OPD_START_ROLLOUT_ID=0
export OPD_MANIFEST_FROM_ROLLOUT_LOGS=0
export OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=0
export OPD_RETIRE_PREVIOUS_FULL_SAVE_DIR=""
export OPD_RETIRE_PREVIOUS_HF_DIR=""

jid_opd1="$(
  sbatch --parsable "${SBATCH_FOUR[@]}" \
    --dependency=afterok:${jid_sft_eval} \
    --time="${OPENR1_WALLTIME_OPD}" \
    --cpus-per-task=32 \
    --job-name=slime-qwen3-openr1-opd1k \
    --export=ALL \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
)"
jid_opd1_eval="$(
  sbatch --parsable "${SBATCH_FOUR[@]}" \
    --dependency=afterok:${jid_opd1} \
    --time="${OPENR1_WALLTIME_EVAL}" \
    --cpus-per-task=32 \
    --job-name=slime-qwen3-openr1-opd1k-eval \
    --export=ALL,EVAL_TARGETS=opd,EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch
)"

export OPD_RUN_LABEL="${OPENR1_OPD_CONT_RUN_LABEL}"
export OPD_JSONL="${CLEANED_OPD_4K_JSONL}"
export OPD_POOL_SIZE=4096
export OPD_SIZE=4096
export OPD_SAVE_DIR="${OPENR1_OPD_CONT_SAVE_DIR}"
export OPD_HF_SNAPSHOT_DIR="${OPENR1_OPD_CONT_HF_SNAPSHOT_DIR}"
export OPD_HF_SNAPSHOT_TEMPLATE="${OPD_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export OPD_ROLLOUT_LOG_DIR="${OPENR1_OPD_CONT_ROLLOUT_LOG_DIR}"
export OPD_SANITY_REPORT_DIR="${OPENR1_OPD_CONT_SANITY_DIR}"
export OPD_SANITY_SUMMARY_DIR="${OPENR1_OPD_CONT_SANITY_DIR}"
export OPD_TRAINED_MANIFEST="${OPD_SAVE_DIR}/opd_trained_manifest.json"
export OPD_INITIAL_LOAD_MODE=megatron
export OPD_INITIAL_LOAD_DIR="${OPENR1_OPD1_SAVE_DIR}"
export OPD_REF_LOAD_DIR="${SFT_FINAL_HF_DIR}"
export OPD_PREVIOUS_RUN_LABEL="${OPENR1_OPD1_RUN_LABEL}"
export OPD_PREVIOUS_SAVE_DIR="${OPENR1_OPD1_SAVE_DIR}"
export OPD_PREVIOUS_HF_SNAPSHOT_DIR="${OPENR1_OPD1_HF_SNAPSHOT_DIR}"
export OPD_PREVIOUS_ROLLOUT_LOG_DIR="${OPENR1_OPD1_ROLLOUT_LOG_DIR}"
export OPD_PREVIOUS_TRAINED_MANIFEST="${OPENR1_OPD1_SAVE_DIR}/opd_trained_manifest.json"
export OPD_CONTINUATION_METADATA="${SPLIT_METADATA}"
export OPD_ALREADY_TRAINED_SAMPLES=1024
export OPD_MANIFEST_FROM_ROLLOUT_LOGS=1
export OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=1
export OPD_SAVE_INTERVAL=8

prior_jid="${jid_opd1_eval}"
continuation_jobs=()
for endpoint in 2048 3072 4096 5120; do
  rid=$((endpoint / 128 - 1))
  export OPD_TRAIN_SIZE="${endpoint}"
  export OPD_NUM_ROLLOUT="$((rid + 1))"
  export OPD_FINAL_ROLLOUT_ID="${rid}"
  export OPD_EFFECTIVE_TRAIN_SAMPLES="${endpoint}"
  export OPD_MILESTONE_ROLLOUT_IDS="${rid}"
  printf -v OPD_FINAL_HF_DIR "%s/iter_%07d" "${OPD_HF_SNAPSHOT_DIR}" "${rid}"
  export OPD_FINAL_HF_DIR
  export OPD_SANITY_MAX_ROLLOUT_ID="${rid}"
  export OPD_START_ROLLOUT_ID="$((rid - 7))"
  if [[ "${endpoint}" -eq 2048 ]]; then
    export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=1
    export OPD_RETIRE_PREVIOUS_FULL_SAVE_DIR="${OPENR1_OPD1_SAVE_DIR}"
    export OPD_RETIRE_PREVIOUS_HF_DIR="${OPENR1_OPD1_HF_SNAPSHOT_DIR}/iter_0000007"
  else
    export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=0
    export OPD_RETIRE_PREVIOUS_FULL_SAVE_DIR=""
    export OPD_RETIRE_PREVIOUS_HF_DIR=""
  fi
  jid_segment="$(
    sbatch --parsable "${SBATCH_FOUR[@]}" \
      --dependency=afterok:${prior_jid} \
      --time="${OPENR1_WALLTIME_OPD}" \
      --cpus-per-task=32 \
      --job-name="slime-qwen3-openr1-opd${endpoint}" \
      --export=ALL \
      examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch
  )"
  continuation_jobs+=("${jid_segment}")
  prior_jid="${jid_segment}"
done

jid_opd_final_eval="$(
  sbatch --parsable "${SBATCH_FOUR[@]}" \
    --dependency=afterok:${prior_jid} \
    --time="${OPENR1_WALLTIME_EVAL}" \
    --cpus-per-task=32 \
    --job-name=slime-qwen3-openr1-opd5120-eval \
    --export=ALL,EVAL_TARGETS=opd,EVAL_SKIP_COMPLETED=1 \
    examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch
)"
jid_report="$(
  sbatch --parsable "${SBATCH_ONE[@]}" \
    --dependency=afterok:${jid_opd_final_eval} \
    --time="${OPENR1_WALLTIME_REPORT}" \
    --cpus-per-task=8 \
    --job-name=slime-qwen3-openr1-report \
    --export=ALL,EVAL_TARGETS=report \
    examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch
)"

{
  echo "experiment=${OPENR1_EXPERIMENT_LABEL}"
  echo "submit_time=$(date --iso-8601=seconds)"
  echo "implementation_commit=$(git rev-parse HEAD)"
  echo "data=${jid_data}"
  echo "resume_from_sft_checkpoint=${OPENR1_RESUME_FROM_SFT_CHECKPOINT}"
  if [[ "${OPENR1_RESUME_FROM_SFT_CHECKPOINT}" == "1" ]]; then
    echo "resume_sft_checkpoint_iteration=${resume_sft_checkpoint_iteration}"
  fi
  echo "sft_train=${jid_sft}"
  echo "sft_eval=${jid_sft_eval}"
  echo "opd_001024_train=${jid_opd1}"
  echo "opd_001024_eval=${jid_opd1_eval}"
  echo "opd_002048_train=${continuation_jobs[0]}"
  echo "opd_003072_train=${continuation_jobs[1]}"
  echo "opd_004096_train=${continuation_jobs[2]}"
  echo "opd_005120_train=${continuation_jobs[3]}"
  echo "opd_005120_eval=${jid_opd_final_eval}"
  echo "report=${jid_report}"
  echo "dependency_policy=strict_serial_afterok"
  echo "max_gpu_per_job=4"
  echo "data_dir=${OPENR1_DATA_DIR}"
  echo "sft_max_tokens_per_gpu=${SFT_MAX_TOKENS_PER_GPU}"
  echo "sft_log_probs_chunk_size=${SFT_LOG_PROBS_CHUNK_SIZE}"
  echo "sft_optimizer_cpu_offload=${SFT_OPTIMIZER_CPU_OFFLOAD}"
  echo "sft_recompute_loss_function=${SFT_RECOMPUTE_LOSS_FUNCTION}"
  echo "vllm_eval_install_spec=${VLLM_EVAL_INSTALL_SPEC}"
  echo "vllm_eval_num_gpus=${VLLM_EVAL_NUM_GPUS}"
  echo "vllm_eval_max_num_seqs=${VLLM_EVAL_MAX_NUM_SEQS}"
  echo "vllm_eval_max_num_batched_tokens=${VLLM_EVAL_MAX_NUM_BATCHED_TOKENS}"
  echo "vllm_eval_gpu_memory_utilization=${VLLM_EVAL_GPU_MEMORY_UTILIZATION}"
  echo "vllm_eval_kv_cache_dtype=${VLLM_EVAL_KV_CACHE_DTYPE}"
  echo "vllm_eval_async_scheduling=${VLLM_EVAL_ASYNC_SCHEDULING}"
  echo "vllm_eval_chunked_prefill=${VLLM_EVAL_ENABLE_CHUNKED_PREFILL}"
  echo "vllm_eval_prefix_caching=${VLLM_EVAL_ENABLE_PREFIX_CACHING}"
  echo "vllm_eval_speculative=${VLLM_EVAL_SPECULATIVE_METHOD}:${VLLM_EVAL_NUM_SPECULATIVE_TOKENS}:${VLLM_EVAL_PROMPT_LOOKUP_MIN}:${VLLM_EVAL_PROMPT_LOOKUP_MAX}"
  echo "sft_save_dir=${SFT_SAVE_DIR}"
  echo "sft_hf_snapshot_dir=${SFT_HF_SNAPSHOT_DIR}"
  echo "opd1_save_dir=${OPENR1_OPD1_SAVE_DIR}"
  echo "opd1_hf_snapshot_dir=${OPENR1_OPD1_HF_SNAPSHOT_DIR}"
  echo "opd_cont_save_dir=${OPENR1_OPD_CONT_SAVE_DIR}"
  echo "opd_cont_hf_snapshot_dir=${OPENR1_OPD_CONT_HF_SNAPSHOT_DIR}"
  echo "base_eval_output=${BASE_EVAL_OUTPUT_DIR}"
  echo "sft_eval_output=${SFT_EVAL_OUTPUT_DIR}"
  echo "opd_eval_output=${OPD_EVAL_OUTPUT_DIR}"
  echo "combined_output=${COMBINED_EVAL_OUTPUT_DIR}"
  echo "walltimes=${OPENR1_WALLTIME_DATA},${OPENR1_WALLTIME_SFT},${OPENR1_WALLTIME_EVAL},${OPENR1_WALLTIME_OPD},${OPENR1_WALLTIME_REPORT}"
} | tee "${submit_log}"

echo "Submitted serialized OpenR1 chain; metadata: ${submit_log}"
