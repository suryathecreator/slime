#!/bin/bash
#SBATCH --job-name=slime-qwen3-strict-dispatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/strict_en_v2_env.sh"

mkdir -p "${SLURM_LOG_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi
cd "${SLIME_REPO_ROOT}"

metadata_value() {
  python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["selection"][sys.argv[2]])' "${SPLIT_METADATA}" "$1"
}

test -f "${STRICT_EN_DATASET_DIR}/_SUCCESS"
test -f "${SPLIT_METADATA}"
test -f "${BASE_EVAL_OUTPUT_DIR}/base/summary.json"

export SFT_SIZE="$(metadata_value sft_effective)"
export OPD_POOL_SIZE="$(metadata_value opd_reserved)"
export OPD_EFFECTIVE_FINAL="$(metadata_value opd_effective)"
export CLEANED_OPD_NEXT_SIZE="$(metadata_value opd_continuation)"
export SFT_NUM_ROLLOUT="$((SFT_SIZE / SFT_ROLLOUT_BATCH_SIZE))"
export SFT_FINAL_ROLLOUT_ID="$((SFT_NUM_ROLLOUT - 1))"
export SFT_SAVE_INTERVAL=20
export SFT_MILESTONE_ROLLOUT_IDS=""
for ((rid=19; rid<SFT_FINAL_ROLLOUT_ID; rid+=20)); do
  SFT_MILESTONE_ROLLOUT_IDS+="${rid} "
done
export SFT_MILESTONE_ROLLOUT_IDS+="${SFT_FINAL_ROLLOUT_ID}"

tag="${STRICT_EN_EXPERIMENT_LABEL}"
printf -v sft_final_tag 'iter_%07d' "${SFT_FINAL_ROLLOUT_ID}"
export SFT_SAVE_DIR="${OUTPUT_ROOT}/qwen3_8b_${tag}_sft_full_optim"
export SFT_HF_SNAPSHOT_DIR="${OUTPUT_ROOT}/qwen3_8b_${tag}_sft_snapshots"
export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/${sft_final_tag}"
export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/${sft_final_tag}"
export SFT_DETAILS_DIR="${OUTPUT_ROOT}/${tag}_sft_details"
export SFT_EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/math500_eval_${tag}_sft_vllm"
export OPD_EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/math500_eval_${tag}_opd_vllm"
export COMBINED_EVAL_OUTPUT_DIR="${OUTPUT_ROOT}/math500_eval_${tag}_combined"
export CHECKPOINT_REPORT_DIR="${OUTPUT_ROOT}/checkpoint_reports_${tag}"
export REPORT_OPD_X_OFFSET_SAMPLES="${SFT_SIZE}"
export REPORT_OPD_LABEL_PREFIX="SFT+"

SBATCH_FOUR=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:4)
SBATCH_ONE=(-A "${ACCOUNT}" -p "${PARTITION}" --qos "${QOS}" --gres gpu:h200:1)

jid_sft="$(sbatch --parsable "${SBATCH_FOUR[@]}" --time=24:00:00 --job-name=slime-qwen3-strict-sft --cpus-per-task=32 --export=ALL examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch)"
jid_sft_eval="$(sbatch --parsable "${SBATCH_FOUR[@]}" --dependency=afterok:${jid_sft} --time=24:00:00 --job-name=slime-qwen3-strict-sft-eval --cpus-per-task=32 --export=ALL,EVAL_TARGETS=sft,EVAL_SKIP_COMPLETED=1 examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch)"

opd1_label="${tag}_opd1k"
opd1_save="${OUTPUT_ROOT}/qwen3_8b_${opd1_label}_full_optim"
opd1_hf="${OUTPUT_ROOT}/qwen3_8b_${opd1_label}_snapshots"
opd1_logs="${OUTPUT_ROOT}/${opd1_label}_rollout_logs"
opd1_sanity="${OUTPUT_ROOT}/${opd1_label}_sanity"
export OPD_RUN_LABEL="${opd1_label}"
export OPD_JSONL="${CLEANED_OPD_1K_JSONL}"
export OPD_POOL_SIZE="${CLEANED_OPD_FIRST_SIZE}"
export OPD_SIZE="${CLEANED_OPD_FIRST_SIZE}"
export OPD_TRAIN_SIZE="${CLEANED_OPD_FIRST_SIZE}"
export OPD_NUM_ROLLOUT="$((CLEANED_OPD_FIRST_SIZE / OPD_ROLLOUT_BATCH_SIZE))"
export OPD_FINAL_ROLLOUT_ID="$((OPD_NUM_ROLLOUT - 1))"
export OPD_EFFECTIVE_TRAIN_SAMPLES="${CLEANED_OPD_FIRST_SIZE}"
export OPD_MILESTONE_ROLLOUT_IDS="${OPD_FINAL_ROLLOUT_ID}"
export OPD_SAVE_INTERVAL=8
export OPD_SAVE_DIR="${opd1_save}"
export OPD_HF_SNAPSHOT_DIR="${opd1_hf}"
export OPD_HF_SNAPSHOT_TEMPLATE="${opd1_hf}/iter_{rollout_id:07d}"
printf -v OPD_FINAL_HF_DIR '%s/iter_%07d' "${opd1_hf}" "${OPD_FINAL_ROLLOUT_ID}"
export OPD_FINAL_HF_DIR
export OPD_ROLLOUT_LOG_DIR="${opd1_logs}"
export OPD_SANITY_REPORT_DIR="${opd1_sanity}"
export OPD_SANITY_SUMMARY_DIR="${opd1_sanity}"
export OPD_TRAINED_MANIFEST="${opd1_save}/opd_trained_manifest.json"
export OPD_INITIAL_LOAD_MODE=hf
export OPD_INITIAL_LOAD_DIR="${SFT_FINAL_HF_DIR}"
export OPD_REF_LOAD_DIR="${SFT_FINAL_HF_DIR}"
export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=0
export OPD_ALREADY_TRAINED_SAMPLES=0
export OPD_START_ROLLOUT_ID=0
export OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=0

jid_opd1="$(sbatch --parsable "${SBATCH_FOUR[@]}" --dependency=afterok:${jid_sft_eval} --time=24:00:00 --job-name=slime-qwen3-strict-opd1k --cpus-per-task=32 --export=ALL examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch)"
jid_opd1_eval="$(sbatch --parsable "${SBATCH_FOUR[@]}" --dependency=afterok:${jid_opd1} --time=24:00:00 --job-name=slime-qwen3-strict-opd1k-eval --cpus-per-task=32 --export=ALL,EVAL_TARGETS=opd,EVAL_SKIP_COMPLETED=1 examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch)"

opdf_label="${tag}_opd_final"
opdf_save="${OUTPUT_ROOT}/qwen3_8b_${opdf_label}_full_optim"
opdf_hf="${OUTPUT_ROOT}/qwen3_8b_${opdf_label}_snapshots"
opdf_logs="${OUTPUT_ROOT}/${opdf_label}_rollout_logs"
opdf_sanity="${OUTPUT_ROOT}/${opdf_label}_sanity"
export OPD_RUN_LABEL="${opdf_label}"
export OPD_JSONL="${CLEANED_OPD_4K_JSONL}"
export OPD_POOL_SIZE="${CLEANED_OPD_NEXT_SIZE}"
export OPD_SIZE="${CLEANED_OPD_NEXT_SIZE}"
export OPD_TRAIN_SIZE="${OPD_EFFECTIVE_FINAL}"
export OPD_NUM_ROLLOUT="$((OPD_EFFECTIVE_FINAL / OPD_ROLLOUT_BATCH_SIZE))"
export OPD_FINAL_ROLLOUT_ID="$((OPD_NUM_ROLLOUT - 1))"
export OPD_EFFECTIVE_TRAIN_SAMPLES="${OPD_EFFECTIVE_FINAL}"
export OPD_MILESTONE_ROLLOUT_IDS="${OPD_FINAL_ROLLOUT_ID}"
export OPD_SAVE_INTERVAL=8
export OPD_SAVE_DIR="${opdf_save}"
export OPD_HF_SNAPSHOT_DIR="${opdf_hf}"
export OPD_HF_SNAPSHOT_TEMPLATE="${opdf_hf}/iter_{rollout_id:07d}"
printf -v OPD_FINAL_HF_DIR '%s/iter_%07d' "${opdf_hf}" "${OPD_FINAL_ROLLOUT_ID}"
export OPD_FINAL_HF_DIR
export OPD_ROLLOUT_LOG_DIR="${opdf_logs}"
export OPD_SANITY_REPORT_DIR="${opdf_sanity}"
export OPD_SANITY_SUMMARY_DIR="${opdf_sanity}"
export OPD_TRAINED_MANIFEST="${opdf_save}/opd_trained_manifest.json"
export OPD_INITIAL_LOAD_MODE=megatron
export OPD_INITIAL_LOAD_DIR="${opd1_save}"
export OPD_REF_LOAD_DIR="${SFT_FINAL_HF_DIR}"
export OPD_PREVIOUS_RUN_LABEL="${opd1_label}"
export OPD_PREVIOUS_SAVE_DIR="${opd1_save}"
export OPD_PREVIOUS_HF_SNAPSHOT_DIR="${opd1_hf}"
export OPD_PREVIOUS_ROLLOUT_LOG_DIR="${opd1_logs}"
export OPD_PREVIOUS_TRAINED_MANIFEST="${opd1_save}/opd_trained_manifest.json"
export OPD_CONTINUATION_METADATA="${SPLIT_METADATA}"
export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=1
export OPD_ALREADY_TRAINED_SAMPLES="${CLEANED_OPD_FIRST_SIZE}"
export OPD_START_ROLLOUT_ID=8
export OPD_MANIFEST_FROM_ROLLOUT_LOGS=1
export OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=1

jid_opdf="$(sbatch --parsable "${SBATCH_FOUR[@]}" --dependency=afterok:${jid_opd1_eval} --time=24:00:00 --job-name=slime-qwen3-strict-opdfinal --cpus-per-task=32 --export=ALL examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch)"
jid_opdf_eval="$(sbatch --parsable "${SBATCH_FOUR[@]}" --dependency=afterok:${jid_opdf} --time=24:00:00 --job-name=slime-qwen3-strict-opdfinal-eval --cpus-per-task=32 --export=ALL,EVAL_TARGETS=opd,EVAL_SKIP_COMPLETED=1 examples/qwen3_8b_opd_tillicum/06_eval_math500_vllm.sbatch)"
jid_report="$(sbatch --parsable "${SBATCH_ONE[@]}" --dependency=afterok:${jid_opdf_eval} --time=06:00:00 --job-name=slime-qwen3-strict-report --cpus-per-task=8 --export=ALL,EVAL_TARGETS=report examples/qwen3_8b_opd_tillicum/07_report_math500.sbatch)"

dispatch_log="${SLURM_LOG_DIR}/strict_en_v2_dispatched_jobs_${SLURM_JOB_ID:-manual}.txt"
{
  echo "dispatch_time=$(date --iso-8601=seconds)"
  echo "implementation_commit=${STRICT_EN_IMPLEMENTATION_COMMIT:-unknown}"
  echo "sft_size=${SFT_SIZE}"
  echo "opd_effective_final=${OPD_EFFECTIVE_FINAL}"
  echo "sft_train=${jid_sft}"
  echo "sft_eval=${jid_sft_eval}"
  echo "opd1_train=${jid_opd1}"
  echo "opd1_eval=${jid_opd1_eval}"
  echo "opd_final_train=${jid_opdf}"
  echo "opd_final_eval=${jid_opdf_eval}"
  echo "report=${jid_report}"
  echo "dependency_policy=strict_afterok_serial"
  echo "max_h200_at_once=4"
  echo "base_eval=reused_165035"
  echo "metadata=${SPLIT_METADATA}"
} | tee "${dispatch_log}"
