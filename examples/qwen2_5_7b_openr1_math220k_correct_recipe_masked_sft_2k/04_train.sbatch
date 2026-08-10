#!/bin/bash
#SBATCH --job-name=q25crm-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_2k"
CONTAINER_EXEC="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k/container_exec.sh"
MASKED_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_openr1_math220k_masked_sft_2k"
SHARED_RUNNER="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
HANDOFF_TOOL="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
source "${SCRIPT_DIR}/env.sh"
: "${SFT_VARIANT:?Set SFT_VARIANT}"
: "${SLURM_JOB_ID:?Training must run under Slurm}"
export SFT_PARQUET
SFT_PARQUET="$(variant_data_path "${SFT_VARIANT}")"
export TMPDIR="/tmp/${USER:-suryadv}/q25crm-train-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
export TRAINING_ROOT="${OUTPUT_ROOT}/training/${SFT_VARIANT}"
export SFT_SAVE_DIR="${TRAINING_ROOT}/full_state"
export SFT_HF_SNAPSHOT_DIR="${TRAINING_ROOT}/weights"
export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000124"
export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/iter_0000124"
export SFT_DETAILS_DIR="${TRAINING_ROOT}/details"
export CHECKPOINT_REPORT_DIR="${TRAINING_ROOT}/checkpoint_reports"
export SFT_WANDB_GROUP="qwen2.5-7b-correct-recipe-${SFT_VARIANT}"
export TRAINED_MANIFEST="${HANDOFF_ROOT}/checkpoints/${SFT_VARIANT}.json"
for target in "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${SFT_DETAILS_DIR}" "${CHECKPOINT_REPORT_DIR}" "${TRAINED_MANIFEST}"; do
  [[ ! -e "${target}" ]] || { echo "Refusing to overwrite ${target}" >&2; exit 1; }
done
mkdir -p "${TRAINING_ROOT}" "${TMPDIR}" "${RAY_TMPDIR}" "${SLURM_LOG_DIR}" "${HANDOFF_ROOT}/checkpoints"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
mixed_hash="$(jq -r '.ordered_mixed_trace_ids_sha256' "${PREP_STATS_JSON}")"
"${CONTAINER_EXEC}" python3 "${MASKED_DIR}/validate_dataset.py" \
  --data "${SFT_PARQUET}" --variant "${SFT_VARIANT}" \
  --inventory "${TOKENIZER_INVENTORY_JSON}" --expected-trace-hash "${mixed_hash}" \
  --max-sequence-length "${MAX_SEQUENCE_LENGTH}"
test -f "${SCHEDULE_AUDIT_JSON}"
echo "TRAIN_CORRECT_RECIPE variant=${SFT_VARIANT} updates=${SFT_NUM_ROLLOUT} full_sft=1"
bash "${SHARED_RUNNER}"
test "$(cat "${SFT_SAVE_DIR}/latest_checkpointed_iteration.txt")" = 124
test -f "${SFT_FINAL_HF_DIR}/config.json"
python3 "${HANDOFF_TOOL}" record \
  --checkpoint "${SFT_FINAL_HF_DIR}" --output "${TRAINED_MANIFEST}" \
  --family 2k --variant "qwen2_5_7b_8k.${SFT_VARIANT}" \
  --final-iteration "${SFT_FINAL_ROLLOUT_ID}" --contract-hash "${CONTRACT_HASH}" \
  --repo-root "${SLIME_REPO_ROOT}" --full-sft
echo "TRAIN_CORRECT_RECIPE_COMPLETE variant=${SFT_VARIANT} at=$(date -Is)"
