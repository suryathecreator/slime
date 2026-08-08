#!/bin/bash
#SBATCH --job-name=qco-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=16:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k"
SHARED_RUNNER="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
HANDOFF_TOOL="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
source "${SCRIPT_DIR}/env.sh"
: "${RUN_KEY:?Set RUN_KEY}"
: "${SLURM_JOB_ID:?Training must run under Slurm}"
resolve_run "${RUN_KEY}"
export TMPDIR="/tmp/${USER:-suryadv}/qco-train-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
export SFT_SAVE_DIR="${OUTPUT_ROOT}/training/${RUN_KEY}/correct_only/full_state"
export SFT_HF_SNAPSHOT_DIR="${OUTPUT_ROOT}/training/${RUN_KEY}/correct_only/weights"
export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000124"
export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/iter_0000124"
export SFT_DETAILS_DIR="${OUTPUT_ROOT}/training/${RUN_KEY}/correct_only/details"
export CHECKPOINT_REPORT_DIR="${OUTPUT_ROOT}/training/${RUN_KEY}/correct_only/checkpoint_reports"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${SLURM_LOG_DIR}" \
  "${OUTPUT_ROOT}/training/${RUN_KEY}/correct_only"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_dataset.py" \
  --data "${SFT_PARQUET}" --run-key "${RUN_KEY}" --stats "${PREP_STATS_JSON}"
test -f "${DATA_ROOT}/schedule_audits/${RUN_KEY}.json"
echo "TRAIN_CORRECT_ONLY run=${RUN_KEY} initialization=${STUDENT_TORCH_DIST_DIR} full_sft=1"
bash "${SHARED_RUNNER}"
python3 "${HANDOFF_TOOL}" record \
  --checkpoint "${SFT_FINAL_HF_DIR}" \
  --output "${HANDOFF_ROOT}/checkpoints/${RUN_KEY}.json" \
  --family 2k --variant "${RUN_KEY}.correct_only" \
  --final-iteration "${SFT_FINAL_ROLLOUT_ID}" \
  --contract-hash "${CONTRACT_HASH}" --repo-root "${SLIME_REPO_ROOT}" --full-sft
echo "TRAIN_CORRECT_ONLY_COMPLETE run=${RUN_KEY} at=$(date -Is)"
