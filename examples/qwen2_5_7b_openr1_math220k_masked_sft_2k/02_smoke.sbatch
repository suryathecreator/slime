#!/bin/bash
#SBATCH --job-name=q25-7b-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_openr1_math220k_masked_sft_2k"
source "${SCRIPT_DIR}/env.sh"
export TMPDIR="/tmp/${USER:-suryadv}/q25-smoke-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${OUTPUT_ROOT}/smoke" "${SLURM_LOG_DIR}"
"${SCRIPT_DIR}/container_exec.sh" python3 \
  "${SCRIPT_DIR}/../qwen3_8b_openr1_math220k_masked_sft_40k/check_train_tmpdir.py"
export SFT_PARQUET="${OUTPUT_ROOT}/smoke/correct_only_first_200.jsonl"
[[ ! -e "${SFT_PARQUET}" ]] || { echo "Refusing to overwrite ${SFT_PARQUET}" >&2; exit 1; }
sed -n '1,200p' "${CORRECT_ONLY_JSONL}" >"${SFT_PARQUET}"
[[ "$(wc -l <"${SFT_PARQUET}")" == "200" ]]
export SFT_SAVE_DIR="${OUTPUT_ROOT}/smoke/full_state"
export SFT_HF_SNAPSHOT_DIR="${OUTPUT_ROOT}/smoke/weights"
export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000000"
export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/iter_0000000"
export SFT_DETAILS_DIR="${OUTPUT_ROOT}/smoke/details"
export SFT_NUM_ROLLOUT=1
export SFT_FINAL_ROLLOUT_ID=0
export SFT_MILESTONE_ROLLOUT_IDS=0
export SFT_SAVE_INTERVAL=1
export SFT_SKIP_LOG_REDIRECT=0
bash "${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
test -f "${SFT_FINAL_HF_DIR}/config.json"
echo "QWEN25_SMOKE_COMPLETE at=$(date -Is)"
