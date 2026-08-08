#!/bin/bash
#SBATCH --job-name=qco-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=240G
#SBATCH --time=16:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k"
source "${SCRIPT_DIR}/env.sh"
: "${SLURM_JOB_ID:?Data preparation must run under Slurm}"
REUSE_PREPARED_DATA="${REUSE_PREPARED_DATA:-0}"
[[ "${REUSE_PREPARED_DATA}" == "0" || "${REUSE_PREPARED_DATA}" == "1" ]] || {
  echo "REUSE_PREPARED_DATA must be 0 or 1, got ${REUSE_PREPARED_DATA}" >&2
  exit 2
}
export TMPDIR="/tmp/${USER:-suryadv}/qco-data-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=1
mkdir -p "${DATA_ROOT}" "${SLURM_LOG_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

for model_key in qwen2_5_3b qwen2_5_7b qwen3_4b qwen3_8b; do
  resolve_model "${model_key}"
  test -f "${STUDENT_HF_DIR}/config.json"
done
if [[ "${REUSE_PREPARED_DATA}" == "1" ]]; then
  test -f "${PREP_STATS_JSON}"
  echo "DATA_PREP_REUSE_EXISTING stats=${PREP_STATS_JSON}"
else
  "${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/prepare_data.py" \
    --cache-dir "${HF_HOME}" --data-root "${DATA_ROOT}" --model-root "${MODEL_ROOT}" \
    --rows "${CORRECT_ONLY_ROWS}" --seed "${DATA_SEED}"
fi

mkdir -p "${DATA_ROOT}/schedule_audits"
for run_key in \
  qwen2_5_3b_8k qwen2_5_3b_16k qwen2_5_7b_8k qwen3_4b_8k qwen3_8b_8k; do
  resolve_run "${run_key}"
  dp_size=$((SFT_ACTOR_GPUS / SFT_TENSOR_MODEL_PARALLEL_SIZE))
  "${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_dataset.py" \
    --data "${SFT_PARQUET}" --run-key "${RUN_KEY}" --stats "${PREP_STATS_JSON}"
  "${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_schedule.py" \
    --data "${SFT_PARQUET}" \
    --output "${DATA_ROOT}/schedule_audits/${RUN_KEY}.json" \
    --dp-size "${dp_size}" --max-tokens-per-gpu "${SFT_MAX_TOKENS_PER_GPU}" \
    --seed "${DATA_SEED}"
done
echo "DATA_PREP_COMPLETE at=$(date -Is)"
