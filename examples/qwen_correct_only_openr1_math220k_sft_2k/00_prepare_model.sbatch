#!/bin/bash
#SBATCH --job-name=qco-model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=08:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen_correct_only_openr1_math220k_sft_2k"
HANDOFF_TOOL="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff/checkpoint_manifest.py"
source "${SCRIPT_DIR}/env.sh"
: "${MODEL_KEY:?Set MODEL_KEY}"
: "${SLURM_JOB_ID:?Model preparation must run under Slurm}"
resolve_model "${MODEL_KEY}"
export TMPDIR="/tmp/${USER:-suryadv}/qco-model-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
mkdir -p "${MODEL_ROOT}" "${SLURM_LOG_DIR}" "${TMPDIR}" "${RAY_TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

"${SCRIPT_DIR}/container_exec.sh" python3 \
  "${SLURM_REPO_ROOT:-${SLIME_REPO_ROOT}}/examples/qwen2_5_7b_openr1_math220k_masked_sft_2k/check_checkpoint_tmpdir.py"
"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
hf download "${STUDENT_HF_REPO}" \
  --revision "${STUDENT_HF_REVISION}" \
  --local-dir "${STUDENT_HF_DIR}"
python3 examples/qwen_correct_only_openr1_math220k_sft_2k/validate_base_model.py \
  --model-key "$1" --model "${STUDENT_HF_DIR}"
if [[ ! -f "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]]; then
  source "${STUDENT_MODEL_ARGS_SCRIPT}"
  mkdir -p "${STUDENT_TORCH_DIST_DIR}"
  PYTHONPATH="/root/Megatron-LM:${SLIME_REPO_ROOT}:${PYTHONPATH:-}" \
  torchrun --nproc_per_node=4 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${STUDENT_HF_DIR}" \
    --save "${STUDENT_TORCH_DIST_DIR}"
fi
test -f "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt"
' bash "${MODEL_KEY}"
python3 "${HANDOFF_TOOL}" record \
  --checkpoint "${STUDENT_HF_DIR}" \
  --output "${HANDOFF_ROOT}/checkpoints/base_${MODEL_KEY}.json" \
  --family shared --variant "base_${MODEL_KEY}" \
  --contract-hash "${CONTRACT_HASH}" --repo-root "${SLIME_REPO_ROOT}" --no-full-sft
echo "MODEL_PREP_COMPLETE model=${MODEL_KEY} at=$(date -Is)"
