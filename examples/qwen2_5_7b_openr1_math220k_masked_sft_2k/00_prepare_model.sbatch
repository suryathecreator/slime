#!/bin/bash
#SBATCH --job-name=q25-7b-model
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=04:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen2_5_7b_openr1_math220k_masked_sft_2k"
HANDOFF_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff"
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${MODEL_ROOT}" "${SLURM_LOG_DIR}" "${TMPDIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
export HF_HUB_OFFLINE=0
export TRANSFORMERS_OFFLINE=0
if [[ ! -f "${STUDENT_HF_DIR}/config.json" ]]; then
  hf download "${STUDENT_HF_REPO}" \
    --revision "${STUDENT_HF_REVISION}" \
    --local-dir "${STUDENT_HF_DIR}"
fi
python3 examples/qwen2_5_7b_openr1_math220k_masked_sft_2k/validate_base_model.py \
  --model "${STUDENT_HF_DIR}"
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
'
python3 "${HANDOFF_DIR}/checkpoint_manifest.py" record \
  --checkpoint "${STUDENT_HF_DIR}" \
  --output "${EXPERIMENT_ROOT}/handoff/checkpoints/base_7b.json" \
  --family shared \
  --variant base_7b \
  --contract-hash "${CONTRACT_HASH}" \
  --repo-root "${SLIME_REPO_ROOT}" \
  --no-full-sft
echo "QWEN25_MODEL_PREP_COMPLETE at=$(date -Is)"
