#!/bin/bash
#SBATCH --job-name=q8b32b-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_32b_full_repro"
source "${SCRIPT_DIR}/env.sh"
export TMPDIR="/tmp/${USER:-suryadv}/q8b32b-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
unset LD_LIBRARY_PATH
mkdir -p "${SLURM_LOG_DIR}" "${MANIFEST_ROOT}" "${TMPDIR}" "${RAY_TMPDIR}" "${WANDB_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

cd "${SLIME_REPO_ROOT}"
python3 "${SCRIPT_DIR}/validate_config.py" --require-pinned-commit

for path in \
  "${SLIME_SIF}" \
  "${STUDENT_HF_DIR}/config.json" \
  "${TEACHER_HF_DIR}/config.json" \
  "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" \
  "${STRICT_LANGUAGE_SITE}/lingua" \
  "${MATH_VERIFY_SITE}/math_verify" \
  "${VLLM_EVAL_SITE}/vllm"; do
  [[ -e "${path}" ]] || { echo "Missing pinned prerequisite: ${path}" >&2; exit 1; }
done

PYTHONPATH="${STRICT_LANGUAGE_SITE}:${MATH_VERIFY_SITE}:${PYTHONPATH:-}" \
  "${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/capture_environment.py"

VLLM_TARGET_LIBS="$(find "${VLLM_EVAL_SITE}" -path '*/nvidia/*/lib' -type d -printf '%p:' 2>/dev/null || true)"
LD_LIBRARY_PATH="${VLLM_TARGET_LIBS}" \
PYTHONPATH="${VLLM_EVAL_SITE}:${MATH_VERIFY_SITE}:${PYTHONPATH:-}" \
SLIME_VLLM_PATCH_SITE=0 SLIME_SGLANG_PATCH_SITE=0 \
  "${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/capture_eval_environment.py"

PYTHONPATH="${STRICT_LANGUAGE_SITE}:${MATH_VERIFY_SITE}:${PYTHONPATH:-}" \
  "${SCRIPT_DIR}/container_exec.sh" python3 -m pytest -q \
    "${SCRIPT_DIR}/tests"

echo "PRELIGHT_COMPLETE contract=${CONTRACT_HASH} commit=$(git rev-parse HEAD)"
