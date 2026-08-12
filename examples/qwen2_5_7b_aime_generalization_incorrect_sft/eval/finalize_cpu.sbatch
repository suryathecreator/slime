#!/usr/bin/env bash
#SBATCH --job-name=q25-aime-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
EVAL_DIR="${REPO_ROOT}/examples/qwen2_5_7b_aime_generalization_incorrect_sft/eval"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
: "${EVAL_EXPERIMENT_ROOT:?Set EVAL_EXPERIMENT_ROOT}"
: "${EVAL_TARGET:?Set EVAL_TARGET}"
: "${EVAL_GIT_COMMIT:?Set EVAL_GIT_COMMIT}"
: "${AIME_EXPECTED_GIT_COMMIT:?Set AIME_EXPECTED_GIT_COMMIT}"
[[ "${EVAL_GIT_COMMIT}" == "${AIME_EXPECTED_GIT_COMMIT}" ]] || {
  echo "Scorer label and runtime commit pin differ." >&2
  exit 2
}
"${RUNTIME_PYTHON}" "${EVAL_DIR}/../runtime_repo_gate.py" \
  --repo-root "${REPO_ROOT}" --expected-commit "${AIME_EXPECTED_GIT_COMMIT}"
cd "${REPO_ROOT}"
"${RUNTIME_PYTHON}" "${EVAL_DIR}/aime_eval.py" --repo-root "${REPO_ROOT}" --experiment-root "${EVAL_EXPERIMENT_ROOT}" merge-target \
  --target "${EVAL_TARGET}" --scorer-commit "${EVAL_GIT_COMMIT}"
echo "AIME_EVAL_TARGET_FINALIZE_COMPLETE target=${EVAL_TARGET} job=${SLURM_JOB_ID} at=$(date -Is)"
