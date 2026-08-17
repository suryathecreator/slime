#!/usr/bin/env bash
#SBATCH --job-name=q25-nous-aime-base-reuse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
EVAL_DIR="${REPO_ROOT}/examples/qwen2_5_7b_nous_aime_mixed_outcome_sft/eval"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-${EVAL_DIR}/runtime_python.sh}"
: "${EVAL_EXPERIMENT_ROOT:?Set EVAL_EXPERIMENT_ROOT}"
: "${EVAL_BASE_SOURCE_EXPERIMENT_ROOT:?Set EVAL_BASE_SOURCE_EXPERIMENT_ROOT}"
: "${EVAL_BASE_SOURCE_RESULT_ROOT:?Set EVAL_BASE_SOURCE_RESULT_ROOT}"
: "${AIME_EXPECTED_GIT_COMMIT:?Set AIME_EXPECTED_GIT_COMMIT}"
"${RUNTIME_PYTHON}" "${EVAL_DIR}/../../qwen2_5_7b_aime_generalization_incorrect_sft/runtime_repo_gate.py" \
  --repo-root "${REPO_ROOT}" --expected-commit "${AIME_EXPECTED_GIT_COMMIT}"

cd "${REPO_ROOT}"
"${RUNTIME_PYTHON}" "${EVAL_DIR}/mixed_aime_eval.py" --repo-root "${REPO_ROOT}" \
  --experiment-root "${EVAL_EXPERIMENT_ROOT}" reuse-base \
  --source-experiment-root "${EVAL_BASE_SOURCE_EXPERIMENT_ROOT}" \
  --source-result-root "${EVAL_BASE_SOURCE_RESULT_ROOT}"
echo "MIXED_AIME_BASE_REUSE_JOB_COMPLETE job=${SLURM_JOB_ID:?} at=$(date -Is)"
