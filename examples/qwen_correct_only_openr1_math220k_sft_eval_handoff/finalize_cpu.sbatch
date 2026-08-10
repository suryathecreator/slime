#!/usr/bin/env bash
#SBATCH --job-name=qco3-m500-finalize
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen_correct_only_openr1_math220k_sft_eval_handoff"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$HANDOFF/runtime_python.sh}"
: "${EVAL_TARGET:?Set EVAL_TARGET}"; : "${EVAL_GIT_COMMIT:?Set EVAL_GIT_COMMIT}"
cd "$REPO_ROOT"
"$RUNTIME_PYTHON" "$HANDOFF/math500_eval.py" --repo-root "$REPO_ROOT" merge \
  --target "$EVAL_TARGET" --scorer-commit "$EVAL_GIT_COMMIT"
echo "QCO3_MATH500_FINALIZE_COMPLETE target=$EVAL_TARGET job=$SLURM_JOB_ID at=$(date -Is)"
