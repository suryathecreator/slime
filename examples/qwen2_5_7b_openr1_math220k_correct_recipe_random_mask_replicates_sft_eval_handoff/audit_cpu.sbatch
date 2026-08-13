#!/usr/bin/env bash
#SBATCH --job-name=q25r-m500-audit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen2_5_7b_openr1_math220k_correct_recipe_random_mask_replicates_sft_eval_handoff"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$HANDOFF/runtime_python.sh}"
cd "$REPO_ROOT"
"$RUNTIME_PYTHON" "$HANDOFF/math500_eval.py" --repo-root "$REPO_ROOT" audit
echo "Q25R_MATH500_AUDIT_COMPLETE job=$SLURM_JOB_ID at=$(date -Is)"
