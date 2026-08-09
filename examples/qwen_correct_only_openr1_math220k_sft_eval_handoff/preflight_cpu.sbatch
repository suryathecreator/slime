#!/usr/bin/env bash
#SBATCH --job-name=qco2-m500-preflight
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --requeue

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen_correct_only_openr1_math220k_sft_eval_handoff"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$HANDOFF/runtime_python.sh}"
CONTROL="$HANDOFF/math500_eval.py"
CONTROL_ROOT="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field control)"
VERIFY_STATUS="$CONTROL_ROOT/checkpoint_verification.json"
cd "$REPO_ROOT"

"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" verify-checkpoints --full --status "$VERIFY_STATUS"
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" validate-gold
"$RUNTIME_PYTHON" -m pytest -q \
  "$HANDOFF/tests/test_handoff.py" \
  examples/qwen3_8b_32b_full_repro/tests/test_math500_last_boxed_scorer_v6.py
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" mark-preflight \
  --job-id "${SLURM_JOB_ID:?}" --verification "$VERIFY_STATUS"
echo "QCO_MATH500_PREFLIGHT_COMPLETE job=$SLURM_JOB_ID at=$(date -Is)"
