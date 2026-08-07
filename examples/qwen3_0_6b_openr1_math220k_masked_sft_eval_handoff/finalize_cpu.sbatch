#!/usr/bin/env bash
#SBATCH --job-name=q06-m500-v5-finalize
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_0_6b_openr1_math220k_masked_sft_eval_handoff"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
: "${EVAL_VARIANT:?Set EVAL_VARIANT}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
cd "$REPO_ROOT"
"$RUNTIME_PYTHON" "$HANDOFF/math500_eval.py" --repo-root "$REPO_ROOT" merge --variant "$EVAL_VARIANT"
echo "MATH500_V5_FINALIZE_COMPLETE variant=$EVAL_VARIANT job=$SLURM_JOB_ID at=$(date -Is)"
