#!/usr/bin/env bash
#SBATCH --job-name=q8b-m500-current-finalize
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff"
CONTROL="$HANDOFF/current_math500_eval.py"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$AXOLOTL_ROOT/.venv-vllm/bin/python}"
: "${EVAL_ID:?Set EVAL_ID to one approved base/40K checkpoint ID}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"
"$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" \
  merge-current --eval-id "$EVAL_ID"
"$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" \
  rescore-one --eval-id "$EVAL_ID" --bootstrap-seed 1234
echo "CURRENT_EVAL_FINALIZE_COMPLETE id=$EVAL_ID job=$SLURM_JOB_ID at=$(date -Is)"
