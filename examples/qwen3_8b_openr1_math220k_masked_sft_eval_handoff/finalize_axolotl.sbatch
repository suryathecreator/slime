#!/usr/bin/env bash
#SBATCH --job-name=q8b-m500-axo-merge
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
PYTHON_BIN="${PYTHON_BIN:-$AXOLOTL_ROOT/.venv/bin/python}"
: "${EVAL_VARIANT:?Set EVAL_VARIANT to one approved available variant}"

cd "$REPO_ROOT"
"$PYTHON_BIN" "$HANDOFF/axolotl_math500_eval.py" \
  --repo-root "$REPO_ROOT" merge --variant "$EVAL_VARIANT"
echo "AXOLOTL_EVAL_FINALIZE_COMPLETE variant=$EVAL_VARIANT job=$SLURM_JOB_ID at=$(date -Is)"
