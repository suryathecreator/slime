#!/usr/bin/env bash
#SBATCH --job-name=q25-m500-strict-audit
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff"
PYTHON_BIN="${LLAMA_GATE_PYTHON_BIN:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
"$PYTHON_BIN" "$HANDOFF/llama_math500_eval.py" --repo-root "$REPO_ROOT" audit
