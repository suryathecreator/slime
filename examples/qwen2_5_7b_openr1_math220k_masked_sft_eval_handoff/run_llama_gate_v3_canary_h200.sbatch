#!/usr/bin/env bash
#SBATCH --job-name=q25-v3-canary
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --gpus=h200:1
#SBATCH --time=00:20:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff"
PYTHON_BIN="${LLAMA_GATE_PYTHON_BIN:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/checkpoints/cache/llama-gate/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/checkpoints/cache/llama-gate/xdg}"
export TMPDIR="${TMPDIR:-$REPO_ROOT/checkpoints/cache/llama-gate/v3-tmp}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
"$PYTHON_BIN" "$HANDOFF/llama_math500_eval_v3.py" --repo-root "$REPO_ROOT" run-canary
