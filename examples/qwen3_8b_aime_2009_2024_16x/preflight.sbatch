#!/usr/bin/env bash
#SBATCH --job-name=q3-aime16-preflight

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_8b_aime_2009_2024_16x"
PYTHON_BIN="$REPO_ROOT/checkpoints/runtime/qwen-math500-v1/venv/bin/python"

export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="$HANDOFF:${PYTHONPATH:-}"

cd "$REPO_ROOT"
"$PYTHON_BIN" "$HANDOFF/pipeline.py" preflight
