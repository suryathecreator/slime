#!/usr/bin/env bash
#SBATCH --job-name=q3-aime16-publish

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_8b_aime_2009_2024_16x"
PYTHON_BIN="$REPO_ROOT/checkpoints/runtime/qwen-math500-v1/venv/bin/python"
OUTPUT_ROOT="$("$PYTHON_BIN" "$HANDOFF/pipeline.py" path --field output_root)"
PUBLISH_DEPS="$OUTPUT_ROOT/publish_deps"

if ! PYTHONPATH="$PUBLISH_DEPS:${PYTHONPATH:-}" "$PYTHON_BIN" -c \
  'import pyarrow; assert pyarrow.__version__ == "19.0.1"'; then
  "$PYTHON_BIN" -m pip install --no-cache-dir --target "$PUBLISH_DEPS" \
    'pyarrow==19.0.1'
fi

export PYTHONPATH="$PUBLISH_DEPS:$HANDOFF:${PYTHONPATH:-}"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1

cd "$REPO_ROOT"
"$PYTHON_BIN" "$HANDOFF/publish_dataset.py" publish
