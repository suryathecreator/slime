#!/usr/bin/env bash
#SBATCH --job-name=q3-aime16-generate

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_8b_aime_2009_2024_16x"
PYTHON_BIN="$REPO_ROOT/checkpoints/runtime/qwen-math500-v1/venv/bin/python"
TASK_ID="${TASK_ID_OVERRIDE:-${SLURM_ARRAY_TASK_ID:?Set an array task or TASK_ID_OVERRIDE}}"
LIMIT_ARGS=()
if [[ -n "${CANARY_LIMIT:-}" ]]; then
  LIMIT_ARGS=(--limit "$CANARY_LIMIT")
fi

OUTPUT_ROOT="$("$PYTHON_BIN" "$HANDOFF/pipeline.py" path --field output_root)"
CACHE_ROOT="${QWEN3_AIME_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/qwen3-aime-2009-2024-16x}"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg"
export VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="/tmp/${USER:-suryadv}/q3-aime16-${SLURM_JOB_ID}-${TASK_ID}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$HANDOFF:${PYTHONPATH:-}"

mkdir -p "$OUTPUT_ROOT" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$TMPDIR"
cd "$REPO_ROOT"

child_pid=""
requeue_generation() {
  trap - USR1
  echo "QWEN3_AIME_REQUEUE task=$TASK_ID job=$SLURM_JOB_ID at=$(date -Is)"
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_generation USR1

"$PYTHON_BIN" "$HANDOFF/run_generation.py" \
  --task-id "$TASK_ID" "${LIMIT_ARGS[@]}" &
child_pid="$!"
wait "$child_pid"
child_pid=""

echo "QWEN3_AIME_SHARD_COMPLETE task=$TASK_ID job=$SLURM_JOB_ID at=$(date -Is)"
