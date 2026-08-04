#!/usr/bin/env bash
#SBATCH --job-name=q25-m500-v3
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=220G
#SBATCH --gpus=h200:1
#SBATCH --time=00:20:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@180
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen2_5_7b_openr1_math220k_masked_sft_eval_handoff"
PYTHON_BIN="${LLAMA_GATE_PYTHON_BIN:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
: "${EVAL_VARIANT:?Set EVAL_VARIANT}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?Run as shard array 0-3}"
[[ "$SHARD_INDEX" =~ ^[0-3]$ ]]
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
GATE_ROOT="$("$PYTHON_BIN" "$HANDOFF/llama_math500_eval_v3.py" --repo-root "$REPO_ROOT" path --field gate)"
TASK_CACHE_KEY="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_$SHARD_INDEX"
export HF_HOME="${HF_HOME:-$REPO_ROOT/checkpoints/cache/llama-gate/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$REPO_ROOT/checkpoints/cache/llama-gate/xdg}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$REPO_ROOT/checkpoints/cache/llama-gate/vllm/$TASK_CACHE_KEY}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$REPO_ROOT/checkpoints/cache/llama-gate/torchinductor/$TASK_CACHE_KEY}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$REPO_ROOT/checkpoints/cache/llama-gate/triton/$TASK_CACHE_KEY}"
export TMPDIR="${TMPDIR:-$GATE_ROOT/tmp}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_NO_USAGE_STATS=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

child_pid=""
requeue_v3() {
  trap - USR1
  if [[ -n "$child_pid" ]]; then
    kill -USR1 "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_v3 USR1
set +e
"$PYTHON_BIN" "$HANDOFF/llama_math500_eval_v3.py" --repo-root "$REPO_ROOT" run-extraction \
  --variant "$EVAL_VARIANT" --shard-index "$SHARD_INDEX" &
child_pid="$!"
wait "$child_pid"
status="$?"
child_pid=""
set -e
if [[ "$status" -eq 75 ]]; then
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi
[[ "$status" -eq 0 ]] || exit "$status"
echo "LLAMA_V3_H200_COMPLETE variant=$EVAL_VARIANT shard=$SHARD_INDEX job=$SLURM_JOB_ID"
