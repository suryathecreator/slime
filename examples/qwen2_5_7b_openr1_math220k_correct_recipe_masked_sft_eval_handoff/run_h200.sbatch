#!/usr/bin/env bash
#SBATCH --job-name=q25m-m500
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=h200:1
#SBATCH --time=02:00:00
#SBATCH --exclude=g3130
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_eval_handoff"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$HANDOFF/runtime_python.sh}"
CONTROL="$HANDOFF/math500_eval.py"
GENERATOR="$HANDOFF/generate_math500.py"
: "${EVAL_TARGET:?Set EVAL_TARGET}"
task="${SLURM_ARRAY_TASK_ID:?Run as array 0-11}"
[[ "$task" =~ ^([0-9]|1[01])$ ]] || { echo "Invalid array task: $task" >&2; exit 2; }
repeat=$((task / 4)); shard=$((task % 4)); mode=sampled

MODEL="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field model --target "$EVAL_TARGET")"
TOKENIZER="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field tokenizer --target "$EVAL_TARGET")"
IDENTITY="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field checkpoint-identity --target "$EVAL_TARGET")"
EVAL_FILE="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field shard --shard "$shard")"
PASS_OUTPUT="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field pass-output --target "$EVAL_TARGET" --mode "$mode" --repeat "$repeat")"
OUTPUT="$PASS_OUTPUT/shards/shard_$(printf '%02d' "$shard")"
family=qwen2_5

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/q25m-math500}/array_${SLURM_ARRAY_JOB_ID:?}/$EVAL_TARGET/task_$(printf '%02d' "$task")"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="$(mktemp -d "/tmp/q25m5-${SLURM_ARRAY_JOB_ID:?}-${task}.XXXXXX")"
mkdir -p "$OUTPUT" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
cd "$REPO_ROOT"
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" check-canary

child_pid=""
requeue_job() {
  trap - USR1
  [[ -z "$child_pid" ]] || { kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; }
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_job USR1
"$RUNTIME_PYTHON" "$GENERATOR" \
  --model "$MODEL" --tokenizer "$TOKENIZER" --tokenizer-family "$family" \
  --checkpoint-manifest-sha256 "$IDENTITY" --target "$EVAL_TARGET" \
  --mode "$mode" --repeat "$repeat" --eval-file "$EVAL_FILE" --output-dir "$OUTPUT" &
child_pid="$!"; wait "$child_pid"; child_pid=""
echo "Q25M_MATH500_SHARD_COMPLETE target=$EVAL_TARGET mode=$mode repeat=$repeat shard=$shard job=$SLURM_JOB_ID at=$(date -Is)"
