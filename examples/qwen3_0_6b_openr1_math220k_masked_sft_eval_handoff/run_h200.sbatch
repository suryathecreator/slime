#!/usr/bin/env bash
#SBATCH --job-name=q06-m500-v5
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=h200:1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_0_6b_openr1_math220k_masked_sft_eval_handoff"
CONTROL="$HANDOFF/math500_eval.py"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$REPO_ROOT/checkpoints/runtime/qwen25-math500-llama-gate-v1/venv/bin/python}"
: "${EVAL_VARIANT:?Set EVAL_VARIANT to an approved checkpoint variant}"
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?Run as the 0-3 shard array}"
[[ "$SHARD_INDEX" =~ ^[0-3]$ ]] || { echo "Invalid shard: $SHARD_INDEX" >&2; exit 2; }

CONTROL_ROOT="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field control_root)"
MODEL_DIR="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field model --variant "$EVAL_VARIANT")"
TOKENIZER_DIR="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field tokenizer)"
VARIANT_OUTPUT="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field output --variant "$EVAL_VARIANT")"
EVAL_FILE="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field shard --shard-index "$SHARD_INDEX")"
SHARD_OUTPUT="$VARIANT_OUTPUT/shards/shard_$(printf '%02d' "$SHARD_INDEX")"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export SCRUBBED_CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/q06-math500-v5}"
TASK_CACHE_ROOT="$SCRUBBED_CACHE_ROOT/tasks/array_${SLURM_ARRAY_JOB_ID:?}/$EVAL_VARIANT/shard_$(printf '%02d' "$SHARD_INDEX")"
export XDG_CACHE_HOME="$TASK_CACHE_ROOT/xdg"
export VLLM_CACHE_ROOT="$TASK_CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE_ROOT/torchinductor"
export TRITON_CACHE_DIR="$TASK_CACHE_ROOT/triton"
TMPDIR="$(mktemp -d "/tmp/q06m5-${SLURM_ARRAY_JOB_ID:?}-${SHARD_INDEX}.XXXXXX")"
export TMPDIR
export HF_HUB_DISABLE_XET=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTHONPATH="$REPO_ROOT:$AXOLOTL_ROOT/scripts/openr1_axolotl:${PYTHONPATH:-}"

mkdir -p "$SHARD_OUTPUT" "$TMPDIR" "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
echo "MATH500_V5_CACHE_NAMESPACE variant=$EVAL_VARIANT shard=$SHARD_INDEX root=$TASK_CACHE_ROOT tmp=$TMPDIR"
cd "$REPO_ROOT"

"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" check-canary
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" repair-shard \
  --eval-file "$EVAL_FILE" --output-dir "$SHARD_OUTPUT"

child_pid=""
requeue_eval() {
  trap - USR1
  if [[ -n "$child_pid" ]]; then
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_eval USR1

"$RUNTIME_PYTHON" "$AXOLOTL_ROOT/scripts/openr1_axolotl/evaluate_math_vllm.py" \
  --eval_file "$EVAL_FILE" \
  --output_dir "$SHARD_OUTPUT" \
  --variant "$EVAL_VARIANT" \
  --model_name "$TOKENIZER_DIR" \
  --adapter_or_model_dir "$MODEL_DIR" \
  --max_tokens 32768 \
  --max_model_len 32768 \
  --chunk_size 32 \
  --max_num_seqs 8 \
  --gpu_memory_utilization 0.92 \
  --require_math_verify &
child_pid="$!"
wait "$child_pid"
child_pid=""

"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" repair-shard \
  --eval-file "$EVAL_FILE" --output-dir "$SHARD_OUTPUT"
echo "MATH500_V5_SHARD_COMPLETE variant=$EVAL_VARIANT shard=$SHARD_INDEX job=$SLURM_JOB_ID at=$(date -Is)"
