#!/usr/bin/env bash
#SBATCH --job-name=qco-m500-canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=h200:1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen_correct_only_openr1_math220k_sft_eval_handoff"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$HANDOFF/runtime_python.sh}"
CONTROL="$HANDOFF/math500_eval.py"
GENERATOR="$HANDOFF/generate_math500.py"
TARGET="base_qwen2_5_3b"
CONTROL_ROOT="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field control)"
MODEL="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field model --target "$TARGET")"
TOKENIZER="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field tokenizer --target "$TARGET")"
IDENTITY="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field checkpoint-identity --target "$TARGET")"
SMOKE="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field smoke)"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/qco-math500}/job_${SLURM_JOB_ID:?}/canary"
export XDG_CACHE_HOME="$CACHE_ROOT/xdg" VLLM_CACHE_ROOT="$CACHE_ROOT/vllm"
export TORCHINDUCTOR_CACHE_DIR="$CACHE_ROOT/torchinductor" TRITON_CACHE_DIR="$CACHE_ROOT/triton"
export TMPDIR="$(mktemp -d "/tmp/qcom5-${SLURM_JOB_ID:?}.XXXXXX")"
mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"
cd "$REPO_ROOT"
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" check-preflight

child_pid=""
requeue_job() {
  trap - USR1
  [[ -z "$child_pid" ]] || { kill -TERM "$child_pid" 2>/dev/null || true; wait "$child_pid" 2>/dev/null || true; }
  scontrol requeue "$SLURM_JOB_ID"
  exit 0
}
trap requeue_job USR1
for mode_repeat in greedy:0 sampled:0; do
  mode="${mode_repeat%%:*}"; repeat="${mode_repeat##*:}"
  output="$CONTROL_ROOT/canary/$mode"
  [[ "$mode" == sampled ]] && output="$CONTROL_ROOT/canary/sampled/repeat_00"
  "$RUNTIME_PYTHON" "$GENERATOR" \
    --model "$MODEL" --tokenizer "$TOKENIZER" --tokenizer-family qwen2_5 \
    --checkpoint-manifest-sha256 "$IDENTITY" --target "$TARGET" \
    --mode "$mode" --repeat "$repeat" --eval-file "$SMOKE" --output-dir "$output" \
    --chunk-size 1 --max-num-seqs 1 &
  child_pid="$!"; wait "$child_pid"; child_pid=""
done
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" mark-canary --job-id "$SLURM_JOB_ID"
echo "QCO_MATH500_CANARY_COMPLETE job=$SLURM_JOB_ID at=$(date -Is)"
