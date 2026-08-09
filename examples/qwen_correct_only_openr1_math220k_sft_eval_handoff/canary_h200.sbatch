#!/usr/bin/env bash
#SBATCH --job-name=qco2-m500-canary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=h200:1
#SBATCH --time=00:30:00
#SBATCH --exclude=g3130
#SBATCH --requeue
#SBATCH --signal=B:USR1@600

set -euo pipefail
REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen_correct_only_openr1_math220k_sft_eval_handoff"
RUNTIME_PYTHON="${RUNTIME_PYTHON:-$HANDOFF/runtime_python.sh}"
CONTROL="$HANDOFF/math500_eval.py"
GENERATOR="$HANDOFF/generate_math500.py"
SMOKE="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field smoke)"

export HF_HOME="${HF_HOME:-/gscratch/scrubbed/suryadv/hf_cache}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false VLLM_WORKER_MULTIPROC_METHOD=spawn
CACHE_ROOT="${SCRUBBED_CACHE_ROOT:-/gscratch/scrubbed/suryadv/cache/qco2-math500}/job_${SLURM_JOB_ID:?}/canary"
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
for canary_case in \
  qwen2_5:base_qwen2_5_3b:greedy:0 \
  qwen3:base_qwen3_4b:sampled:0; do
  IFS=: read -r family target mode repeat <<< "$canary_case"
  model="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field model --target "$target")"
  tokenizer="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field tokenizer --target "$target")"
  identity="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field checkpoint-identity --target "$target")"
  output="$("$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" path --field canary-output --target "$target" --mode "$mode" --repeat "$repeat")"
  "$RUNTIME_PYTHON" "$GENERATOR" \
    --model "$model" --tokenizer "$tokenizer" --tokenizer-family "$family" \
    --checkpoint-manifest-sha256 "$identity" --target "$target" \
    --mode "$mode" --repeat "$repeat" --eval-file "$SMOKE" --output-dir "$output" \
    --chunk-size 1 --max-num-seqs 1 &
  child_pid="$!"; wait "$child_pid"; child_pid=""
done
"$RUNTIME_PYTHON" "$CONTROL" --repo-root "$REPO_ROOT" mark-canary --job-id "$SLURM_JOB_ID"
echo "QCO2_MATH500_CANARY_COMPLETE job=$SLURM_JOB_ID cases=2 at=$(date -Is)"
