#!/usr/bin/env bash
#SBATCH --job-name=q8-m5-base-2k-v4-rescore
#SBATCH --account=raivn-ckpt
#SBATCH --partition=ckpt-all
#SBATCH --qos=ckpt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the SLIME repository}}"
HANDOFF="$REPO_ROOT/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff"
CONTROL="$HANDOFF/current_math500_eval.py"
AXOLOTL_ROOT="${AXOLOTL_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/Axolotl-Masked-SFT}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-$AXOLOTL_ROOT/.venv-vllm/bin/python}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO_ROOT"

pids=()
labels=()

"$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" \
  rescore-one --eval-id base_8b --bootstrap-seed 1234 &
pids+=("$!")
labels+=("base_8b")

two_k_ids=(
  2k_correct_only
  2k_unmasked
  2k_random_mask_25
  2k_random_mask_50
  2k_random_mask_70
  2k_random_mask_80
  2k_random_mask_90
  2k_inverse_tau_0p20
  2k_inverse_tau_0p05
  2k_margin_mask
  2k_prob_ratio_mask
)

for index in "${!two_k_ids[@]}"; do
  eval_id="${two_k_ids[$index]}"
  "$VLLM_PYTHON_BIN" "$CONTROL" --repo-root "$REPO_ROOT" \
    rescore-2k-one --eval-id "$eval_id" \
    --bootstrap-seed "$((1234 + index))" &
  pids+=("$!")
  labels+=("$eval_id")
done

status=0
for index in "${!pids[@]}"; do
  if ! wait "${pids[$index]}"; then
    echo "BASE_2K_V4_RESCORE_FAILED id=${labels[$index]}" >&2
    status=1
  fi
done
[[ "$status" -eq 0 ]]

echo "BASE_2K_V4_RESCORE_COMPLETE checkpoints=12 rows=6000 job=$SLURM_JOB_ID at=$(date -Is)"
