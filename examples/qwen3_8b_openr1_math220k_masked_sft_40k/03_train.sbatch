#!/bin/bash
#SBATCH --job-name=q8b-masksft-train
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_40k"
HANDOFF_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_eval_handoff"
source "${SCRIPT_DIR}/env.sh"
: "${SFT_VARIANT:?Set SFT_VARIANT}"
: "${SLURM_JOB_ID:?Training must run under Slurm}"

# Python multiprocessing.Manager uses an AF_UNIX socket while Megatron creates
# async torch_dist checkpoints. Keep that socket off the long contract path.
export TMPDIR="/tmp/${USER:-suryadv}/q8b-masksft-train-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}"
"${SCRIPT_DIR}/container_exec.sh" python3 \
  "${SCRIPT_DIR}/check_train_tmpdir.py"

case "${SFT_VARIANT}" in
  correct_only) SFT_PARQUET="${CORRECT_ONLY_JSONL}" ;;
  weighted_tau_0p20) SFT_PARQUET="${WEIGHTED_JSONL}" ;;
  unmasked) SFT_PARQUET="${UNMASKED_JSONL}" ;;
  *) echo "Unknown SFT_VARIANT=${SFT_VARIANT}" >&2; exit 2 ;;
esac
export SFT_PARQUET
export SFT_SAVE_DIR="${OUTPUT_ROOT}/training/${SFT_VARIANT}/full_state"
export SFT_HF_SNAPSHOT_DIR="${OUTPUT_ROOT}/training/${SFT_VARIANT}/weights"
export SFT_HF_SNAPSHOT_TEMPLATE="${SFT_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
export SFT_FINAL_HF_DIR="${SFT_HF_SNAPSHOT_DIR}/iter_0000199"
export SFT_FINAL_FULL_CKPT_DIR="${SFT_SAVE_DIR}/iter_0000199"
export SFT_DETAILS_DIR="${OUTPUT_ROOT}/training/${SFT_VARIANT}/details"
export SFT_WANDB_GROUP="qwen3-8b-openr1-40k-${SFT_VARIANT}"

mkdir -p "${SLURM_LOG_DIR}" "${OUTPUT_ROOT}/training/${SFT_VARIANT}"
"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/validate_dataset.py" \
  --data "${SFT_PARQUET}" --variant "${SFT_VARIANT}"

echo "TRAIN_VARIANT=${SFT_VARIANT} initialization=${STUDENT_TORCH_DIST_DIR} full_sft=1"
bash "${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/04_run_sft_100k_8xh200.sbatch"
python3 "${HANDOFF_DIR}/checkpoint_manifest.py" record \
  --checkpoint "${SFT_FINAL_HF_DIR}" \
  --output "${EXPERIMENT_ROOT}/handoff/checkpoints/${SFT_VARIANT}.json" \
  --family 40k \
  --variant "${SFT_VARIANT}" \
  --final-iteration "${SFT_FINAL_ROLLOUT_ID}" \
  --contract-hash "${CONTRACT_HASH}" \
  --repo-root "${SLIME_REPO_ROOT}" \
  --full-sft
echo "TRAIN_JOB_COMPLETE variant=${SFT_VARIANT} at=$(date -Is)"
