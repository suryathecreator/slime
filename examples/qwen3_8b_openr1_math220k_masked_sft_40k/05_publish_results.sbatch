#!/bin/bash
#SBATCH --job-name=q8b-masksft-publish
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_openr1_math220k_masked_sft_40k"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

[[ "$(git branch --show-current)" == "qwen3-8b-openr1-masked-sft-40k" ]] || { echo "Wrong publish branch" >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing to publish from a dirty worktree" >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$(git rev-parse origin/qwen3-8b-openr1-masked-sft-40k)" ]] || {
  echo "Publish worktree and remote branch differ; refusing conflict resolution" >&2
  exit 1
}

declare -A SOURCES=(
  [RESULTS.json]="${EXPERIMENT_ROOT}/RESULTS.json"
  [RESULTS.md]="${EXPERIMENT_ROOT}/RESULTS.md"
  [DATA_SELECTION_AUDIT.json]="${PREP_STATS_JSON}"
  [WEIGHTED_STATS.json]="${WEIGHTED_STATS_JSON}"
  [SLURM_SUBMISSION.json]="${MANIFEST_ROOT}/submission.json"
  [SLURM_RESUBMISSION.json]="${MANIFEST_ROOT}/resubmission_from_margin.json"
  [MATH500_CORRECT_ONLY_SUMMARY.json]="${EVAL_OUTPUT_ROOT}/correct_only/summary.json"
  [MATH500_WEIGHTED_TAU_0P20_SUMMARY.json]="${EVAL_OUTPUT_ROOT}/weighted_tau_0p20/summary.json"
  [MATH500_UNMASKED_SUMMARY.json]="${EVAL_OUTPUT_ROOT}/unmasked/summary.json"
)
for destination in "${!SOURCES[@]}"; do
  source_path="${SOURCES[${destination}]}"
  [[ -s "${source_path}" ]] || { echo "Missing publish artifact ${source_path}" >&2; exit 1; }
  cp "${source_path}" "${SCRIPT_DIR}/${destination}"
done

git add -- \
  "${SCRIPT_DIR}/RESULTS.json" \
  "${SCRIPT_DIR}/RESULTS.md" \
  "${SCRIPT_DIR}/DATA_SELECTION_AUDIT.json" \
  "${SCRIPT_DIR}/WEIGHTED_STATS.json" \
  "${SCRIPT_DIR}/SLURM_SUBMISSION.json" \
  "${SCRIPT_DIR}/SLURM_RESUBMISSION.json" \
  "${SCRIPT_DIR}/MATH500_CORRECT_ONLY_SUMMARY.json" \
  "${SCRIPT_DIR}/MATH500_WEIGHTED_TAU_0P20_SUMMARY.json" \
  "${SCRIPT_DIR}/MATH500_UNMASKED_SUMMARY.json"
git commit -m "Publish Qwen3 8B OpenR1 masked SFT results"
git push origin HEAD:qwen3-8b-openr1-masked-sft-40k
echo "RESULTS_PUBLISHED commit=$(git rev-parse HEAD) at=$(date -Is)"
