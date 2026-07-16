#!/bin/bash
#SBATCH --job-name=q3-4b-opd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
: "${OPD_STAGE:?Set OPD_STAGE=1k or 5k}"
if [[ ! -s "${OPD_TUNING_FILE}" ]]; then
  echo "Missing OPD tuning selection: ${OPD_TUNING_FILE}" >&2
  exit 1
fi
source "${OPD_TUNING_FILE}"
cd "${SLIME_REPO_ROOT}"
sha256sum -c "${SOURCE_MANIFEST}"

if [[ "${OPD_STAGE}" == "1k" ]]; then
  export OPD_RUN_LABEL=qwen3_4b_harp_v2_opd_001024
  export OPD_JSONL="${CLEANED_OPD_1K_JSONL}" OPD_POOL_SIZE=1024 OPD_SIZE=1024 OPD_TRAIN_SIZE=1024
  export OPD_NUM_ROLLOUT=8 OPD_FINAL_ROLLOUT_ID=7 OPD_EFFECTIVE_TRAIN_SAMPLES=1024 OPD_MILESTONE_ROLLOUT_IDS=7 OPD_SAVE_INTERVAL=8
  export OPD_SAVE_DIR="${OPD1_SAVE_DIR}" OPD_HF_SNAPSHOT_DIR="${OPD1_HF_DIR}"
  export OPD_HF_SNAPSHOT_TEMPLATE="${OPD_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}" OPD_FINAL_HF_DIR="${OPD_HF_SNAPSHOT_DIR}/iter_0000007"
  export OPD_ROLLOUT_LOG_DIR="${OPD1_ROLLOUT_LOG_DIR}" OPD_SANITY_REPORT_DIR="${OPD1_SANITY_DIR}" OPD_SANITY_SUMMARY_DIR="${OPD1_SANITY_DIR}"
  export OPD_TRAINED_MANIFEST="${OPD_SAVE_DIR}/opd_trained_manifest.json"
  export OPD_INITIAL_LOAD_MODE=hf OPD_INITIAL_LOAD_DIR="${STUDENT_HF_DIR}" OPD_REF_LOAD_DIR="${STUDENT_HF_DIR}"
  export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=0 OPD_ALREADY_TRAINED_SAMPLES=0 OPD_START_ROLLOUT_ID=0
  export OPD_MANIFEST_FROM_ROLLOUT_LOGS=0 OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=0
  export OPD_RETIRE_PREVIOUS_FULL_SAVE_DIR="" OPD_RETIRE_PREVIOUS_HF_DIR=""
  export OPD_SANITY_MAX_ROLLOUT_ID=7
elif [[ "${OPD_STAGE}" == "5k" ]]; then
  export OPD_RUN_LABEL=qwen3_4b_harp_v2_opd_005120
  export OPD_JSONL="${CLEANED_OPD_4K_JSONL}" OPD_POOL_SIZE=4096 OPD_SIZE=4096 OPD_TRAIN_SIZE=5120
  export OPD_NUM_ROLLOUT=40 OPD_FINAL_ROLLOUT_ID=39 OPD_EFFECTIVE_TRAIN_SAMPLES=5120 OPD_MILESTONE_ROLLOUT_IDS=39 OPD_SAVE_INTERVAL=8
  export OPD_SAVE_DIR="${OPD_CONT_SAVE_DIR}" OPD_HF_SNAPSHOT_DIR="${OPD_CONT_HF_DIR}"
  export OPD_HF_SNAPSHOT_TEMPLATE="${OPD_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}" OPD_FINAL_HF_DIR="${OPD_HF_SNAPSHOT_DIR}/iter_0000039"
  export OPD_ROLLOUT_LOG_DIR="${OPD_CONT_ROLLOUT_LOG_DIR}" OPD_SANITY_REPORT_DIR="${OPD_CONT_SANITY_DIR}" OPD_SANITY_SUMMARY_DIR="${OPD_CONT_SANITY_DIR}"
  export OPD_TRAINED_MANIFEST="${OPD_SAVE_DIR}/opd_trained_manifest.json"
  export OPD_INITIAL_LOAD_MODE=megatron OPD_INITIAL_LOAD_DIR="${OPD1_SAVE_DIR}" OPD_REF_LOAD_DIR="${STUDENT_HF_DIR}"
  export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=1 OPD_ALREADY_TRAINED_SAMPLES=1024 OPD_START_ROLLOUT_ID=8
  export OPD_MANIFEST_FROM_ROLLOUT_LOGS=1 OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=1
  export OPD_PREVIOUS_RUN_LABEL=qwen3_4b_harp_v2_opd_001024
  export OPD_PREVIOUS_SAVE_DIR="${OPD1_SAVE_DIR}" OPD_PREVIOUS_HF_SNAPSHOT_DIR="${OPD1_HF_DIR}"
  export OPD_PREVIOUS_ROLLOUT_LOG_DIR="${OPD1_ROLLOUT_LOG_DIR}" OPD_PREVIOUS_TRAINED_MANIFEST="${OPD1_SAVE_DIR}/opd_trained_manifest.json"
  export OPD_CONTINUATION_METADATA="${HARP_PREP_METADATA}"
  export OPD_RETIRE_PREVIOUS_FULL_SAVE_DIR="" OPD_RETIRE_PREVIOUS_HF_DIR=""
  export OPD_SANITY_MAX_ROLLOUT_ID=39
else
  echo "Unknown OPD_STAGE=${OPD_STAGE}" >&2
  exit 2
fi

python3 - "${OPD_JSONL}" <<'PY'
import json, sys
prefix = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\n"
with open(sys.argv[1], encoding="utf-8") as handle:
    first = json.loads(next(handle))
if not str(first.get("prompt", "")).startswith(prefix):
    raise SystemExit("OPD data does not use the canonical HARP prompt")
if (first.get("metadata") or {}).get("prompt_contract_version") != "qwen3_harp_boxed_v1":
    raise SystemExit("OPD prompt contract version is missing or incorrect")
PY

bash "${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch"
