#!/bin/bash
#SBATCH --job-name=q3-1.7b-opd-aime26-rem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=512G
#SBATCH --time=24:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_1_7b_opd_aime2026/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_1_7b_opd_aime2026"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
: "${OPD_STAGE:?Set OPD_STAGE=2k or 5k}"
: "${REMAINING_SOURCE_MANIFEST:?Set REMAINING_SOURCE_MANIFEST}"
export CC="${AIME_ZIG_CC}"
export CPATH="${AIME_ZIG_INCLUDE_ROOT}/python3.12:${AIME_ZIG_INCLUDE_ROOT}${CPATH:+:${CPATH}}"
cd "${SLIME_REPO_ROOT}"
sha256sum -c "${SOURCE_MANIFEST}"
sha256sum -c "${REMAINING_SOURCE_MANIFEST}"
test -x "${CC}"
test -f "${AIME_ZIG_INCLUDE_ROOT}/python3.12/Python.h"
test -f "${AIME_ZIG_INCLUDE_ROOT}/x86_64-linux-gnu/python3.12/pyconfig.h"
echo "OPD Triton C compiler: ${CC}"

REMAINING_DATA_ROOT="${DATA_ROOT}/remaining_2k_chain"
REMAINING_OPD_2K_JSONL="${REMAINING_DATA_ROOT}/opd_002048_qwen3_aime_prompt.jsonl"
REMAINING_OPD_3K_JSONL="${REMAINING_DATA_ROOT}/opd_next_003072_qwen3_aime_prompt.jsonl"
REMAINING_OPD_METADATA="${REMAINING_DATA_ROOT}/repartition_metadata.json"
OPD2_SAVE_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_002048_full_optim"
OPD2_HF_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_002048_hf"
OPD2_ROLLOUT_LOG_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_002048_rollouts"
OPD2_SANITY_DIR="${OUTPUT_ROOT}/qwen3_1.7b_opd_002048_sanity"
export SPLIT_METADATA="${REMAINING_OPD_METADATA}"

if [[ "${OPD_STAGE}" == "2k" ]]; then
  export OPD_RUN_LABEL=qwen3_1.7b_aime2026_opd_002048
  export OPD_JSONL="${REMAINING_OPD_2K_JSONL}" OPD_POOL_SIZE=2048 OPD_SIZE=2048 OPD_TRAIN_SIZE=2048
  export OPD_NUM_ROLLOUT=16 OPD_FINAL_ROLLOUT_ID=15 OPD_EFFECTIVE_TRAIN_SAMPLES=2048 OPD_MILESTONE_ROLLOUT_IDS=15 OPD_SAVE_INTERVAL=8
  export OPD_SAVE_DIR="${OPD2_SAVE_DIR}" OPD_HF_SNAPSHOT_DIR="${OPD2_HF_DIR}"
  export OPD_HF_SNAPSHOT_TEMPLATE="${OPD_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}" OPD_FINAL_HF_DIR="${OPD_HF_SNAPSHOT_DIR}/iter_0000015"
  export OPD_ROLLOUT_LOG_DIR="${OPD2_ROLLOUT_LOG_DIR}" OPD_SANITY_REPORT_DIR="${OPD2_SANITY_DIR}" OPD_SANITY_SUMMARY_DIR="${OPD2_SANITY_DIR}"
  export OPD_TRAINED_MANIFEST="${OPD_SAVE_DIR}/opd_trained_manifest.json"
  export OPD_INITIAL_LOAD_MODE=hf OPD_INITIAL_LOAD_DIR="${STUDENT_HF_DIR}" OPD_REF_LOAD_DIR="${STUDENT_HF_DIR}"
  export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=0 OPD_ALREADY_TRAINED_SAMPLES=0 OPD_START_ROLLOUT_ID=0
  export OPD_MANIFEST_FROM_ROLLOUT_LOGS=0 OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=0
  export OPD_RETIRE_PREVIOUS_FULL_SAVE_DIR="" OPD_RETIRE_PREVIOUS_HF_DIR=""
  export OPD_SANITY_MAX_ROLLOUT_ID=15
  expected_rows=2048
  expected_first_offset=0
elif [[ "${OPD_STAGE}" == "5k" ]]; then
  export OPD_RUN_LABEL=qwen3_1.7b_aime2026_opd_005120
  export OPD_JSONL="${REMAINING_OPD_3K_JSONL}" OPD_POOL_SIZE=3072 OPD_SIZE=3072 OPD_TRAIN_SIZE=5120
  export OPD_NUM_ROLLOUT=40 OPD_FINAL_ROLLOUT_ID=39 OPD_EFFECTIVE_TRAIN_SAMPLES=5120 OPD_MILESTONE_ROLLOUT_IDS=39 OPD_SAVE_INTERVAL=8
  export OPD_SAVE_DIR="${OPD_CONT_SAVE_DIR}" OPD_HF_SNAPSHOT_DIR="${OPD_CONT_HF_DIR}"
  export OPD_HF_SNAPSHOT_TEMPLATE="${OPD_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}" OPD_FINAL_HF_DIR="${OPD_HF_SNAPSHOT_DIR}/iter_0000039"
  export OPD_ROLLOUT_LOG_DIR="${OPD_CONT_ROLLOUT_LOG_DIR}" OPD_SANITY_REPORT_DIR="${OPD_CONT_SANITY_DIR}" OPD_SANITY_SUMMARY_DIR="${OPD_CONT_SANITY_DIR}"
  export OPD_TRAINED_MANIFEST="${OPD_SAVE_DIR}/opd_trained_manifest.json"
  export OPD_INITIAL_LOAD_MODE=megatron OPD_INITIAL_LOAD_DIR="${OPD2_SAVE_DIR}" OPD_REF_LOAD_DIR="${STUDENT_HF_DIR}"
  export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=1 OPD_ALREADY_TRAINED_SAMPLES=2048 OPD_START_ROLLOUT_ID=16
  export OPD_MANIFEST_FROM_ROLLOUT_LOGS=1 OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=1
  export OPD_PREVIOUS_RUN_LABEL=qwen3_1.7b_aime2026_opd_002048
  export OPD_PREVIOUS_SAVE_DIR="${OPD2_SAVE_DIR}" OPD_PREVIOUS_HF_SNAPSHOT_DIR="${OPD2_HF_DIR}"
  export OPD_PREVIOUS_ROLLOUT_LOG_DIR="${OPD2_ROLLOUT_LOG_DIR}" OPD_PREVIOUS_TRAINED_MANIFEST="${OPD2_SAVE_DIR}/opd_trained_manifest.json"
  export OPD_CONTINUATION_METADATA="${REMAINING_OPD_METADATA}"
  export OPD_RETIRE_PREVIOUS_FULL_SAVE_DIR="" OPD_RETIRE_PREVIOUS_HF_DIR=""
  export OPD_SANITY_MAX_ROLLOUT_ID=39
  expected_rows=3072
  expected_first_offset=2048
else
  echo "Unknown OPD_STAGE=${OPD_STAGE}" >&2
  exit 2
fi

model_parallel_size=$((OPD_TENSOR_MODEL_PARALLEL_SIZE * OPD_CONTEXT_PARALLEL_SIZE))
if (( OPD_ACTOR_GPUS % model_parallel_size != 0 )); then
  echo "Actor GPU count ${OPD_ACTOR_GPUS} is not divisible by model-parallel size ${model_parallel_size}" >&2
  exit 2
fi
data_parallel_size=$((OPD_ACTOR_GPUS / model_parallel_size))
if (( OPD_GLOBAL_BATCH_SIZE % data_parallel_size != 0 )); then
  echo "Global batch ${OPD_GLOBAL_BATCH_SIZE} is not divisible by data-parallel size ${data_parallel_size}" >&2
  exit 2
fi
samples_per_rollout=$((OPD_ROLLOUT_BATCH_SIZE * OPD_N_SAMPLES_PER_PROMPT))
if (( samples_per_rollout % OPD_GLOBAL_BATCH_SIZE != 0 )); then
  echo "Rollout samples ${samples_per_rollout} are not divisible by global batch ${OPD_GLOBAL_BATCH_SIZE}" >&2
  exit 2
fi
if (( OPD_EFFECTIVE_TRAIN_SAMPLES != OPD_NUM_ROLLOUT * OPD_ROLLOUT_BATCH_SIZE )); then
  echo "Effective samples ${OPD_EFFECTIVE_TRAIN_SAMPLES} do not equal rollout count times rollout batch" >&2
  exit 2
fi
echo "Validated OPD geometry: actor=${OPD_ACTOR_GPUS} rollout=${OPD_ROLLOUT_GPUS} dp=${data_parallel_size} global_batch=${OPD_GLOBAL_BATCH_SIZE}"

python3 - "${OPD_JSONL}" "${expected_rows}" "${expected_first_offset}" <<'PY'
import json, sys
from pathlib import Path

path = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
expected_first = int(sys.argv[3])
prefix = "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\n"
rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
if len(rows) != expected_rows:
    raise SystemExit(f"Expected {expected_rows} OPD rows, found {len(rows)}")
offsets = [int((row.get("metadata") or {}).get("selected_offset", -1)) for row in rows]
if offsets != list(range(expected_first, expected_first + expected_rows)):
    raise SystemExit("OPD selected_offset coverage/order mismatch")
if any(not str(row.get("prompt", "")).startswith(prefix) for row in rows):
    raise SystemExit("OPD data does not use the canonical AIME prompt")
if any((row.get("metadata") or {}).get("prompt_contract_version") != "qwen3_aime2026_boxed_v1" for row in rows):
    raise SystemExit("OPD prompt contract version is missing or incorrect")
print(f"Validated {len(rows)} OPD rows from selected_offset {offsets[0]} through {offsets[-1]}")
PY

bash "${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch"
