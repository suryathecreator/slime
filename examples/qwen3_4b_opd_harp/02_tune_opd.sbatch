#!/bin/bash
#SBATCH --job-name=q3-4b-opd-tune
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=24:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${SLURM_LOG_DIR}" "${OPD_TUNING_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi
cd "${SLIME_REPO_ROOT}"
sha256sum -c "${SOURCE_MANIFEST}"
if [[ -s "${OPD_TUNING_FILE}" ]]; then
  echo "Reusing OPD tuning selection ${OPD_TUNING_FILE}"
  exit 0
fi

runner="${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch"
results="${OPD_TUNING_ROOT}/results.tsv"
printf 'candidate\tstatus\telapsed_seconds\tcontext_parallel\tmax_tokens_per_gpu\tlog_probs_chunk_size\n' >"${results}"

run_candidate() {
  local name="$1"
  local context_parallel="$2"
  local max_tokens="$3"
  local log_chunk="$4"
  local root="${OPD_TUNING_ROOT}/${name}"
  export OPD_RUN_LABEL="harp_v2_tune_${name}"
  export OPD_JSONL="${CLEANED_OPD_1K_JSONL}"
  export OPD_POOL_SIZE=1024 OPD_SIZE=1024 OPD_TRAIN_SIZE=256
  export OPD_NUM_ROLLOUT=2 OPD_FINAL_ROLLOUT_ID=1 OPD_EFFECTIVE_TRAIN_SAMPLES=256 OPD_MILESTONE_ROLLOUT_IDS=1
  export OPD_SAVE_INTERVAL=2
  export OPD_CONTEXT_PARALLEL_SIZE="${context_parallel}"
  export OPD_MAX_TOKENS_PER_GPU="${max_tokens}"
  export OPD_LOG_PROBS_CHUNK_SIZE="${log_chunk}"
  export OPD_SAVE_DIR="${root}/full_optim"
  export OPD_HF_SNAPSHOT_DIR="${root}/hf"
  export OPD_HF_SNAPSHOT_TEMPLATE="${OPD_HF_SNAPSHOT_DIR}/iter_{rollout_id:07d}"
  export OPD_FINAL_HF_DIR="${OPD_HF_SNAPSHOT_DIR}/iter_0000001"
  export OPD_ROLLOUT_LOG_DIR="${root}/rollouts"
  export OPD_SANITY_REPORT_DIR="${root}/sanity"
  export OPD_SANITY_SUMMARY_DIR="${root}/sanity"
  export OPD_SANITY_MAX_ROLLOUT_ID=1
  export OPD_TRAINED_MANIFEST="${OPD_SAVE_DIR}/opd_trained_manifest.json"
  export OPD_INITIAL_LOAD_MODE=hf OPD_INITIAL_LOAD_DIR="${STUDENT_HF_DIR}" OPD_REF_LOAD_DIR="${STUDENT_HF_DIR}"
  export OPD_SKIP_ROLLOUT_DATA_STATE_LOAD=0 OPD_ALREADY_TRAINED_SAMPLES=0 OPD_START_ROLLOUT_ID=0
  export OPD_MANIFEST_FROM_ROLLOUT_LOGS=0 OPD_ALLOW_OPT_PARAM_SCHEDULER_MISMATCH=0
  export OPD_RETIRE_PREVIOUS_FULL_SAVE_DIR="" OPD_RETIRE_PREVIOUS_HF_DIR=""
  local start end elapsed status=ok
  start=$(date +%s)
  if ! SLURM_JOB_NAME="q3-4b-opd-tune-${name}" bash "${runner}"; then
    status=failed
  fi
  end=$(date +%s)
  elapsed=$((end - start))
  log="${SLURM_LOG_DIR}/q3-4b-opd-tune-${name}_${SLURM_JOB_ID:-manual}.log"
  if [[ "${status}" == ok ]] && { [[ ! -f "${OPD_SAVE_DIR}/latest_checkpointed_iteration.txt" ]] || [[ ! -f "${OPD_FINAL_HF_DIR}/config.json" ]]; }; then
    status=invalid_checkpoint
  fi
  if [[ -f "${log}" ]] && grep -Eqi 'CUDA out of memory|out of memory|(^|[^A-Za-z])nan([^A-Za-z]|$)|non-finite' "${log}"; then
    status=invalid_numeric_or_oom
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${name}" "${status}" "${elapsed}" "${context_parallel}" "${max_tokens}" "${log_chunk}" >>"${results}"
}

run_candidate baseline_cp3_t2048_c512 3 2048 512
run_candidate cp1_t4096_c1024 1 4096 1024
run_candidate cp1_t8192_c2048 1 8192 2048

python3 - "${results}" "${OPD_TUNING_FILE}" <<'PY'
import csv
import json
import sys
from pathlib import Path

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8"), delimiter="\t"))
valid = [row for row in rows if row["status"] == "ok"]
if not valid:
    raise SystemExit("No valid OPD throughput candidate completed")
selected = min(valid, key=lambda row: int(row["elapsed_seconds"]))
path = Path(sys.argv[2])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(
    "export OPD_CONTEXT_PARALLEL_SIZE={context_parallel}\n"
    "export OPD_MAX_TOKENS_PER_GPU={max_tokens_per_gpu}\n"
    "export OPD_LOG_PROBS_CHUNK_SIZE={log_probs_chunk_size}\n".format(**selected),
    encoding="utf-8",
)
(path.parent / "selection.json").write_text(
    json.dumps({"selection_metric": "two_update_end_to_end_elapsed_seconds", "selected": selected, "candidates": rows}, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print("Selected OPD candidate", selected)
PY
