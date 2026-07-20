#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

[[ "$(git branch --show-current)" == "qwen3-8b-32b-full-repro" ]] || { echo "Wrong branch." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing submission from a dirty worktree." >&2; exit 1; }
python3 "${SCRIPT_DIR}/validate_config.py" --require-pinned-commit

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
[[ -n "${upstream}" ]] || { echo "Push this branch and set its upstream before submission." >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$(git rev-parse "${upstream}")" ]] || { echo "Local HEAD is not the pushed upstream commit." >&2; exit 1; }

for script in 00_preflight.sbatch 01_cleanup.sbatch 02_eval_math500.sbatch 03_sft.sbatch 04_opd.sbatch; do
  grep -Eq '^#SBATCH --gres=gpu:h200:4$' "${SCRIPT_DIR}/${script}" || {
    echo "GPU contract violation in ${script}" >&2
    exit 1
  }
done

if [[ -e "${MANIFEST_ROOT}/submission.json" || -e "${MANIFEST_ROOT}/submission.pending.json" || -e "${MANIFEST_ROOT}/submission.failed.json" ]]; then
  echo "Refusing to overwrite a prior or partial submission under ${MANIFEST_ROOT}" >&2
  exit 1
fi
if find "${OUTPUT_ROOT}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing nonempty output root for a new contract: ${OUTPUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${SLURM_LOG_DIR}" "${MANIFEST_ROOT}" "${OUTPUT_ROOT}" "${TMPDIR}"
export SUBMISSION_PENDING_PATH="${MANIFEST_ROOT}/submission.pending.json"
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["SUBMISSION_PENDING_PATH"])
value = {"status": "submitting", "started_at": datetime.now(timezone.utc).isoformat(), "contract_hash": os.environ["CONTRACT_HASH"]}
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temp = Path(handle.name)
temp.replace(path)
PY

submitted_jobs=()
submission_complete=0
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

record_failed_submission() {
  local status="$1"
  export FAILED_SUBMISSION_STATUS="${status}"
  export FAILED_SUBMITTED_JOB_IDS="${submitted_jobs[*]:-}"
  python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

pending = Path(os.environ["SUBMISSION_PENDING_PATH"])
failed = pending.with_name("submission.failed.json")
value = json.loads(pending.read_text()) if pending.is_file() else {}
value.update(
    {
        "status": "submission_failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "shell_status": int(os.environ["FAILED_SUBMISSION_STATUS"]),
        "accepted_jobs_canceled": os.environ.get("FAILED_SUBMITTED_JOB_IDS", "").split(),
    }
)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=failed.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temp = Path(handle.name)
temp.replace(failed)
pending.unlink(missing_ok=True)
PY
}

cancel_partial_submission() {
  local status=$?
  trap - EXIT
  if [[ "${submission_complete}" != "1" ]]; then
    if [[ "${#submitted_jobs[@]}" -gt 0 ]]; then
      echo "Submission failed; canceling accepted partial jobs: ${submitted_jobs[*]}" >&2
      scancel "${submitted_jobs[@]}" || echo "Warning: partial-job cancellation returned nonzero." >&2
    fi
    record_failed_submission "${status}" || echo "Warning: failed-submission manifest could not be written." >&2
  fi
  exit "${status}"
}
trap cancel_partial_submission EXIT

submit_job() {
  local output_name="$1"
  local dependency="$2"
  shift 2
  local -a args=(--parsable --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" --partition="${PARTITION}" --qos="${QOS}")
  if [[ -n "${dependency}" ]]; then args+=(--dependency="afterok:${dependency}"); fi
  local raw
  if ! raw="$("${SBATCH_BIN}" "${args[@]}" "$@")"; then
    echo "sbatch failed for $*" >&2
    return 1
  fi
  local job_id="${raw%%;*}"
  if [[ ! "${job_id}" =~ ^[0-9]+$ ]]; then
    echo "Unexpected sbatch job ID: ${raw}" >&2
    return 1
  fi
  submitted_jobs+=("${job_id}")
  printf -v "${output_name}" '%s' "${job_id}"
}

preflight_job=""
cleanup_job=""
base_eval_job=""
teacher_eval_job=""
sft_job=""
sft_eval_job=""
opd_job=""
opd_eval_job=""
submit_job preflight_job "" --job-name=q8b32b-preflight "${SCRIPT_DIR}/00_preflight.sbatch"
submit_job cleanup_job "${preflight_job}" --job-name=q8b32b-cleanup "${SCRIPT_DIR}/01_cleanup.sbatch"
submit_job base_eval_job "${cleanup_job}" --job-name=q8b32b-eval-base --export=ALL,EVAL_STAGE=qwen3_8b_base,EVAL_MODEL_DIR="${STUDENT_HF_DIR}" "${SCRIPT_DIR}/02_eval_math500.sbatch"
submit_job teacher_eval_job "${base_eval_job}" --job-name=q8b32b-eval-teacher --export=ALL,EVAL_STAGE=qwen3_32b_teacher,EVAL_MODEL_DIR="${TEACHER_HF_DIR}" "${SCRIPT_DIR}/02_eval_math500.sbatch"
submit_job sft_job "${teacher_eval_job}" --job-name=q8b32b-sft200k "${SCRIPT_DIR}/03_sft.sbatch"
submit_job sft_eval_job "${sft_job}" --job-name=q8b32b-eval-sft --export=ALL,EVAL_STAGE=sft_200000,EVAL_MODEL_DIR="${SFT_FINAL_HF_DIR}" "${SCRIPT_DIR}/02_eval_math500.sbatch"
submit_job opd_job "${sft_eval_job}" --job-name=q8b32b-opd1472 "${SCRIPT_DIR}/04_opd.sbatch"
submit_job opd_eval_job "${opd_job}" --job-name=q8b32b-eval-opd --export=ALL,EVAL_STAGE=opd_100pct,EVAL_MODEL_DIR="${OPD_FINAL_HF_DIR}",EVAL_COMPARE_TO="${EVAL_OUTPUT_ROOT}/sft_200000/summary.json",BUILD_FINAL_REPORT=1 "${SCRIPT_DIR}/02_eval_math500.sbatch"

export preflight_job cleanup_job base_eval_job teacher_eval_job sft_job sft_eval_job opd_job opd_eval_job
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["MANIFEST_ROOT"]) / "submission.json"
value = {
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "contract_hash": os.environ["CONTRACT_HASH"],
    "commit": os.popen("git rev-parse HEAD").read().strip(),
    "gpu_contract": "every job requests exactly four H200 GPUs",
    "jobs": {
        "preflight": os.environ["preflight_job"],
        "cleanup": os.environ["cleanup_job"],
        "base_eval": os.environ["base_eval_job"],
        "teacher_eval": os.environ["teacher_eval_job"],
        "sft_200k": os.environ["sft_job"],
        "sft_eval": os.environ["sft_eval_job"],
        "opd_1472x4": os.environ["opd_job"],
        "opd_eval_100pct": os.environ["opd_eval_job"],
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temp = Path(handle.name)
temp.replace(path)
Path(os.environ["SUBMISSION_PENDING_PATH"]).unlink()
print(json.dumps(value, indent=2, sort_keys=True))
PY

submission_complete=1
trap - EXIT

echo "Submitted serial afterok chain ending in job ${opd_eval_job}."
