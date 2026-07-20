#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

[[ "$(git branch --show-current)" == "qwen3-8b-32b-full-repro" ]] || { echo "Wrong branch." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing submission from a dirty worktree." >&2; exit 1; }
python3 "${SCRIPT_DIR}/validate_config.py" --require-pinned-commit

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
[[ -n "${upstream}" ]] || { echo "Push this branch and set its upstream before resubmission." >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$(git rev-parse "${upstream}")" ]] || { echo "Local HEAD is not the pushed upstream commit." >&2; exit 1; }

original_manifest="${MANIFEST_ROOT}/submission.json"
resubmission_path="${MANIFEST_ROOT}/resubmission_after_base_eval.json"
resubmission_pending="${MANIFEST_ROOT}/resubmission_after_base_eval.pending.json"
resubmission_failed="${MANIFEST_ROOT}/resubmission_after_base_eval.failed.json"

[[ -f "${original_manifest}" ]] || { echo "Missing original submission manifest: ${original_manifest}" >&2; exit 1; }
for path in "${resubmission_path}" "${resubmission_pending}" "${resubmission_failed}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing to overwrite prior recovery state: ${path}" >&2; exit 1; }
done

export ORIGINAL_SUBMISSION_MANIFEST="${original_manifest}"
python3 - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["ORIGINAL_SUBMISSION_MANIFEST"]).read_text())
if manifest.get("contract_hash") != os.environ["CONTRACT_HASH"]:
    raise SystemExit("Original submission contract hash does not match the active contract")
required = {"base_eval", "teacher_eval", "sft_200k", "sft_eval", "opd_1472x4", "opd_eval_100pct"}
if not required.issubset(manifest.get("jobs", {})):
    raise SystemExit("Original submission manifest is missing recovery job IDs")

audit = json.loads(Path(os.environ["CLEANUP_AUDIT_JSON"]).read_text())
if not audit.get("validation_gate", {}).get("passed"):
    raise SystemExit("Cleanup audit did not pass; refusing base-eval recovery")
if audit.get("counts", {}).get("retained", 0) < 300_000:
    raise SystemExit("Cleanup audit retained fewer than 300,000 rows")
PY

for path in \
  "${EVAL_ZIG_CC}" \
  "${EVAL_ZIG_INCLUDE_ROOT}/python3.12/Python.h" \
  "${EVAL_ZIG_INCLUDE_ROOT}/x86_64-linux-gnu/python3.12/pyconfig.h"; do
  [[ -e "${path}" ]] || { echo "Missing recovery prerequisite: ${path}" >&2; exit 1; }
done
[[ -x "${EVAL_ZIG_CC}" ]] || { echo "Eval compiler is not executable: ${EVAL_ZIG_CC}" >&2; exit 1; }

for path in "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${OPD_SAVE_DIR}" "${OPD_HF_SNAPSHOT_DIR}"; do
  if [[ -d "${path}" ]] && find "${path}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing base-eval recovery because training output already exists: ${path}" >&2
    exit 1
  fi
done

mapfile -t original_jobs < <(python3 - <<'PY'
import json
import os
from pathlib import Path

jobs = json.loads(Path(os.environ["ORIGINAL_SUBMISSION_MANIFEST"]).read_text())["jobs"]
for key in ("base_eval", "teacher_eval", "sft_200k", "sft_eval", "opd_1472x4", "opd_eval_100pct"):
    print(jobs[key])
PY
)
failed_base_job="${original_jobs[0]}"
superseded_pending_jobs=("${original_jobs[@]:1}")
expected_names=(q8b32b-eval-teacher q8b32b-sft200k q8b32b-eval-sft q8b32b-opd1472 q8b32b-eval-opd)

failed_state="$(sacct -X -n -P --jobs="${failed_base_job}" --format=State | sed -n '1{s/|.*//;p}')"
[[ "${failed_state}" == FAILED* ]] || { echo "Expected failed base eval ${failed_base_job}; found state ${failed_state:-missing}." >&2; exit 1; }
for index in "${!superseded_pending_jobs[@]}"; do
  job_id="${superseded_pending_jobs[${index}]}"
  job_record="$(scontrol show job -o "${job_id}")"
  [[ "${job_record}" == *"JobName=${expected_names[${index}]} "* ]] || { echo "Unexpected job identity for ${job_id}." >&2; exit 1; }
  [[ "${job_record}" == *"JobState=PENDING "* ]] || { echo "Expected pending superseded job ${job_id}." >&2; exit 1; }
done

mkdir -p "${SLURM_LOG_DIR}" "${MANIFEST_ROOT}" "${OUTPUT_ROOT}" "${TMPDIR}"
export RESUBMISSION_PENDING_PATH="${resubmission_pending}"
export FAILED_BASE_JOB_ID="${failed_base_job}"
export SUPERSEDED_PENDING_JOB_IDS="${superseded_pending_jobs[*]}"
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["RESUBMISSION_PENDING_PATH"])
value = {
    "status": "canceling_superseded_chain",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "contract_hash": os.environ["CONTRACT_HASH"],
    "failed_base_eval": os.environ["FAILED_BASE_JOB_ID"],
    "superseded_pending_jobs": os.environ["SUPERSEDED_PENDING_JOB_IDS"].split(),
}
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temp = Path(handle.name)
temp.replace(path)
PY

submitted_jobs=()
submission_complete=0
SBATCH_BIN="${SBATCH_BIN:-sbatch}"

record_failed_resubmission() {
  local status="$1"
  export FAILED_RESUBMISSION_STATUS="${status}"
  export FAILED_RESUBMITTED_JOB_IDS="${submitted_jobs[*]:-}"
  python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

pending = Path(os.environ["RESUBMISSION_PENDING_PATH"])
failed = pending.with_name("resubmission_after_base_eval.failed.json")
value = json.loads(pending.read_text()) if pending.is_file() else {}
value.update(
    {
        "status": "resubmission_failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "shell_status": int(os.environ["FAILED_RESUBMISSION_STATUS"]),
        "accepted_replacement_jobs_canceled": os.environ.get("FAILED_RESUBMITTED_JOB_IDS", "").split(),
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

cancel_partial_resubmission() {
  local status=$?
  trap - EXIT
  if [[ "${submission_complete}" != "1" ]]; then
    if [[ "${#submitted_jobs[@]}" -gt 0 ]]; then
      echo "Resubmission failed; canceling accepted replacement jobs: ${submitted_jobs[*]}" >&2
      scancel "${submitted_jobs[@]}" || echo "Warning: replacement-job cancellation returned nonzero." >&2
    fi
    record_failed_resubmission "${status}" || echo "Warning: failed-resubmission manifest could not be written." >&2
  fi
  exit "${status}"
}
trap cancel_partial_resubmission EXIT

scancel "${superseded_pending_jobs[@]}"

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

base_eval_job=""
teacher_eval_job=""
sft_job=""
sft_eval_job=""
opd_job=""
opd_eval_job=""
submit_job base_eval_job "" --job-name=q8b32b-eval-base --export=ALL,EVAL_STAGE=qwen3_8b_base,EVAL_MODEL_DIR="${STUDENT_HF_DIR}" "${SCRIPT_DIR}/02_eval_math500.sbatch"
submit_job teacher_eval_job "${base_eval_job}" --job-name=q8b32b-eval-teacher --export=ALL,EVAL_STAGE=qwen3_32b_teacher,EVAL_MODEL_DIR="${TEACHER_HF_DIR}" "${SCRIPT_DIR}/02_eval_math500.sbatch"
submit_job sft_job "${teacher_eval_job}" --job-name=q8b32b-sft200k "${SCRIPT_DIR}/03_sft.sbatch"
submit_job sft_eval_job "${sft_job}" --job-name=q8b32b-eval-sft --export=ALL,EVAL_STAGE=sft_200000,EVAL_MODEL_DIR="${SFT_FINAL_HF_DIR}" "${SCRIPT_DIR}/02_eval_math500.sbatch"
submit_job opd_job "${sft_eval_job}" --job-name=q8b32b-opd1472 "${SCRIPT_DIR}/04_opd.sbatch"
submit_job opd_eval_job "${opd_job}" --job-name=q8b32b-eval-opd --export=ALL,EVAL_STAGE=opd_100pct,EVAL_MODEL_DIR="${OPD_FINAL_HF_DIR}",EVAL_COMPARE_TO="${EVAL_OUTPUT_ROOT}/sft_200000/summary.json",BUILD_FINAL_REPORT=1 "${SCRIPT_DIR}/02_eval_math500.sbatch"

export base_eval_job teacher_eval_job sft_job sft_eval_job opd_job opd_eval_job
export RESUBMISSION_FINAL_PATH="${resubmission_path}"
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

pending = Path(os.environ["RESUBMISSION_PENDING_PATH"])
path = Path(os.environ["RESUBMISSION_FINAL_PATH"])
value = json.loads(pending.read_text())
value.update(
    {
        "status": "submitted",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "reason": "base eval failed because Triton could not find a C compiler",
        "commit": os.popen("git rev-parse HEAD").read().strip(),
        "jobs": {
            "base_eval": os.environ["base_eval_job"],
            "teacher_eval": os.environ["teacher_eval_job"],
            "sft_200k": os.environ["sft_job"],
            "sft_eval": os.environ["sft_eval_job"],
            "opd_1472x4": os.environ["opd_job"],
            "opd_eval_100pct": os.environ["opd_eval_job"],
        },
    }
)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temp = Path(handle.name)
temp.replace(path)
pending.unlink()
print(json.dumps(value, indent=2, sort_keys=True))
PY

submission_complete=1
trap - EXIT
echo "Submitted replacement chain ending in job ${opd_eval_job}."
