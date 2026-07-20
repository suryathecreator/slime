#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

[[ "$(git branch --show-current)" == "qwen3-8b-32b-full-repro" ]] || { echo "Wrong branch." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing resubmission from a dirty worktree." >&2; exit 1; }
python3 "${SCRIPT_DIR}/validate_config.py" --require-pinned-commit

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
[[ -n "${upstream}" ]] || { echo "Push this branch and set its upstream before resubmission." >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$(git rev-parse "${upstream}")" ]] || { echo "Local HEAD is not the pushed upstream commit." >&2; exit 1; }
export LAUNCH_COMMIT="$(git rev-parse HEAD)"

source_manifest="${RECOVERY_SOURCE_MANIFEST:-${MANIFEST_ROOT}/resubmission_after_cuda_visibility.json}"
resubmission_stem="${RECOVERY_MANIFEST_STEM:-resubmission_after_sft_oom}"
recovery_reason="${RECOVERY_REASON:-SFT exhausted H200 memory while materializing 32K TP2 FP32 vocabulary logits}"
[[ "${resubmission_stem}" =~ ^resubmission_[a-z0-9_]+$ ]] || { echo "Invalid recovery manifest stem: ${resubmission_stem}" >&2; exit 1; }
resubmission_path="${MANIFEST_ROOT}/${resubmission_stem}.json"
resubmission_pending="${MANIFEST_ROOT}/${resubmission_stem}.pending.json"
resubmission_failed="${MANIFEST_ROOT}/${resubmission_stem}.failed.json"

[[ -f "${source_manifest}" ]] || { echo "Missing source submission manifest: ${source_manifest}" >&2; exit 1; }
for path in "${resubmission_path}" "${resubmission_pending}" "${resubmission_failed}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing to overwrite prior recovery state: ${path}" >&2; exit 1; }
done

export RECOVERY_SOURCE_MANIFEST="${source_manifest}"
export RECOVERY_REASON="${recovery_reason}"
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["RECOVERY_SOURCE_MANIFEST"]).read_text())
if manifest.get("contract_hash") != os.environ["CONTRACT_HASH"]:
    raise SystemExit("Source submission contract hash does not match the active contract")
required = {"base_eval", "teacher_eval", "sft_200k", "sft_eval", "opd_1472x4", "opd_eval_100pct"}
if not required.issubset(manifest.get("jobs", {})):
    raise SystemExit("Source submission manifest is missing recovery job IDs")

tracked = json.loads((Path(os.environ["FULL_REPRO_DIR"]) / "results.json").read_text())
checks = {
    "qwen3_8b_base": (500, 235, "ba252b5bfa78e7e5ccc6f803c9af4a9d32df7121b0d87001a2761646fbf4fef9"),
    "qwen3_32b_teacher": (500, 476, "d1d8ee1fed7472f81c7524764b7a255e52779eabe137ee1a8ff393e31ad5d103"),
}
for stage, (total, correct, expected_sha) in checks.items():
    summary_path = Path(os.environ["EVAL_OUTPUT_ROOT"]) / stage / "summary.json"
    if not summary_path.is_file():
        raise SystemExit(f"Missing completed evaluation summary: {summary_path}")
    raw = summary_path.read_bytes()
    summary = json.loads(raw)
    if summary.get("stage") != stage or summary.get("total") != total or summary.get("correct") != correct:
        raise SystemExit(f"Completed evaluation summary changed for {stage}")
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected_sha:
        raise SystemExit(f"Completed evaluation summary hash changed for {stage}: {actual_sha}")
    if tracked["evaluations"][stage]["summary_sha256"] != expected_sha:
        raise SystemExit(f"Tracked evaluation hash differs for {stage}")
PY

for path in "${SFT_SAVE_DIR}" "${SFT_HF_SNAPSHOT_DIR}" "${OPD_SAVE_DIR}" "${OPD_HF_SNAPSHOT_DIR}"; do
  if [[ -d "${path}" ]] && find "${path}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing SFT recovery because checkpoint output already exists: ${path}" >&2
    exit 1
  fi
done

mapfile -t source_jobs < <(python3 - <<'PY'
import json
import os
from pathlib import Path

jobs = json.loads(Path(os.environ["RECOVERY_SOURCE_MANIFEST"]).read_text())["jobs"]
for key in ("base_eval", "teacher_eval", "sft_200k", "sft_eval", "opd_1472x4", "opd_eval_100pct"):
    print(jobs[key])
PY
)
base_eval_job="${source_jobs[0]}"
teacher_eval_job="${source_jobs[1]}"
failed_sft_job="${source_jobs[2]}"
superseded_pending_jobs=("${source_jobs[@]:3}")
expected_pending_names=(q8b32b-eval-sft q8b32b-opd1472 q8b32b-eval-opd)

job_state() {
  local job_id="$1"
  sacct -X -n -P --jobs="${job_id}" --format=State | sed -n '1{s/|.*//;p}'
}
[[ "$(job_state "${base_eval_job}")" == COMPLETED* ]] || { echo "Expected completed base eval ${base_eval_job}." >&2; exit 1; }
[[ "$(job_state "${teacher_eval_job}")" == COMPLETED* ]] || { echo "Expected completed teacher eval ${teacher_eval_job}." >&2; exit 1; }
[[ "$(job_state "${failed_sft_job}")" == FAILED* ]] || { echo "Expected failed SFT ${failed_sft_job}." >&2; exit 1; }
for index in "${!superseded_pending_jobs[@]}"; do
  job_id="${superseded_pending_jobs[${index}]}"
  job_record="$(scontrol show job -o "${job_id}")"
  [[ "${job_record}" == *"JobName=${expected_pending_names[${index}]} "* ]] || { echo "Unexpected job identity for ${job_id}." >&2; exit 1; }
  [[ "${job_record}" == *"JobState=PENDING "* ]] || { echo "Expected pending superseded job ${job_id}." >&2; exit 1; }
done

failed_sft_log="${SLURM_LOG_DIR}/q8b32b-sft200k_${failed_sft_job}.log"
failed_rollout="${SFT_DETAILS_DIR}/rollout_data/1.pt"
failed_rollout_archive="${failed_rollout}.failed_${failed_sft_job}"
[[ -f "${failed_sft_log}" ]] || { echo "Missing failed SFT log: ${failed_sft_log}" >&2; exit 1; }
[[ -f "${failed_rollout}" ]] || { echo "Missing failed rollout tensor: ${failed_rollout}" >&2; exit 1; }
[[ ! -e "${failed_rollout_archive}" ]] || { echo "Refusing to overwrite failed rollout archive: ${failed_rollout_archive}" >&2; exit 1; }
failed_sft_log_sha256="$(sha256sum "${failed_sft_log}" | awk '{print $1}')"
failed_rollout_sha256="$(sha256sum "${failed_rollout}" | awk '{print $1}')"
if [[ -n "${RECOVERY_EXPECTED_FAILED_SFT_LOG_SHA256:-}" ]]; then
  [[ "${failed_sft_log_sha256}" == "${RECOVERY_EXPECTED_FAILED_SFT_LOG_SHA256}" ]] || { echo "Failed SFT log hash differs from the requested recovery hash." >&2; exit 1; }
fi
if [[ -n "${RECOVERY_EXPECTED_FAILED_ROLLOUT_SHA256:-}" ]]; then
  [[ "${failed_rollout_sha256}" == "${RECOVERY_EXPECTED_FAILED_ROLLOUT_SHA256}" ]] || { echo "Failed rollout tensor hash differs from the requested recovery hash." >&2; exit 1; }
fi

mkdir -p "${SLURM_LOG_DIR}" "${MANIFEST_ROOT}" "${OUTPUT_ROOT}" "${TMPDIR}"
export RESUBMISSION_PENDING_PATH="${resubmission_pending}"
export RESUBMISSION_FAILED_PATH="${resubmission_failed}"
export FAILED_SFT_JOB_ID="${failed_sft_job}"
export FAILED_SFT_LOG="${failed_sft_log}"
export FAILED_SFT_LOG_SHA256="${failed_sft_log_sha256}"
export FAILED_ROLLOUT_ARCHIVE="${failed_rollout_archive}"
export FAILED_ROLLOUT_SHA256="${failed_rollout_sha256}"
export REUSED_BASE_EVAL_JOB_ID="${base_eval_job}"
export REUSED_TEACHER_EVAL_JOB_ID="${teacher_eval_job}"
export SUPERSEDED_PENDING_JOB_IDS="${superseded_pending_jobs[*]}"
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["RESUBMISSION_PENDING_PATH"])
value = {
    "status": "preserving_diagnostic_and_canceling_descendants",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "contract_hash": os.environ["CONTRACT_HASH"],
    "launch_commit": os.environ["LAUNCH_COMMIT"],
    "source_manifest": os.environ["RECOVERY_SOURCE_MANIFEST"],
    "reused_completed_jobs": {
        "base_eval": os.environ["REUSED_BASE_EVAL_JOB_ID"],
        "teacher_eval": os.environ["REUSED_TEACHER_EVAL_JOB_ID"],
    },
    "failed_sft": os.environ["FAILED_SFT_JOB_ID"],
    "superseded_pending_jobs": os.environ["SUPERSEDED_PENDING_JOB_IDS"].split(),
    "diagnostic": {
        "sft_log": os.environ["FAILED_SFT_LOG"],
        "sft_log_sha256": os.environ["FAILED_SFT_LOG_SHA256"],
        "rollout_archive": os.environ["FAILED_ROLLOUT_ARCHIVE"],
        "rollout_sha256": os.environ["FAILED_ROLLOUT_SHA256"],
    },
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
failed = Path(os.environ["RESUBMISSION_FAILED_PATH"])
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

mv -- "${failed_rollout}" "${failed_rollout_archive}"
[[ "$(sha256sum "${failed_rollout_archive}" | awk '{print $1}')" == "${failed_rollout_sha256}" ]] || { echo "Archived rollout hash mismatch." >&2; exit 1; }
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

sft_job=""
sft_eval_job=""
opd_job=""
opd_eval_job=""
submit_job sft_job "" --job-name=q8b32b-sft200k "${SCRIPT_DIR}/03_sft.sbatch"
submit_job sft_eval_job "${sft_job}" --job-name=q8b32b-eval-sft --export=ALL,EVAL_STAGE=sft_200000,EVAL_MODEL_DIR="${SFT_FINAL_HF_DIR}" "${SCRIPT_DIR}/02_eval_math500.sbatch"
submit_job opd_job "${sft_eval_job}" --job-name=q8b32b-opd1472 "${SCRIPT_DIR}/04_opd.sbatch"
submit_job opd_eval_job "${opd_job}" --job-name=q8b32b-eval-opd --export=ALL,EVAL_STAGE=opd_100pct,EVAL_MODEL_DIR="${OPD_FINAL_HF_DIR}",EVAL_COMPARE_TO="${EVAL_OUTPUT_ROOT}/sft_200000/summary.json",BUILD_FINAL_REPORT=1 "${SCRIPT_DIR}/02_eval_math500.sbatch"

export sft_job sft_eval_job opd_job opd_eval_job
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
        "reason": os.environ["RECOVERY_REASON"],
        "commit": os.environ["LAUNCH_COMMIT"],
        "sft_dynamic_packing_max_tokens_per_gpu": int(os.environ["SFT_MAX_TOKENS_PER_GPU"]),
        "sft_native_context_length": int(os.environ["SFT_SEQ_LENGTH"]),
        "jobs": {
            "base_eval": os.environ["REUSED_BASE_EVAL_JOB_ID"],
            "teacher_eval": os.environ["REUSED_TEACHER_EVAL_JOB_ID"],
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
echo "Submitted SFT recovery chain ending in job ${opd_eval_job}."
