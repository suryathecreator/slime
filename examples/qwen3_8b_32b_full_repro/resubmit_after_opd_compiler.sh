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

source_manifest="${RECOVERY_SOURCE_MANIFEST:-${MANIFEST_ROOT}/resubmission_after_sft_python_include.json}"
resubmission_stem="${RECOVERY_MANIFEST_STEM:-resubmission_after_opd_compiler_and_32k_proxy}"
recovery_reason="${RECOVERY_REASON:-OPD teacher health generation failed because CC and CPATH were not forwarded}"
[[ "${resubmission_stem}" =~ ^resubmission_[a-z0-9_]+$ ]] || { echo "Invalid recovery manifest stem: ${resubmission_stem}" >&2; exit 1; }
resubmission_path="${MANIFEST_ROOT}/${resubmission_stem}.json"
resubmission_pending="${MANIFEST_ROOT}/${resubmission_stem}.pending.json"
resubmission_failed="${MANIFEST_ROOT}/${resubmission_stem}.failed.json"

[[ -f "${source_manifest}" ]] || { echo "Missing source submission manifest: ${source_manifest}" >&2; exit 1; }
for path in "${resubmission_path}" "${resubmission_pending}" "${resubmission_failed}"; do
  [[ ! -e "${path}" ]] || { echo "Refusing to overwrite prior recovery state: ${path}" >&2; exit 1; }
done

export RECOVERY_SOURCE_MANIFEST="${source_manifest}"
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path

source = Path(os.environ["RECOVERY_SOURCE_MANIFEST"])
manifest = json.loads(source.read_text())
if manifest.get("contract_hash") != os.environ["CONTRACT_HASH"]:
    raise SystemExit("Source submission contract hash does not match the active contract")
expected_jobs = {
    "base_eval": "181406",
    "teacher_eval": "181407",
    "sft_200k": "181911",
    "sft_eval": "181912",
    "opd_1472x4": "181913",
    "opd_eval_100pct": "181914",
}
if manifest.get("jobs") != expected_jobs:
    raise SystemExit(f"Unexpected source job topology: {manifest.get('jobs')}")
tracked = json.loads((Path(os.environ["FULL_REPRO_DIR"]) / "results.json").read_text())
for stage in ("qwen3_8b_base", "qwen3_32b_teacher", "sft_200000"):
    expected = tracked["evaluations"][stage]
    summary_path = Path(expected["summary_path"])
    raw = summary_path.read_bytes()
    summary = json.loads(raw)
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != expected["summary_sha256"]:
        raise SystemExit(f"Completed evaluation summary hash changed for {stage}: {actual_sha}")
    if summary.get("stage") != stage or summary.get("total") != 500:
        raise SystemExit(f"Completed evaluation summary identity changed for {stage}")
checkpoint_report = Path(os.environ["CHECKPOINT_REPORT_DIR"]) / "sft_manifest.json"
expected_checkpoint_sha = tracked["evaluations"]["sft_200000"]["checkpoint_report_sha256"]
if hashlib.sha256(checkpoint_report.read_bytes()).hexdigest() != expected_checkpoint_sha:
    raise SystemExit("SFT checkpoint report hash changed")
opd = tracked["evaluations"]["opd_100pct"]
opd_log = Path(os.environ["SLURM_LOG_DIR"]) / "q8b32b-opd1472_181913.log"
teacher_log = Path(os.environ["TEACHER_LOG_DIR"]) / "sglang_teacher_181913.log"
if hashlib.sha256(opd_log.read_bytes()).hexdigest() != opd["opd_log_sha256"]:
    raise SystemExit("Failed OPD log hash changed")
if hashlib.sha256(teacher_log.read_bytes()).hexdigest() != opd["teacher_log_sha256"]:
    raise SystemExit("Failed OPD teacher log hash changed")
PY

for path in "${OPD_SAVE_DIR}" "${OPD_HF_SNAPSHOT_DIR}" "${OPD_ROLLOUT_LOG_DIR}"; do
  if [[ -d "${path}" ]] && find "${path}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing OPD recovery because scientific output already exists: ${path}" >&2
    exit 1
  fi
done
for stage in opd_100pct opd_100pct_32k_proxy sft_200000_32k_proxy qwen3_32b_teacher_32k_proxy qwen3_8b_base_32k_proxy; do
  path="${EVAL_OUTPUT_ROOT}/${stage}"
  if [[ -d "${path}" ]] && find "${path}" -mindepth 1 -print -quit | grep -q .; then
    echo "Refusing recovery because target evaluation output already exists: ${path}" >&2
    exit 1
  fi
done

SACCT_BIN="${SACCT_BIN:-sacct}"
SCONTROL_BIN="${SCONTROL_BIN:-scontrol}"
SCANCEL_BIN="${SCANCEL_BIN:-scancel}"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
job_state() {
  local job_id="$1"
  "${SACCT_BIN}" -X -n -P --jobs="${job_id}" --format=State | sed -n '1{s/|.*//;p}'
}
for job_id in 181406 181407 181911 181912; do
  [[ "$(job_state "${job_id}")" == COMPLETED* ]] || { echo "Expected completed source job ${job_id}." >&2; exit 1; }
done
[[ "$(job_state 181913)" == FAILED* ]] || { echo "Expected failed OPD job 181913." >&2; exit 1; }
pending_record="$("${SCONTROL_BIN}" show job -o 181914)"
[[ "${pending_record}" == *"JobName=q8b32b-eval-opd "* ]] || { echo "Unexpected job identity for 181914." >&2; exit 1; }
[[ "${pending_record}" == *"JobState=PENDING "* ]] || { echo "Expected pending superseded job 181914." >&2; exit 1; }
[[ "${pending_record}" == *"Reason=DependencyNeverSatisfied "* ]] || { echo "Expected DependencyNeverSatisfied for 181914." >&2; exit 1; }

failed_sanity_files=()
for extension in json csv md; do
  path="${OPD_SANITY_SUMMARY_DIR}/opd_sanity_summary.${extension}"
  archive="${path}.failed_181913"
  [[ -f "${path}" ]] || { echo "Missing failed OPD sanity artifact: ${path}" >&2; exit 1; }
  [[ ! -e "${archive}" ]] || { echo "Refusing to overwrite failed OPD sanity archive: ${archive}" >&2; exit 1; }
  failed_sanity_files+=("${path}")
done

mkdir -p "${MANIFEST_ROOT}" "${SLURM_LOG_DIR}"
export RESUBMISSION_PENDING_PATH="${resubmission_pending}"
export RESUBMISSION_FAILED_PATH="${resubmission_failed}"
export RESUBMISSION_FINAL_PATH="${resubmission_path}"
export RECOVERY_REASON="${recovery_reason}"
export FAILED_SANITY_PATHS="${failed_sanity_files[*]}"
python3 - <<'PY'
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

source = Path(os.environ["RECOVERY_SOURCE_MANIFEST"])
sanity = []
for raw_path in os.environ["FAILED_SANITY_PATHS"].split():
    path = Path(raw_path)
    sanity.append(
        {
            "source": str(path),
            "archive": str(path) + ".failed_181913",
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    )
value = {
    "status": "preserving_failed_opd_and_canceling_descendant",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "contract_hash": os.environ["CONTRACT_HASH"],
    "launch_commit": os.environ["LAUNCH_COMMIT"],
    "source_manifest": str(source),
    "source_manifest_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
    "failed_opd_job": "181913",
    "superseded_pending_jobs": ["181914"],
    "failed_sanity_artifacts": sanity,
    "reused_completed_jobs": {
        "base_eval": "181406",
        "teacher_eval": "181407",
        "sft_200k": "181911",
        "sft_eval": "181912",
    },
}
path = Path(os.environ["RESUBMISSION_PENDING_PATH"])
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temp = Path(handle.name)
temp.replace(path)
PY

submitted_jobs=()
submission_complete=0
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
      "${SCANCEL_BIN}" "${submitted_jobs[@]}" || echo "Warning: replacement-job cancellation returned nonzero." >&2
    fi
    record_failed_resubmission "${status}" || echo "Warning: failed-resubmission manifest could not be written." >&2
  fi
  exit "${status}"
}
trap cancel_partial_resubmission EXIT

for path in "${failed_sanity_files[@]}"; do
  mv -- "${path}" "${path}.failed_181913"
done
"${SCANCEL_BIN}" 181914

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

opd_job=""
opd_eval_job=""
opd_proxy_job=""
sft_proxy_job=""
teacher_proxy_job=""
base_proxy_job=""
submit_job opd_job "" --job-name=q8b32b-opd1472 "${SCRIPT_DIR}/04_opd.sbatch"
submit_job opd_eval_job "${opd_job}" --job-name=q8b32b-eval-opd --export=ALL,EVAL_STAGE=opd_100pct,EVAL_MODEL_DIR="${OPD_FINAL_HF_DIR}",EVAL_COMPARE_TO="${EVAL_OUTPUT_ROOT}/sft_200000/summary.json" "${SCRIPT_DIR}/02_eval_math500.sbatch"
submit_job opd_proxy_job "${opd_eval_job}" --job-name=q8b32b-proxy-opd --export=ALL,CAP_PROXY_STAGE=opd_100pct_32k_proxy,CAP_PROXY_SOURCE_DIR="${EVAL_OUTPUT_ROOT}/opd_100pct",CAP_PROXY_MODEL_DIR="${OPD_FINAL_HF_DIR}" "${SCRIPT_DIR}/05_eval_cap_hits_32k.sbatch"
submit_job sft_proxy_job "${opd_proxy_job}" --job-name=q8b32b-proxy-sft --export=ALL,CAP_PROXY_STAGE=sft_200000_32k_proxy,CAP_PROXY_SOURCE_DIR="${EVAL_OUTPUT_ROOT}/sft_200000",CAP_PROXY_MODEL_DIR="${SFT_FINAL_HF_DIR}" "${SCRIPT_DIR}/05_eval_cap_hits_32k.sbatch"
submit_job teacher_proxy_job "${sft_proxy_job}" --job-name=q8b32b-proxy-teacher --export=ALL,CAP_PROXY_STAGE=qwen3_32b_teacher_32k_proxy,CAP_PROXY_SOURCE_DIR="${EVAL_OUTPUT_ROOT}/qwen3_32b_teacher",CAP_PROXY_MODEL_DIR="${TEACHER_HF_DIR}" "${SCRIPT_DIR}/05_eval_cap_hits_32k.sbatch"
submit_job base_proxy_job "${teacher_proxy_job}" --job-name=q8b32b-proxy-base --export=ALL,CAP_PROXY_STAGE=qwen3_8b_base_32k_proxy,CAP_PROXY_SOURCE_DIR="${EVAL_OUTPUT_ROOT}/qwen3_8b_base",CAP_PROXY_MODEL_DIR="${STUDENT_HF_DIR}",BUILD_FINAL_REPORT=1 "${SCRIPT_DIR}/05_eval_cap_hits_32k.sbatch"

export opd_job opd_eval_job opd_proxy_job sft_proxy_job teacher_proxy_job base_proxy_job
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

pending = Path(os.environ["RESUBMISSION_PENDING_PATH"])
path = Path(os.environ["RESUBMISSION_FINAL_PATH"])
value = json.loads(pending.read_text())
jobs = {
    "opd_1472x4": os.environ["opd_job"],
    "opd_eval_100pct_fixed_16k_output": os.environ["opd_eval_job"],
    "opd_100pct_32k_proxy": os.environ["opd_proxy_job"],
    "sft_200000_32k_proxy": os.environ["sft_proxy_job"],
    "qwen3_32b_teacher_32k_proxy": os.environ["teacher_proxy_job"],
    "qwen3_8b_base_32k_proxy_and_final_report": os.environ["base_proxy_job"],
}
value.update(
    {
        "status": "submitted",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "reason": os.environ["RECOVERY_REASON"],
        "commit": os.environ["LAUNCH_COMMIT"],
        "jobs": jobs,
        "serial_order": list(jobs),
        "primary_eval_response_budget": "min(16384, 32768 - rendered_prompt_tokens - 64)",
        "proxy_eval_response_budget": "32768 - rendered_prompt_tokens - 64",
        "proxy_context_accounting": "rendered prompt tokens + generated response tokens + 64 reserved safety tokens",
        "proxy_exactness_gate": "all cap-hit reruns must exactly reproduce the complete saved source generated-token prefix",
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
echo "Submitted serial OPD recovery and 32K cap-hit proxy chain ending in job ${base_proxy_job}."
