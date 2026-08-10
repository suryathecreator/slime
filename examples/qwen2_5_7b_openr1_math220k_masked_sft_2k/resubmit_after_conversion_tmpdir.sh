#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

readonly FAILED_MODEL_JOB=198594
readonly BLOCKED_DATA_JOB=198095
readonly NEXT_SMOKE_JOB=198096
readonly TAIL_TRAIN_JOB=198108
readonly PRIOR_REPAIR_MANIFEST="${MANIFEST_ROOT}/model_validation_repair.json"
readonly FAILED_LOG="${SLURM_LOG_DIR}/q25-7b-model-r1_${FAILED_MODEL_JOB}.log"
readonly REPAIR_STEM="conversion_tmpdir_repair"
readonly REPAIR_MANIFEST="${MANIFEST_ROOT}/${REPAIR_STEM}.json"
readonly REPAIR_PENDING="${MANIFEST_ROOT}/${REPAIR_STEM}.pending.json"
readonly REPAIR_FAILED="${MANIFEST_ROOT}/${REPAIR_STEM}.failed.json"
readonly PARTIAL_DIR="${STUDENT_TORCH_DIST_DIR}"
readonly PARTIAL_COMMON="${PARTIAL_DIR}/iter_0000001/common.pt"
readonly FAILED_ARCHIVE="${EXPERIMENT_ROOT}/failed_attempts/model_conversion_job_${FAILED_MODEL_JOB}"
readonly BASE_MANIFEST="${EXPERIMENT_ROOT}/handoff/checkpoints/base_7b.json"

python3 "${SCRIPT_DIR}/validate_config.py" \
  --example-dir "${SCRIPT_DIR}" --require-pushed-clean

for path in \
  "${REPAIR_MANIFEST}" \
  "${REPAIR_PENDING}" \
  "${REPAIR_FAILED}" \
  "${FAILED_ARCHIVE}"; do
  [[ ! -e "${path}" ]] || {
    echo "Refusing to overwrite prior repair state: ${path}" >&2
    exit 1
  }
done

job_state() {
  sacct -X -n -P --jobs="$1" --format=State | sed -n '1{s/|.*//;p}'
}
[[ "$(job_state "${FAILED_MODEL_JOB}")" == FAILED* ]] || {
  echo "Expected failed replacement model job ${FAILED_MODEL_JOB}." >&2
  exit 1
}
[[ -f "${FAILED_LOG}" ]] || {
  echo "Missing failed replacement log: ${FAILED_LOG}" >&2
  exit 1
}
grep -Fq "OSError: AF_UNIX path too long" "${FAILED_LOG}" || {
  echo "Replacement log does not contain the approved AF_UNIX failure." >&2
  exit 1
}
grep -Fq "tools/convert_hf_to_torch_dist.py FAILED" "${FAILED_LOG}" || {
  echo "Replacement log does not identify the failed conversion." >&2
  exit 1
}

blocked_record="$(scontrol show job -o "${BLOCKED_DATA_JOB}")"
[[ "${blocked_record}" == *"JobName=q25-7b-data "* ]] || {
  echo "Unexpected blocked job identity: ${blocked_record}" >&2
  exit 1
}
[[ "${blocked_record}" == *"JobState=PENDING "* ]] || {
  echo "Expected blocked data job ${BLOCKED_DATA_JOB} to remain pending." >&2
  exit 1
}
[[ "${blocked_record}" == *"Dependency=afterok:${FAILED_MODEL_JOB}(failed)"* ]] || {
  echo "Blocked data dependency changed; refusing repair." >&2
  exit 1
}
smoke_record="$(scontrol show job -o "${NEXT_SMOKE_JOB}")"
[[ "${smoke_record}" == *"Dependency=afterok:${BLOCKED_DATA_JOB}(unfulfilled)"* ]] || {
  echo "Smoke dependency changed; refusing repair." >&2
  exit 1
}
tail_record="$(scontrol show job -o "${TAIL_TRAIN_JOB}")"
[[ "${tail_record}" == *"Dependency=afterok:198107(unfulfilled)"* ]] || {
  echo "Tail dependency changed; refusing repair." >&2
  exit 1
}

export PRIOR_REPAIR_MANIFEST
python3 - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["PRIOR_REPAIR_MANIFEST"]).read_text())
if manifest["status"] != "submitted_and_rewired":
    raise RuntimeError("prior model-validation repair is not finalized")
if manifest["replacement_model_preparation"] != 198594:
    raise RuntimeError("prior replacement job drift")
if manifest["rewired_dependency"] != {
    "job_id": 198095,
    "dependency": "afterok:198594",
}:
    raise RuntimeError("prior repaired dependency drift")
if manifest["reused_jobs"] != list(range(198095, 198109)):
    raise RuntimeError("prior reused job inventory drift")
PY

[[ -d "${PARTIAL_DIR}" ]] || {
  echo "Missing failed partial conversion: ${PARTIAL_DIR}" >&2
  exit 1
}
[[ -f "${PARTIAL_COMMON}" ]] || {
  echo "Missing expected partial checkpoint marker: ${PARTIAL_COMMON}" >&2
  exit 1
}
[[ ! -e "${PARTIAL_DIR}/latest_checkpointed_iteration.txt" ]] || {
  echo "Partial conversion unexpectedly has a resumable checkpoint pointer." >&2
  exit 1
}
[[ "$(find "${PARTIAL_DIR}" -type f | wc -l)" == "1" ]] || {
  echo "Partial conversion inventory drifted; refusing repair." >&2
  exit 1
}
[[ ! -e "${DATA_ROOT}" && ! -e "${OUTPUT_ROOT}" && ! -e "${BASE_MANIFEST}" ]] || {
  echo "Downstream artifacts unexpectedly exist; refusing narrow repair." >&2
  exit 1
}

export FAILED_LOG_SHA256 PARTIAL_COMMON_SHA256 PARTIAL_FILE_COUNT PARTIAL_BYTES
FAILED_LOG_SHA256="$(sha256sum "${FAILED_LOG}" | awk '{print $1}')"
PARTIAL_COMMON_SHA256="$(sha256sum "${PARTIAL_COMMON}" | awk '{print $1}')"
PARTIAL_FILE_COUNT="$(find "${PARTIAL_DIR}" -type f | wc -l)"
PARTIAL_BYTES="$(du -sb "${PARTIAL_DIR}" | awk '{print $1}')"
export REPAIR_COMMIT
REPAIR_COMMIT="$(git rev-parse HEAD)"
mkdir -p "${MANIFEST_ROOT}" "$(dirname -- "${FAILED_ARCHIVE}")"

export REPAIR_PENDING REPAIR_FAILED FAILED_ARCHIVE FAILED_LOG PARTIAL_COMMON
export FAILED_MODEL_JOB BLOCKED_DATA_JOB NEXT_SMOKE_JOB TAIL_TRAIN_JOB
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["REPAIR_PENDING"])
value = {
    "artifact_schema_version": 1,
    "status": "archiving_failed_conversion",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "commit": os.environ["REPAIR_COMMIT"],
    "contract_hash": os.environ["CONTRACT_HASH"],
    "reason": (
        "multiprocessing.Manager AF_UNIX path too long while saving the "
        "converted base torch_dist checkpoint"
    ),
    "jobs": {
        "failed_model_preparation": int(os.environ["FAILED_MODEL_JOB"]),
        "blocked_data_to_rewire": int(os.environ["BLOCKED_DATA_JOB"]),
        "preserved_downstream_first": int(os.environ["NEXT_SMOKE_JOB"]),
        "preserved_downstream_last": int(os.environ["TAIL_TRAIN_JOB"]),
    },
    "failed_log": {
        "path": os.environ["FAILED_LOG"],
        "sha256": os.environ["FAILED_LOG_SHA256"],
    },
    "failed_conversion": {
        "original": os.environ["STUDENT_TORCH_DIST_DIR"],
        "archive": os.environ["FAILED_ARCHIVE"],
        "file_count": int(os.environ["PARTIAL_FILE_COUNT"]),
        "bytes": int(os.environ["PARTIAL_BYTES"]),
        "partial_common_original": os.environ["PARTIAL_COMMON"],
        "partial_common_archive": str(
            Path(os.environ["FAILED_ARCHIVE"]) / "iter_0000001/common.pt"
        ),
        "partial_common_sha256": os.environ["PARTIAL_COMMON_SHA256"],
        "resumable": False,
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(path)
PY

replacement_job=""
dependency_rewired=0
repair_complete=0
record_failed_repair() {
  local shell_status="$1"
  export REPAIR_SHELL_STATUS="${shell_status}" REPLACEMENT_JOB="${replacement_job}"
  python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

pending = Path(os.environ["REPAIR_PENDING"])
failed = Path(os.environ["REPAIR_FAILED"])
value = json.loads(pending.read_text()) if pending.is_file() else {}
value.update(
    {
        "status": "repair_failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "shell_status": int(os.environ["REPAIR_SHELL_STATUS"]),
        "replacement_job": (
            int(os.environ["REPLACEMENT_JOB"])
            if os.environ.get("REPLACEMENT_JOB")
            else None
        ),
    }
)
with tempfile.NamedTemporaryFile("w", dir=failed.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(failed)
pending.unlink(missing_ok=True)
PY
}
fail_closed() {
  local status=$?
  trap - EXIT
  if [[ "${repair_complete}" != "1" ]]; then
    if [[ -n "${replacement_job}" && "${dependency_rewired}" != "1" ]]; then
      scancel "${replacement_job}" || true
    fi
    record_failed_repair "${status}" || true
  fi
  exit "${status}"
}
trap fail_closed EXIT

mv -- "${PARTIAL_DIR}" "${FAILED_ARCHIVE}"
[[ "$(sha256sum "${FAILED_ARCHIVE}/iter_0000001/common.pt" | awk '{print $1}')" == "${PARTIAL_COMMON_SHA256}" ]] || {
  echo "Archived partial checkpoint hash mismatch." >&2
  exit 1
}

raw="$(sbatch \
  --parsable \
  --chdir="${SLIME_REPO_ROOT}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name=q25-7b-model-r2 \
  "${SCRIPT_DIR}/00_prepare_model.sbatch")"
replacement_job="${raw%%;*}"
[[ "${replacement_job}" =~ ^[0-9]+$ ]] || {
  echo "Unexpected sbatch response: ${raw}" >&2
  exit 1
}

scontrol update \
  JobId="${BLOCKED_DATA_JOB}" \
  Dependency="afterok:${replacement_job}"
dependency_rewired=1

rewired_record="$(scontrol show job -o "${BLOCKED_DATA_JOB}")"
[[ "${rewired_record}" == *"JobState=PENDING "* ]] || {
  echo "Rewired data job is not pending: ${rewired_record}" >&2
  exit 1
}
[[ "${rewired_record}" == *"Dependency=afterok:${replacement_job}(unfulfilled)"* ]] || {
  echo "Data job was not rewired to ${replacement_job}: ${rewired_record}" >&2
  exit 1
}
smoke_record="$(scontrol show job -o "${NEXT_SMOKE_JOB}")"
[[ "${smoke_record}" == *"Dependency=afterok:${BLOCKED_DATA_JOB}(unfulfilled)"* ]] || {
  echo "Smoke dependency changed during repair." >&2
  exit 1
}
tail_record="$(scontrol show job -o "${TAIL_TRAIN_JOB}")"
[[ "${tail_record}" == *"Dependency=afterok:198107(unfulfilled)"* ]] || {
  echo "Tail dependency changed during repair." >&2
  exit 1
}

export REPLACEMENT_JOB="${replacement_job}" REPAIR_MANIFEST
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

pending = Path(os.environ["REPAIR_PENDING"])
output = Path(os.environ["REPAIR_MANIFEST"])
value = json.loads(pending.read_text())
value.update(
    {
        "status": "submitted_and_rewired",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "replacement_model_preparation": int(os.environ["REPLACEMENT_JOB"]),
        "rewired_dependency": {
            "job_id": int(os.environ["BLOCKED_DATA_JOB"]),
            "dependency": f"afterok:{os.environ['REPLACEMENT_JOB']}",
        },
        "reused_jobs": list(range(198095, 198109)),
    }
)
with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(output)
pending.unlink()
PY

repair_complete=1
trap - EXIT
echo "CONVERSION_TMPDIR_REPAIR_COMPLETE failed=${FAILED_MODEL_JOB} replacement=${replacement_job} rewired=${BLOCKED_DATA_JOB} reused=198095-198108"
