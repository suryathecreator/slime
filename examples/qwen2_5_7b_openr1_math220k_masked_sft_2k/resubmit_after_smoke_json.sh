#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

readonly COMPLETED_MODEL_JOB=199182
readonly COMPLETED_DATA_JOB=198095
readonly FAILED_SMOKE_JOB=198096
readonly BLOCKED_SCORE_JOB=198097
readonly FIRST_TRAIN_JOB=198098
readonly TAIL_TRAIN_JOB=198108
readonly PRIOR_REPAIR_MANIFEST="${MANIFEST_ROOT}/conversion_tmpdir_repair.json"
readonly FAILED_LOG="${SLURM_LOG_DIR}/q25-7b-smoke_${FAILED_SMOKE_JOB}.log"
readonly REPAIR_STEM="smoke_json_repair"
readonly REPAIR_MANIFEST="${MANIFEST_ROOT}/${REPAIR_STEM}.json"
readonly REPAIR_PENDING="${MANIFEST_ROOT}/${REPAIR_STEM}.pending.json"
readonly REPAIR_FAILED="${MANIFEST_ROOT}/${REPAIR_STEM}.failed.json"
readonly FAILED_OUTPUT="${OUTPUT_ROOT}/smoke"
readonly FAILED_ARCHIVE="${OUTPUT_ROOT}/failed_attempts/smoke_job_${FAILED_SMOKE_JOB}"
readonly PARTIAL_DATA="${FAILED_OUTPUT}/correct_only_first_200.jsonl"
readonly EXPECTED_PARTIAL_DATA_SHA256="78d2ba09a34bd386a7bb17c17d2071bdacb85a185e6be2c06b1b827870d26d9a"
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
[[ "$(job_state "${COMPLETED_MODEL_JOB}")" == COMPLETED* ]] || {
  echo "Expected completed model job ${COMPLETED_MODEL_JOB}." >&2
  exit 1
}
[[ "$(job_state "${COMPLETED_DATA_JOB}")" == COMPLETED* ]] || {
  echo "Expected completed data job ${COMPLETED_DATA_JOB}." >&2
  exit 1
}
[[ "$(job_state "${FAILED_SMOKE_JOB}")" == FAILED* ]] || {
  echo "Expected failed smoke job ${FAILED_SMOKE_JOB}." >&2
  exit 1
}

blocked_record="$(scontrol show job -o "${BLOCKED_SCORE_JOB}")"
[[ "${blocked_record}" == *"JobName=q25-7b-score "* ]] || {
  echo "Unexpected blocked score identity: ${blocked_record}" >&2
  exit 1
}
[[ "${blocked_record}" == *"JobState=PENDING "* ]] || {
  echo "Expected score job ${BLOCKED_SCORE_JOB} to remain pending." >&2
  exit 1
}
[[ "${blocked_record}" == *"Dependency=afterok:${FAILED_SMOKE_JOB}(failed)"* ]] || {
  echo "Blocked score dependency changed; refusing repair." >&2
  exit 1
}
first_train_record="$(scontrol show job -o "${FIRST_TRAIN_JOB}")"
[[ "${first_train_record}" == *"Dependency=afterok:${BLOCKED_SCORE_JOB}(unfulfilled)"* ]] || {
  echo "First training dependency changed; refusing repair." >&2
  exit 1
}
tail_record="$(scontrol show job -o "${TAIL_TRAIN_JOB}")"
[[ "${tail_record}" == *"Dependency=afterok:198107(unfulfilled)"* ]] || {
  echo "Tail dependency changed; refusing repair." >&2
  exit 1
}

[[ -f "${FAILED_LOG}" ]] || {
  echo "Missing failed smoke log: ${FAILED_LOG}" >&2
  exit 1
}
grep -Fq "SFT chat template kwargs: {}}" "${FAILED_LOG}" || {
  echo "Smoke log does not contain the approved shell-expansion failure." >&2
  exit 1
}
grep -Fq "invalid loads value: '{}}'" "${FAILED_LOG}" || {
  echo "Smoke log does not contain the approved JSON parser failure." >&2
  exit 1
}

export PRIOR_REPAIR_MANIFEST
python3 - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["PRIOR_REPAIR_MANIFEST"]).read_text())
if manifest["status"] != "submitted_and_rewired":
    raise RuntimeError("conversion TMPDIR repair is not finalized")
if manifest["replacement_model_preparation"] != 199182:
    raise RuntimeError("completed replacement model job drift")
if manifest["rewired_dependency"] != {
    "job_id": 198095,
    "dependency": "afterok:199182",
}:
    raise RuntimeError("prior repaired dependency drift")
PY

[[ -f "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]] || {
  echo "Completed base torch-dist checkpoint pointer is missing." >&2
  exit 1
}
[[ -f "${BASE_MANIFEST}" ]] || {
  echo "Completed base handoff manifest is missing." >&2
  exit 1
}
for path in \
  "${PREP_STATS_JSON}" \
  "${TOKENIZER_INVENTORY_JSON}" \
  "${CORRECT_ONLY_JSONL}" \
  "${UNMASKED_JSONL}"; do
  [[ -f "${path}" ]] || {
    echo "Completed data artifact is missing: ${path}" >&2
    exit 1
  }
done

[[ -d "${FAILED_OUTPUT}" && -f "${PARTIAL_DATA}" ]] || {
  echo "Missing failed smoke output: ${FAILED_OUTPUT}" >&2
  exit 1
}
[[ "$(find "${FAILED_OUTPUT}" -type f | wc -l)" == "1" ]] || {
  echo "Failed smoke output inventory drifted; refusing repair." >&2
  exit 1
}
[[ "$(wc -l <"${PARTIAL_DATA}")" == "200" ]] || {
  echo "Failed smoke subset row count drifted; refusing repair." >&2
  exit 1
}
[[ "$(sha256sum "${PARTIAL_DATA}" | awk '{print $1}')" == "${EXPECTED_PARTIAL_DATA_SHA256}" ]] || {
  echo "Failed smoke subset hash drifted; refusing repair." >&2
  exit 1
}

export FAILED_LOG_SHA256 FAILED_OUTPUT_BYTES FAILED_FILE_COUNT PARTIAL_DATA_SHA256
FAILED_LOG_SHA256="$(sha256sum "${FAILED_LOG}" | awk '{print $1}')"
FAILED_OUTPUT_BYTES="$(du -sb "${FAILED_OUTPUT}" | awk '{print $1}')"
FAILED_FILE_COUNT="$(find "${FAILED_OUTPUT}" -type f | wc -l)"
PARTIAL_DATA_SHA256="$(sha256sum "${PARTIAL_DATA}" | awk '{print $1}')"
export REPAIR_COMMIT
REPAIR_COMMIT="$(git rev-parse HEAD)"
mkdir -p "${MANIFEST_ROOT}" "$(dirname -- "${FAILED_ARCHIVE}")"

export REPAIR_PENDING REPAIR_FAILED FAILED_ARCHIVE FAILED_LOG FAILED_OUTPUT PARTIAL_DATA
export COMPLETED_MODEL_JOB COMPLETED_DATA_JOB FAILED_SMOKE_JOB
export BLOCKED_SCORE_JOB FIRST_TRAIN_JOB TAIL_TRAIN_JOB
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["REPAIR_PENDING"])
value = {
    "artifact_schema_version": 1,
    "status": "archiving_failed_smoke",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "commit": os.environ["REPAIR_COMMIT"],
    "contract_hash": os.environ["CONTRACT_HASH"],
    "reason": "shell parameter expansion changed explicit {} into invalid {}}",
    "jobs": {
        "completed_model_preparation": int(os.environ["COMPLETED_MODEL_JOB"]),
        "completed_data_preparation": int(os.environ["COMPLETED_DATA_JOB"]),
        "failed_smoke": int(os.environ["FAILED_SMOKE_JOB"]),
        "blocked_score_to_rewire": int(os.environ["BLOCKED_SCORE_JOB"]),
        "preserved_downstream_first": int(os.environ["FIRST_TRAIN_JOB"]),
        "preserved_downstream_last": int(os.environ["TAIL_TRAIN_JOB"]),
    },
    "failed_log": {
        "path": os.environ["FAILED_LOG"],
        "sha256": os.environ["FAILED_LOG_SHA256"],
    },
    "failed_smoke": {
        "original": os.environ["FAILED_OUTPUT"],
        "archive": os.environ["FAILED_ARCHIVE"],
        "file_count": int(os.environ["FAILED_FILE_COUNT"]),
        "bytes": int(os.environ["FAILED_OUTPUT_BYTES"]),
        "partial_data_original": os.environ["PARTIAL_DATA"],
        "partial_data_archive": str(
            Path(os.environ["FAILED_ARCHIVE"]) / "correct_only_first_200.jsonl"
        ),
        "partial_data_sha256": os.environ["PARTIAL_DATA_SHA256"],
        "optimizer_steps": 0,
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

mv -- "${FAILED_OUTPUT}" "${FAILED_ARCHIVE}"
[[ "$(sha256sum "${FAILED_ARCHIVE}/correct_only_first_200.jsonl" | awk '{print $1}')" == "${PARTIAL_DATA_SHA256}" ]] || {
  echo "Archived smoke subset hash mismatch." >&2
  exit 1
}

raw="$(sbatch \
  --parsable \
  --chdir="${SLIME_REPO_ROOT}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name=q25-7b-smoke-r1 \
  "${SCRIPT_DIR}/02_smoke.sbatch")"
replacement_job="${raw%%;*}"
[[ "${replacement_job}" =~ ^[0-9]+$ ]] || {
  echo "Unexpected sbatch response: ${raw}" >&2
  exit 1
}

scontrol update \
  JobId="${BLOCKED_SCORE_JOB}" \
  Dependency="afterok:${replacement_job}"
dependency_rewired=1

rewired_record="$(scontrol show job -o "${BLOCKED_SCORE_JOB}")"
[[ "${rewired_record}" == *"JobState=PENDING "* ]] || {
  echo "Rewired score job is not pending: ${rewired_record}" >&2
  exit 1
}
[[ "${rewired_record}" == *"Dependency=afterok:${replacement_job}(unfulfilled)"* ]] || {
  echo "Score job was not rewired to ${replacement_job}: ${rewired_record}" >&2
  exit 1
}
first_train_record="$(scontrol show job -o "${FIRST_TRAIN_JOB}")"
[[ "${first_train_record}" == *"Dependency=afterok:${BLOCKED_SCORE_JOB}(unfulfilled)"* ]] || {
  echo "First training dependency changed during repair." >&2
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
        "replacement_smoke": int(os.environ["REPLACEMENT_JOB"]),
        "rewired_dependency": {
            "job_id": int(os.environ["BLOCKED_SCORE_JOB"]),
            "dependency": f"afterok:{os.environ['REPLACEMENT_JOB']}",
        },
        "reused_jobs": list(range(198097, 198109)),
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
echo "SMOKE_JSON_REPAIR_COMPLETE failed=${FAILED_SMOKE_JOB} replacement=${replacement_job} rewired=${BLOCKED_SCORE_JOB} reused=198097-198108"
