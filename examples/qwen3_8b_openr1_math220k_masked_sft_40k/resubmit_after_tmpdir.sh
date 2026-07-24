#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
SCRIPT_2K="${SCRIPT_DIR}/../qwen3_8b_openr1_math220k_masked_sft_2k"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

readonly BASE_EVAL_JOB=185500
readonly FAILED_TRAIN_JOB=185501
readonly BLOCKED_EVAL_JOB=185502
readonly DOWNSTREAM_JOB=185503
readonly REPAIR_STEM="dependency_repair_after_tmpdir"
readonly REPAIR_MANIFEST="${MANIFEST_ROOT}/${REPAIR_STEM}.json"
readonly REPAIR_PENDING="${MANIFEST_ROOT}/${REPAIR_STEM}.pending.json"
readonly REPAIR_FAILED="${MANIFEST_ROOT}/${REPAIR_STEM}.failed.json"
readonly FAILED_OUTPUT="${OUTPUT_ROOT}/training/correct_only"
readonly FAILED_ARCHIVE="${OUTPUT_ROOT}/failed_attempts/correct_only_job_${FAILED_TRAIN_JOB}"
readonly FAILED_LOG="${SLURM_LOG_DIR}/q8b-40k-train-correct_${FAILED_TRAIN_JOB}.log"
readonly PARTIAL_COMMON="${FAILED_OUTPUT}/full_state/iter_0000049/common.pt"

[[ "$(git branch --show-current)" == "qwen3-8b-openr1-masked-sft-40k" ]] || {
  echo "Wrong branch." >&2
  exit 1
}
[[ -z "$(git status --porcelain)" ]] || {
  echo "Refusing recovery from a dirty worktree." >&2
  exit 1
}
upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
[[ -n "${upstream}" ]] || {
  echo "Push the branch and set its upstream before recovery." >&2
  exit 1
}
[[ "$(git rev-parse HEAD)" == "$(git rev-parse "${upstream}")" ]] || {
  echo "Local HEAD is not the pushed upstream commit." >&2
  exit 1
}
export REPAIR_COMMIT
REPAIR_COMMIT="$(git rev-parse HEAD)"

python3 "${SCRIPT_DIR}/validate_config.py" \
  --example-dir "${SCRIPT_DIR}" --require-pushed-clean
python3 "${SCRIPT_2K}/validate_config.py" \
  --example-dir "${SCRIPT_2K}" --require-pushed-clean

for path in "${REPAIR_MANIFEST}" "${REPAIR_PENDING}" "${REPAIR_FAILED}" "${FAILED_ARCHIVE}"; do
  [[ ! -e "${path}" ]] || {
    echo "Refusing to overwrite prior recovery state: ${path}" >&2
    exit 1
  }
done

job_state() {
  sacct -X -n -P --jobs="$1" --format=State | sed -n '1{s/|.*//;p}'
}
[[ "$(job_state "${BASE_EVAL_JOB}")" == COMPLETED* ]] || {
  echo "Expected completed base eval ${BASE_EVAL_JOB}." >&2
  exit 1
}
[[ "$(job_state "${FAILED_TRAIN_JOB}")" == FAILED* ]] || {
  echo "Expected failed training job ${FAILED_TRAIN_JOB}." >&2
  exit 1
}

blocked_record="$(scontrol show job -o "${BLOCKED_EVAL_JOB}")"
[[ "${blocked_record}" == *"JobName=q8b-40k-eval-correct "* ]] || {
  echo "Unexpected identity for blocked eval ${BLOCKED_EVAL_JOB}." >&2
  exit 1
}
[[ "${blocked_record}" == *"JobState=PENDING "* ]] || {
  echo "Expected blocked eval ${BLOCKED_EVAL_JOB} to be pending." >&2
  exit 1
}
[[ "${blocked_record}" == *"Dependency=afterok:${FAILED_TRAIN_JOB}(failed)"* ]] || {
  echo "Blocked eval dependency changed; refusing repair." >&2
  exit 1
}

downstream_record="$(scontrol show job -o "${DOWNSTREAM_JOB}")"
[[ "${downstream_record}" == *"JobName=q8b-40k-train-weight "* ]] || {
  echo "Unexpected identity for downstream job ${DOWNSTREAM_JOB}." >&2
  exit 1
}
[[ "${downstream_record}" == *"JobState=PENDING "* ]] || {
  echo "Expected downstream job ${DOWNSTREAM_JOB} to be pending." >&2
  exit 1
}
[[ "${downstream_record}" == *"Dependency=afterok:${BLOCKED_EVAL_JOB}(unfulfilled)"* ]] || {
  echo "Downstream dependency changed; refusing repair." >&2
  exit 1
}

[[ -f "${FAILED_LOG}" ]] || {
  echo "Missing failed training log: ${FAILED_LOG}" >&2
  exit 1
}
[[ -f "${PARTIAL_COMMON}" ]] || {
  echo "Missing expected partial checkpoint marker: ${PARTIAL_COMMON}" >&2
  exit 1
}
[[ ! -e "${FAILED_OUTPUT}/full_state/latest_checkpointed_iteration.txt" ]] || {
  echo "Failed output unexpectedly contains a resumable checkpoint pointer." >&2
  exit 1
}
if find "${FAILED_OUTPUT}/weights" -mindepth 1 -print -quit | grep -q .; then
  echo "Failed output unexpectedly contains an HF snapshot." >&2
  exit 1
fi

export FAILED_LOG_SHA256 PARTIAL_COMMON_SHA256 FAILED_FILE_COUNT FAILED_BYTES
FAILED_LOG_SHA256="$(sha256sum "${FAILED_LOG}" | awk '{print $1}')"
PARTIAL_COMMON_SHA256="$(sha256sum "${PARTIAL_COMMON}" | awk '{print $1}')"
FAILED_FILE_COUNT="$(find "${FAILED_OUTPUT}" -type f | wc -l)"
FAILED_BYTES="$(du -sb "${FAILED_OUTPUT}" | awk '{print $1}')"
mkdir -p "${MANIFEST_ROOT}" "$(dirname -- "${FAILED_ARCHIVE}")"

export REPAIR_PENDING FAILED_ARCHIVE FAILED_LOG PARTIAL_COMMON
export REPAIR_FAILED
export BASE_EVAL_JOB FAILED_TRAIN_JOB BLOCKED_EVAL_JOB DOWNSTREAM_JOB
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["REPAIR_PENDING"])
value = {
    "status": "archiving_failed_attempt",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "commit": os.environ["REPAIR_COMMIT"],
    "reason": "multiprocessing.Manager AF_UNIX path too long during async torch_dist checkpoint creation",
    "jobs": {
        "reused_base_eval": int(os.environ["BASE_EVAL_JOB"]),
        "failed_train": int(os.environ["FAILED_TRAIN_JOB"]),
        "blocked_eval_to_rewire": int(os.environ["BLOCKED_EVAL_JOB"]),
        "preserved_downstream_anchor": int(os.environ["DOWNSTREAM_JOB"]),
    },
    "failed_attempt": {
        "archive": os.environ["FAILED_ARCHIVE"],
        "file_count": int(os.environ["FAILED_FILE_COUNT"]),
        "bytes": int(os.environ["FAILED_BYTES"]),
        "log": os.environ["FAILED_LOG"],
        "log_sha256": os.environ["FAILED_LOG_SHA256"],
        "partial_common_original": os.environ["PARTIAL_COMMON"],
        "partial_common_archive": str(
            Path(os.environ["FAILED_ARCHIVE"])
            / "full_state/iter_0000049/common.pt"
        ),
        "partial_common_sha256": os.environ["PARTIAL_COMMON_SHA256"],
        "resumable": False,
    },
}
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(path)
PY

mv -- "${FAILED_OUTPUT}" "${FAILED_ARCHIVE}"
[[ "$(sha256sum "${FAILED_ARCHIVE}/full_state/iter_0000049/common.pt" | awk '{print $1}')" == "${PARTIAL_COMMON_SHA256}" ]] || {
  echo "Archived partial checkpoint hash mismatch." >&2
  exit 1
}

replacement_job=""
dependency_rewired=0
submission_complete=0
record_failed_recovery() {
  local status="$1"
  export RECOVERY_SHELL_STATUS="${status}" REPLACEMENT_JOB="${replacement_job}"
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
        "status": "recovery_failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "shell_status": int(os.environ["RECOVERY_SHELL_STATUS"]),
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
  if [[ "${submission_complete}" != "1" ]]; then
    if [[ -n "${replacement_job}" && "${dependency_rewired}" != "1" ]]; then
      scancel "${replacement_job}" || true
    fi
    record_failed_recovery "${status}" || true
  fi
  exit "${status}"
}
trap fail_closed EXIT

raw="$(sbatch \
  --parsable \
  --chdir="${SLIME_REPO_ROOT}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name=q8b-40k-train-correct-r1 \
  --export=ALL,SFT_VARIANT=correct_only \
  "${SCRIPT_DIR}/03_train.sbatch")"
replacement_job="${raw%%;*}"
[[ "${replacement_job}" =~ ^[0-9]+$ ]] || {
  echo "Unexpected sbatch response: ${raw}" >&2
  exit 1
}

# Reuse the existing eval and entire downstream chain. Updating this edge is
# sufficient because job 185503 already depends on 185502.
scontrol update \
  JobId="${BLOCKED_EVAL_JOB}" \
  Dependency="afterok:${replacement_job}"
dependency_rewired=1

rewired_record="$(scontrol show job -o "${BLOCKED_EVAL_JOB}")"
[[ "${rewired_record}" == *"JobState=PENDING "* ]] || {
  echo "Rewired eval is not pending: ${rewired_record}" >&2
  exit 1
}
[[ "${rewired_record}" == *"Dependency=afterok:${replacement_job}(unfulfilled)"* ]] || {
  echo "Rewired eval does not depend on replacement ${replacement_job}." >&2
  exit 1
}
downstream_record="$(scontrol show job -o "${DOWNSTREAM_JOB}")"
[[ "${downstream_record}" == *"Dependency=afterok:${BLOCKED_EVAL_JOB}(unfulfilled)"* ]] || {
  echo "Downstream anchor changed during repair." >&2
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
path = Path(os.environ["REPAIR_MANIFEST"])
value = json.loads(pending.read_text())
value.update(
    {
        "status": "submitted_and_rewired",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "replacement_train": int(os.environ["REPLACEMENT_JOB"]),
        "dependency_update": {
            "job": int(os.environ["BLOCKED_EVAL_JOB"]),
            "old": f"afterok:{os.environ['FAILED_TRAIN_JOB']}",
            "new": f"afterok:{os.environ['REPLACEMENT_JOB']}",
        },
    }
)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(path)
pending.unlink()
print(json.dumps(value, indent=2, sort_keys=True))
PY

submission_complete=1
trap - EXIT
echo "RECOVERY_SUBMITTED replacement=${replacement_job} rewired_eval=${BLOCKED_EVAL_JOB}"
