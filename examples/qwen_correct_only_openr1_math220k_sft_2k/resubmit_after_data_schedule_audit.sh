#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

readonly FAILED_DATA_JOB=212331
readonly BLOCKED_CANARY_JOB=212332
readonly NEXT_CANARY_JOB=212333
readonly TAIL_FINALIZE_JOB=212342
readonly ORIGINAL_MANIFEST="${MANIFEST_ROOT}/training_chain.json"
readonly FAILED_LOG="${SLURM_LOG_DIR}/qco-data_${FAILED_DATA_JOB}.log"
readonly REPAIR_MANIFEST="${MANIFEST_ROOT}/data_schedule_repair.json"
readonly REPAIR_PENDING="${MANIFEST_ROOT}/data_schedule_repair.pending.json"
readonly REPAIR_FAILED="${MANIFEST_ROOT}/data_schedule_repair.failed.json"

dry_run=0
if [[ "${1:-}" == "--dry-run" ]]; then
  dry_run=1
  shift
fi
[[ $# -eq 0 ]] || {
  echo "Usage: $0 [--dry-run]" >&2
  exit 2
}

validate_args=(--example-dir "${SCRIPT_DIR}")
if [[ "${dry_run}" == "0" ]]; then
  validate_args+=(--require-pushed-clean)
fi
python3 "${SCRIPT_DIR}/validate_config.py" "${validate_args[@]}"

for path in "${REPAIR_MANIFEST}" "${REPAIR_PENDING}" "${REPAIR_FAILED}"; do
  [[ ! -e "${path}" ]] || {
    echo "Refusing to overwrite prior repair state: ${path}" >&2
    exit 1
  }
done

job_state() {
  sacct -X -n -P --jobs="$1" --format=State | sed -n '1{s/|.*//;p}'
}

[[ "$(job_state "${FAILED_DATA_JOB}")" == FAILED* ]] || {
  echo "Expected failed data job ${FAILED_DATA_JOB}." >&2
  exit 1
}
[[ -f "${FAILED_LOG}" ]] || {
  echo "Missing failed data log ${FAILED_LOG}." >&2
  exit 1
}
grep -Fq "CORRECT_ONLY_PREP_COMPLETE" "${FAILED_LOG}" || {
  echo "Failed job did not finish data preparation." >&2
  exit 1
}
grep -Fq "ValueError: per-rank optimizer update count drift" "${FAILED_LOG}" || {
  echo "Failed job does not contain the approved schedule-audit failure." >&2
  exit 1
}

blocked_record="$(scontrol show job -o "${BLOCKED_CANARY_JOB}")"
[[ "${blocked_record}" == *"JobName=qco-canary-qwen2_5_3b_8k "* ]] || {
  echo "Unexpected blocked canary identity: ${blocked_record}" >&2
  exit 1
}
[[ "${blocked_record}" == *"JobState=PENDING "* ]] || {
  echo "Expected blocked canary ${BLOCKED_CANARY_JOB} to remain pending." >&2
  exit 1
}
[[ "${blocked_record}" == *"Dependency=afterok:${FAILED_DATA_JOB}(failed)"* ]] || {
  echo "Blocked canary dependency changed; refusing repair." >&2
  exit 1
}
next_record="$(scontrol show job -o "${NEXT_CANARY_JOB}")"
[[ "${next_record}" == *"Dependency=afterok:${BLOCKED_CANARY_JOB}(unfulfilled)"* ]] || {
  echo "Next canary dependency changed; refusing repair." >&2
  exit 1
}
tail_record="$(scontrol show job -o "${TAIL_FINALIZE_JOB}")"
[[ "${tail_record}" == *"Dependency=afterok:212341(unfulfilled)"* ]] || {
  echo "Tail dependency changed; refusing repair." >&2
  exit 1
}

export ORIGINAL_MANIFEST PREP_STATS_JSON DATA_ROOT
python3 - <<'PY'
import hashlib
import json
import os
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


manifest = json.loads(Path(os.environ["ORIGINAL_MANIFEST"]).read_text())
expected_names = [
    "prepare_model_qwen2_5_3b",
    "prepare_model_qwen2_5_7b",
    "prepare_model_qwen3_4b",
    "prepare_model_qwen3_8b",
    "prepare_shared_data",
    "canary_qwen2_5_3b_8k",
    "canary_qwen2_5_3b_16k",
    "canary_qwen2_5_7b_8k",
    "canary_qwen3_4b_8k",
    "canary_qwen3_8b_8k",
    "train_qwen2_5_3b_8k",
    "train_qwen2_5_3b_16k",
    "train_qwen2_5_7b_8k",
    "train_qwen3_4b_8k",
    "train_qwen3_8b_8k",
    "finalize_handoff",
]
jobs = manifest["jobs_in_dependency_order"]
if [int(item["job_id"]) for item in jobs] != list(range(212327, 212343)):
    raise RuntimeError("original job inventory drift")
if [item["name"] for item in jobs] != expected_names:
    raise RuntimeError("original job-name inventory drift")
if manifest["contract_hash"] != "3568736a3743c319":
    raise RuntimeError("original contract hash drift")
if manifest["commit"] != "064e6be6c615764c5b4f0271de88c15918dab9ec":
    raise RuntimeError("original submission commit drift")

stats_path = Path(os.environ["PREP_STATS_JSON"])
stats = json.loads(stats_path.read_text())
if stats["contract"]["rows_per_selection"] != 2000:
    raise RuntimeError("prepared row-count contract drift")
if stats["invariants"]["same_ordered_2k_across_all_8k_models"] is not True:
    raise RuntimeError("shared 8K trace invariant drift")
for selection in stats["selection"].values():
    path = Path(selection["path"])
    if sha256(path) != selection["sha256"]:
        raise RuntimeError(f"selected trace artifact hash drift: {path}")
for run in stats["runs"].values():
    for path_key, hash_key in (
        ("dataset", "dataset_sha256"),
        ("messages", "messages_sha256"),
    ):
        path = Path(run[path_key])
        if sha256(path) != run[hash_key]:
            raise RuntimeError(f"prepared artifact hash drift: {path}")

schedule_dir = Path(os.environ["DATA_ROOT"]) / "schedule_audits"
existing_audits = sorted(schedule_dir.glob("*.json"))
if existing_audits:
    raise RuntimeError(f"unexpected partial schedule audits: {existing_audits}")
PY

sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name=qco-data-r1 \
  --export=ALL,REUSE_PREPARED_DATA=1 \
  "${SCRIPT_DIR}/01_prepare_data.sbatch"

if [[ "${dry_run}" == "1" ]]; then
  echo "DATA_SCHEDULE_REPAIR_DRY_RUN failed=${FAILED_DATA_JOB} rewire=${BLOCKED_CANARY_JOB} reused=212332-212342"
  exit 0
fi

export FAILED_DATA_JOB BLOCKED_CANARY_JOB NEXT_CANARY_JOB TAIL_FINALIZE_JOB
export FAILED_LOG REPAIR_PENDING REPAIR_FAILED
export REPAIR_COMMIT
REPAIR_COMMIT="$(git rev-parse HEAD)"
python3 - <<'PY'
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


output = Path(os.environ["REPAIR_PENDING"])
stats = Path(os.environ["PREP_STATS_JSON"])
value = {
    "artifact_schema_version": 1,
    "status": "submitting_replacement",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "commit": os.environ["REPAIR_COMMIT"],
    "contract_hash": os.environ["CONTRACT_HASH"],
    "reason": "schedule audit compared per-rank microbatch count with optimizer-update count",
    "jobs": {
        "failed_data": int(os.environ["FAILED_DATA_JOB"]),
        "blocked_canary_to_rewire": int(os.environ["BLOCKED_CANARY_JOB"]),
        "preserved_downstream_first": int(os.environ["NEXT_CANARY_JOB"]),
        "preserved_downstream_last": int(os.environ["TAIL_FINALIZE_JOB"]),
    },
    "failed_log": {
        "path": os.environ["FAILED_LOG"],
        "sha256": sha256(Path(os.environ["FAILED_LOG"])),
    },
    "prepared_stats": {
        "path": str(stats),
        "sha256": sha256(stats),
    },
}
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(output)
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

raw="$(sbatch \
  --parsable \
  --chdir="${SLIME_REPO_ROOT}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name=qco-data-r1 \
  --export=ALL,REUSE_PREPARED_DATA=1 \
  "${SCRIPT_DIR}/01_prepare_data.sbatch")"
replacement_job="${raw%%;*}"
[[ "${replacement_job}" =~ ^[0-9]+$ ]] || {
  echo "Unexpected sbatch response: ${raw}" >&2
  exit 1
}

scontrol update \
  JobId="${BLOCKED_CANARY_JOB}" \
  Dependency="afterok:${replacement_job}"
dependency_rewired=1

rewired_record="$(scontrol show job -o "${BLOCKED_CANARY_JOB}")"
[[ "${rewired_record}" == *"JobState=PENDING "* ]] || {
  echo "Rewired canary is not pending: ${rewired_record}" >&2
  exit 1
}
[[ "${rewired_record}" == *"Dependency=afterok:${replacement_job}(unfulfilled)"* ]] || {
  echo "Canary was not rewired to ${replacement_job}: ${rewired_record}" >&2
  exit 1
}
next_record="$(scontrol show job -o "${NEXT_CANARY_JOB}")"
[[ "${next_record}" == *"Dependency=afterok:${BLOCKED_CANARY_JOB}(unfulfilled)"* ]] || {
  echo "Next canary dependency changed during repair." >&2
  exit 1
}
tail_record="$(scontrol show job -o "${TAIL_FINALIZE_JOB}")"
[[ "${tail_record}" == *"Dependency=afterok:212341(unfulfilled)"* ]] || {
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
        "replacement_data_audit": int(os.environ["REPLACEMENT_JOB"]),
        "rewired_dependency": {
            "job_id": int(os.environ["BLOCKED_CANARY_JOB"]),
            "dependency": f"afterok:{os.environ['REPLACEMENT_JOB']}",
        },
        "reused_jobs": list(range(212332, 212343)),
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
echo "DATA_SCHEDULE_REPAIR_COMPLETE failed=${FAILED_DATA_JOB} replacement=${replacement_job} rewired=${BLOCKED_CANARY_JOB} reused=212332-212342"
