#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

readonly FAILED_CANARY_JOB=212332
readonly BLOCKED_CANARY_JOB=212333
readonly NEXT_CANARY_JOB=212334
readonly TAIL_FINALIZE_JOB=212342
readonly DATA_REPAIR_JOB=212355
readonly ORIGINAL_MANIFEST="${MANIFEST_ROOT}/training_chain.json"
readonly DATA_REPAIR_MANIFEST="${MANIFEST_ROOT}/data_schedule_repair.json"
readonly FAILED_LOG="${SLURM_LOG_DIR}/qco-canary-qwen2_5_3b_8k_${FAILED_CANARY_JOB}.log"
readonly REPAIR_MANIFEST="${MANIFEST_ROOT}/canary_oom_repair.json"
readonly REPAIR_PENDING="${MANIFEST_ROOT}/canary_oom_repair.pending.json"
readonly REPAIR_FAILED="${MANIFEST_ROOT}/canary_oom_repair.failed.json"
readonly RETRY_ROOT="${OUTPUT_ROOT}/canaries/qwen2_5_3b_8k/attempts/oom_r1"

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
    echo "Refusing to overwrite prior OOM repair state: ${path}" >&2
    exit 1
  }
done

job_state() {
  sacct -X -n -P --jobs="$1" --format=State | sed -n '1{s/|.*//;p}'
}

[[ "$(job_state "${FAILED_CANARY_JOB}")" == FAILED* ]] || {
  echo "Expected failed canary job ${FAILED_CANARY_JOB}." >&2
  exit 1
}
[[ "$(job_state "${DATA_REPAIR_JOB}")" == COMPLETED* ]] || {
  echo "Expected completed data repair job ${DATA_REPAIR_JOB}." >&2
  exit 1
}
[[ -f "${FAILED_LOG}" ]] || {
  echo "Missing failed canary log ${FAILED_LOG}." >&2
  exit 1
}
grep -Fq "SFT max tokens per GPU: 32768" "${FAILED_LOG}" || {
  echo "Failed canary did not use the diagnosed 32768-token cap." >&2
  exit 1
}
grep -Fq "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 684.00 MiB" "${FAILED_LOG}" || {
  echo "Failed canary does not contain the approved OOM signature." >&2
  exit 1
}

blocked_record="$(scontrol show job -o "${BLOCKED_CANARY_JOB}")"
[[ "${blocked_record}" == *"JobName=qco-canary-qwen2_5_3b_16k "* ]] || {
  echo "Unexpected blocked canary identity: ${blocked_record}" >&2
  exit 1
}
[[ "${blocked_record}" == *"JobState=PENDING "* ]] || {
  echo "Expected blocked canary ${BLOCKED_CANARY_JOB} to remain pending." >&2
  exit 1
}
[[ "${blocked_record}" == *"Dependency=afterok:${FAILED_CANARY_JOB}(failed)"* ]] || {
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

export ORIGINAL_MANIFEST DATA_REPAIR_MANIFEST PREP_STATS_JSON SCHEDULE_AUDIT_DIR
export RETRY_ROOT
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


original = json.loads(Path(os.environ["ORIGINAL_MANIFEST"]).read_text())
if original["contract_hash"] != "3568736a3743c319":
    raise RuntimeError("original contract hash drift")
if original["commit"] != "064e6be6c615764c5b4f0271de88c15918dab9ec":
    raise RuntimeError("original submission commit drift")
jobs = original["jobs_in_dependency_order"]
if [int(item["job_id"]) for item in jobs] != list(range(212327, 212343)):
    raise RuntimeError("original job inventory drift")

data_repair = json.loads(Path(os.environ["DATA_REPAIR_MANIFEST"]).read_text())
if data_repair["status"] != "submitted_and_rewired":
    raise RuntimeError("data repair is not complete")
if data_repair["contract_hash"] != "3568736a3743c319":
    raise RuntimeError("data repair contract hash drift")
if data_repair["commit"] != "ee8ceee8115aad816fd5f868cb43a1c9d6cdaf8b":
    raise RuntimeError("data repair commit drift")
if data_repair["replacement_data_audit"] != 212355:
    raise RuntimeError("data repair replacement job drift")
if data_repair["rewired_dependency"] != {
    "dependency": "afterok:212355",
    "job_id": 212332,
}:
    raise RuntimeError("data repair dependency record drift")

stats_path = Path(os.environ["PREP_STATS_JSON"])
stats = json.loads(stats_path.read_text())
if sha256(stats_path) != data_repair["prepared_stats"]["sha256"]:
    raise RuntimeError("prepared stats hash drift")
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

retry_root = Path(os.environ["RETRY_ROOT"])
if retry_root.exists():
    raise RuntimeError(f"refusing to overwrite canary retry: {retry_root}")
schedule_dir = Path(os.environ["SCHEDULE_AUDIT_DIR"])
if schedule_dir.exists():
    raise RuntimeError(f"refusing to overwrite memory-profile audits: {schedule_dir}")
PY

resolve_run qwen2_5_3b_8k
[[ "${SFT_TENSOR_MODEL_PARALLEL_SIZE}:${SFT_MAX_TOKENS_PER_GPU}:${SFT_OPTIMIZER_CPU_OFFLOAD}" == "1:16384:0" ]] || {
  echo "Qwen2.5-3B 8K memory profile drift." >&2
  exit 1
}

sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name=qco-schedule-r2 \
  --export=ALL,REUSE_PREPARED_DATA=1 \
  "${SCRIPT_DIR}/01_prepare_data.sbatch"
sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name=qco-canary-3b8k-r1 \
  --export=ALL,RUN_KEY=qwen2_5_3b_8k,CANARY_ATTEMPT=oom_r1 \
  "${SCRIPT_DIR}/02_canary.sbatch"

if [[ "${dry_run}" == "1" ]]; then
  echo "CANARY_OOM_REPAIR_DRY_RUN failed=${FAILED_CANARY_JOB} rewire=${BLOCKED_CANARY_JOB} reused=212333-212342"
  exit 0
fi

export FAILED_CANARY_JOB BLOCKED_CANARY_JOB NEXT_CANARY_JOB TAIL_FINALIZE_JOB
export DATA_REPAIR_JOB FAILED_LOG REPAIR_PENDING REPAIR_FAILED RETRY_ROOT
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
value = {
    "artifact_schema_version": 1,
    "status": "submitting_replacements",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "commit": os.environ["REPAIR_COMMIT"],
    "contract_hash": os.environ["CONTRACT_HASH"],
    "reason": "first Qwen2.5-3B 8K canary microbatch exhausted H200 memory at a 32768-token cap",
    "memory_profile": {
        "name": "memory_r1",
        "failed": "tp1:max_tokens32768:optimizer_cpu_offload0",
        "replacement": "tp1:max_tokens16384:optimizer_cpu_offload0",
    },
    "jobs": {
        "failed_canary": int(os.environ["FAILED_CANARY_JOB"]),
        "completed_data_repair": int(os.environ["DATA_REPAIR_JOB"]),
        "blocked_canary_to_rewire": int(os.environ["BLOCKED_CANARY_JOB"]),
        "preserved_downstream_first": int(os.environ["NEXT_CANARY_JOB"]),
        "preserved_downstream_last": int(os.environ["TAIL_FINALIZE_JOB"]),
    },
    "failed_log": {
        "path": os.environ["FAILED_LOG"],
        "sha256": sha256(Path(os.environ["FAILED_LOG"])),
    },
    "retry_root": os.environ["RETRY_ROOT"],
}
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(output)
PY

reaudit_job=""
replacement_canary_job=""
dependency_rewired=0
repair_complete=0
record_failed_repair() {
  local shell_status="$1"
  export REPAIR_SHELL_STATUS="${shell_status}" REAUDIT_JOB="${reaudit_job}"
  export REPLACEMENT_CANARY_JOB="${replacement_canary_job}"
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
        "replacement_schedule_audit": (
            int(os.environ["REAUDIT_JOB"]) if os.environ.get("REAUDIT_JOB") else None
        ),
        "replacement_canary": (
            int(os.environ["REPLACEMENT_CANARY_JOB"])
            if os.environ.get("REPLACEMENT_CANARY_JOB")
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
    if [[ "${dependency_rewired}" != "1" ]]; then
      if [[ -n "${replacement_canary_job}" ]]; then
        scancel "${replacement_canary_job}" || true
      fi
      if [[ -n "${reaudit_job}" ]]; then
        scancel "${reaudit_job}" || true
      fi
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
  --job-name=qco-schedule-r2 \
  --export=ALL,REUSE_PREPARED_DATA=1 \
  "${SCRIPT_DIR}/01_prepare_data.sbatch")"
reaudit_job="${raw%%;*}"
[[ "${reaudit_job}" =~ ^[0-9]+$ ]] || {
  echo "Unexpected schedule-audit sbatch response: ${raw}" >&2
  exit 1
}

raw="$(sbatch \
  --parsable \
  --chdir="${SLIME_REPO_ROOT}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --dependency="afterok:${reaudit_job}" \
  --job-name=qco-canary-3b8k-r1 \
  --export=ALL,RUN_KEY=qwen2_5_3b_8k,CANARY_ATTEMPT=oom_r1 \
  "${SCRIPT_DIR}/02_canary.sbatch")"
replacement_canary_job="${raw%%;*}"
[[ "${replacement_canary_job}" =~ ^[0-9]+$ ]] || {
  echo "Unexpected canary sbatch response: ${raw}" >&2
  exit 1
}

scontrol update \
  JobId="${BLOCKED_CANARY_JOB}" \
  Dependency="afterok:${replacement_canary_job}"
dependency_rewired=1

replacement_record="$(scontrol show job -o "${replacement_canary_job}")"
[[ "${replacement_record}" == *"JobState=PENDING "* ]] || {
  echo "Replacement canary is not pending: ${replacement_record}" >&2
  exit 1
}
[[ "${replacement_record}" == *"Dependency=afterok:${reaudit_job}(unfulfilled)"* ]] || {
  echo "Replacement canary was not linked to schedule audit ${reaudit_job}." >&2
  exit 1
}
rewired_record="$(scontrol show job -o "${BLOCKED_CANARY_JOB}")"
[[ "${rewired_record}" == *"JobState=PENDING "* ]] || {
  echo "Rewired canary is not pending: ${rewired_record}" >&2
  exit 1
}
[[ "${rewired_record}" == *"Dependency=afterok:${replacement_canary_job}(unfulfilled)"* ]] || {
  echo "Canary was not rewired to ${replacement_canary_job}: ${rewired_record}" >&2
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

export REAUDIT_JOB="${reaudit_job}" REPLACEMENT_CANARY_JOB="${replacement_canary_job}"
export REPAIR_MANIFEST
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
        "replacement_schedule_audit": int(os.environ["REAUDIT_JOB"]),
        "replacement_canary": int(os.environ["REPLACEMENT_CANARY_JOB"]),
        "replacement_canary_dependency": f"afterok:{os.environ['REAUDIT_JOB']}",
        "rewired_dependency": {
            "job_id": int(os.environ["BLOCKED_CANARY_JOB"]),
            "dependency": f"afterok:{os.environ['REPLACEMENT_CANARY_JOB']}",
        },
        "reused_jobs": list(range(212333, 212343)),
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
echo "CANARY_OOM_REPAIR_COMPLETE failed=${FAILED_CANARY_JOB} reaudit=${reaudit_job} replacement=${replacement_canary_job} rewired=${BLOCKED_CANARY_JOB} reused=212333-212342"
