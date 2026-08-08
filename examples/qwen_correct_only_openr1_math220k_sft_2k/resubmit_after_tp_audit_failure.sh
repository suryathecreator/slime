#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

readonly COMPLETED_PREVIOUS_CANARY_JOB=212333
readonly FAILED_CANARY_JOB=212334
readonly BLOCKED_CANARY_JOB=212335
readonly NEXT_CANARY_JOB=212336
readonly TAIL_FINALIZE_JOB=212342
readonly ORIGINAL_MANIFEST="${MANIFEST_ROOT}/training_chain.json"
readonly PRIOR_REPAIR_MANIFEST="${MANIFEST_ROOT}/canary_oom_repair.json"
readonly FAILED_LOG="${SLURM_LOG_DIR}/qco-canary-qwen2_5_7b_8k_${FAILED_CANARY_JOB}.log"
readonly CANARY_INPUT_ROOT="${OUTPUT_ROOT}/canaries/qwen2_5_7b_8k"
readonly AUDIT_ATTEMPT=tp_audit_r1
readonly AUDIT_ROOT="${CANARY_INPUT_ROOT}/attempts/${AUDIT_ATTEMPT}"
readonly REPAIR_MANIFEST="${MANIFEST_ROOT}/tp_audit_repair.json"
readonly REPAIR_PENDING="${MANIFEST_ROOT}/tp_audit_repair.pending.json"
readonly REPAIR_FAILED="${MANIFEST_ROOT}/tp_audit_repair.failed.json"

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
    echo "Refusing to overwrite prior TP-audit repair state: ${path}" >&2
    exit 1
  }
done

job_state() {
  sacct -X -n -P --jobs="$1" --format=State | sed -n '1{s/|.*//;p}'
}

[[ "$(job_state "${COMPLETED_PREVIOUS_CANARY_JOB}")" == COMPLETED* ]] || {
  echo "Expected completed preceding canary ${COMPLETED_PREVIOUS_CANARY_JOB}." >&2
  exit 1
}
[[ "$(job_state "${FAILED_CANARY_JOB}")" == FAILED* ]] || {
  echo "Expected failed 7B canary ${FAILED_CANARY_JOB}." >&2
  exit 1
}
[[ -f "${FAILED_LOG}" ]] || { echo "Missing failed log ${FAILED_LOG}." >&2; exit 1; }
[[ "$(grep -Ec "Job '.*' succeeded" "${FAILED_LOG}")" -eq 2 ]] || {
  echo "Failed canary did not complete exactly two one-step training paths." >&2
  exit 1
}
grep -Fq "ValueError: expected 2 DP dumps" "${FAILED_LOG}" || {
  echo "Failed canary does not contain the approved TP-audit assertion." >&2
  exit 1
}
grep -Fq "found 4" "${FAILED_LOG}" || {
  echo "Failed canary does not contain the realized four-rank dump count." >&2
  exit 1
}

blocked_record="$(scontrol show job -o "${BLOCKED_CANARY_JOB}")"
[[ "${blocked_record}" == *"JobName=qco-canary-qwen3_4b_8k "* ]] || {
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

export ORIGINAL_MANIFEST PRIOR_REPAIR_MANIFEST CANARY_INPUT_ROOT AUDIT_ROOT
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
if [int(item["job_id"]) for item in original["jobs_in_dependency_order"]] != list(
    range(212327, 212343)
):
    raise RuntimeError("original job inventory drift")

prior = json.loads(Path(os.environ["PRIOR_REPAIR_MANIFEST"]).read_text())
if prior["status"] != "submitted_and_rewired":
    raise RuntimeError("prior canary repair is not complete")
if prior["commit"] != "814290d78fb01fc8ed048cb00d326fff1dcb8400":
    raise RuntimeError("prior canary repair commit drift")
if prior["replacement_canary"] != 212380:
    raise RuntimeError("prior replacement canary drift")

root = Path(os.environ["CANARY_INPUT_ROOT"])
required = [
    root / "source/custom_pretokenized.jsonl",
    root / "source/stock_messages.jsonl",
    root / "custom_pretokenized/weights/iter_0000000/config.json",
    root / "custom_pretokenized/weights/iter_0000000/model.safetensors.index.json",
    root / "stock_messages/weights/iter_0000000/config.json",
    root / "stock_messages/weights/iter_0000000/model.safetensors.index.json",
]
for path in required:
    if not path.is_file():
        raise RuntimeError(f"missing completed 7B canary artifact: {path}")
for mode in ("custom_pretokenized", "stock_messages"):
    checkpoint = root / mode / "weights/iter_0000000"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 4 or any(not (checkpoint / shard).is_file() for shard in shards):
        raise RuntimeError(f"incomplete four-shard HF checkpoint for {mode}")
    dumps = sorted((root / mode / "details/train_data").glob("0_*.pt"))
    if len(dumps) != 4:
        raise RuntimeError(f"expected four realized model-rank dumps for {mode}")
for name in (
    "runtime_batch_audit.json",
    "decoded_shifted_tokens.jsonl",
    "generation_before_after.json",
):
    if (root / name).exists():
        raise RuntimeError(f"failed canary unexpectedly contains {name}")
if Path(os.environ["AUDIT_ROOT"]).exists():
    raise RuntimeError(f"refusing to overwrite audit attempt {os.environ['AUDIT_ROOT']}")

print(
    "TP_AUDIT_INPUTS_VALID "
    + " ".join(
        f"{label}_sha256={sha256(path)}"
        for label, path in {
            "custom_source": required[0],
            "stock_source": required[1],
            "custom_index": required[3],
            "stock_index": required[5],
        }.items()
    ),
    flush=True,
)
PY

sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" \
  --account="${ACCOUNT}" \
  --partition="${PARTITION}" \
  --qos="${QOS}" \
  --job-name=qco-canary-7b8k-a1 \
  --export=ALL,RUN_KEY=qwen2_5_7b_8k,CANARY_AUDIT_ATTEMPT="${AUDIT_ATTEMPT}" \
  "${SCRIPT_DIR}/02b_resume_canary_audit.sbatch"

if [[ "${dry_run}" == "1" ]]; then
  echo "TP_AUDIT_REPAIR_DRY_RUN failed=${FAILED_CANARY_JOB} rewire=${BLOCKED_CANARY_JOB} reused=212335-212342"
  exit 0
fi

export COMPLETED_PREVIOUS_CANARY_JOB FAILED_CANARY_JOB BLOCKED_CANARY_JOB
export NEXT_CANARY_JOB TAIL_FINALIZE_JOB FAILED_LOG AUDIT_ROOT
export REPAIR_PENDING REPAIR_FAILED REPAIR_COMMIT
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
root = Path(os.environ["CANARY_INPUT_ROOT"])
reused_artifacts = {
    "custom_source": root / "source/custom_pretokenized.jsonl",
    "stock_source": root / "source/stock_messages.jsonl",
    "custom_checkpoint_index": root / "custom_pretokenized/weights/iter_0000000/model.safetensors.index.json",
    "stock_checkpoint_index": root / "stock_messages/weights/iter_0000000/model.safetensors.index.json",
}
value = {
    "artifact_schema_version": 1,
    "status": "submitting_replacement",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "commit": os.environ["REPAIR_COMMIT"],
    "contract_hash": os.environ["CONTRACT_HASH"],
    "reason": "TP=2 canary audit expected DP dumps instead of all model-rank dumps",
    "retry_scope": "reuse completed custom and stock checkpoints; audit and generate only",
    "audit_attempt": "tp_audit_r1",
    "audit_root": os.environ["AUDIT_ROOT"],
    "jobs": {
        "completed_preceding_canary": int(os.environ["COMPLETED_PREVIOUS_CANARY_JOB"]),
        "failed_canary": int(os.environ["FAILED_CANARY_JOB"]),
        "blocked_canary_to_rewire": int(os.environ["BLOCKED_CANARY_JOB"]),
        "preserved_downstream_first": int(os.environ["NEXT_CANARY_JOB"]),
        "preserved_downstream_last": int(os.environ["TAIL_FINALIZE_JOB"]),
    },
    "failed_log": {
        "path": os.environ["FAILED_LOG"],
        "sha256": sha256(Path(os.environ["FAILED_LOG"])),
    },
    "reused_artifacts": {
        name: {
            "path": str(path),
            "sha256": sha256(path),
        }
        for name, path in reused_artifacts.items()
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
  --job-name=qco-canary-7b8k-a1 \
  --export=ALL,RUN_KEY=qwen2_5_7b_8k,CANARY_AUDIT_ATTEMPT="${AUDIT_ATTEMPT}" \
  "${SCRIPT_DIR}/02b_resume_canary_audit.sbatch")"
replacement_job="${raw%%;*}"
[[ "${replacement_job}" =~ ^[0-9]+$ ]] || {
  echo "Unexpected audit-resume sbatch response: ${raw}" >&2
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
        "replacement_audit_job": int(os.environ["REPLACEMENT_JOB"]),
        "rewired_dependency": {
            "job_id": int(os.environ["BLOCKED_CANARY_JOB"]),
            "dependency": f"afterok:{os.environ['REPLACEMENT_JOB']}",
        },
        "reused_jobs": list(range(212335, 212343)),
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
echo "TP_AUDIT_REPAIR_COMPLETE failed=${FAILED_CANARY_JOB} replacement=${replacement_job} rewired=${BLOCKED_CANARY_JOB} reused=212335-212342"
