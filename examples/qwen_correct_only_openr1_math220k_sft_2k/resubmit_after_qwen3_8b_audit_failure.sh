#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

readonly COMPLETED_PREVIOUS_CANARY_JOB=212498
readonly FAILED_CANARY_JOB=212336
readonly ORIGINAL_COMMIT=064e6be6c615764c5b4f0271de88c15918dab9ec
readonly ORIGINAL_MANIFEST="${MANIFEST_ROOT}/training_chain.json"
readonly PREVIOUS_REPAIR_MANIFEST="${MANIFEST_ROOT}/qwen3_4b_oom_repair.json"
readonly FAILED_LOG="${SLURM_LOG_DIR}/qco-canary-qwen3_8b_8k_${FAILED_CANARY_JOB}.log"
readonly CANARY_ROOT="${OUTPUT_ROOT}/canaries/qwen3_8b_8k"
readonly AUDIT_ATTEMPT=snapshot_audit_r1
readonly AUDIT_ROOT="${CANARY_ROOT}/attempts/${AUDIT_ATTEMPT}"
readonly REPAIR_MANIFEST="${MANIFEST_ROOT}/qwen3_8b_audit_repair.json"
readonly REPAIR_PENDING="${MANIFEST_ROOT}/qwen3_8b_audit_repair.pending.json"
readonly REPAIR_FAILED="${MANIFEST_ROOT}/qwen3_8b_audit_repair.failed.json"
readonly -a STALE_JOBS=(212337 212338 212339 212340 212341 212342)
readonly -a STALE_NAMES=(
  qco-train-qwen2_5_3b_8k
  qco-train-qwen2_5_3b_16k
  qco-train-qwen2_5_7b_8k
  qco-train-qwen3_4b_8k
  qco-train-qwen3_8b_8k
  qco-finalize
)
readonly -a STALE_PARENTS=(212336 212337 212338 212339 212340 212341)
readonly -a RUN_KEYS=(
  qwen2_5_3b_8k
  qwen2_5_3b_16k
  qwen2_5_7b_8k
  qwen3_4b_8k
  qwen3_8b_8k
)

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
    echo "Refusing to overwrite prior Qwen3-8B audit repair state: ${path}" >&2
    exit 1
  }
done

job_state() {
  sacct -X -n -P --jobs="$1" --format=State | sed -n '1{s/|.*//;p}'
}

[[ "$(job_state "${COMPLETED_PREVIOUS_CANARY_JOB}")" == COMPLETED* ]] || {
  echo "Expected completed Qwen3-4B replacement ${COMPLETED_PREVIOUS_CANARY_JOB}." >&2
  exit 1
}
[[ "$(job_state "${FAILED_CANARY_JOB}")" == FAILED* ]] || {
  echo "Expected failed Qwen3-8B canary ${FAILED_CANARY_JOB}." >&2
  exit 1
}
[[ -f "${FAILED_LOG}" ]] || { echo "Missing failed log ${FAILED_LOG}." >&2; exit 1; }
[[ "$(grep -Ec "Job '.*' succeeded" "${FAILED_LOG}")" -eq 2 ]] || {
  echo "Failed canary did not complete exactly two one-step training paths." >&2
  exit 1
}
grep -Fq "the following arguments are required: --tensor-parallel-size" "${FAILED_LOG}" || {
  echo "Failed canary does not contain the diagnosed audit CLI error." >&2
  exit 1
}

for index in "${!STALE_JOBS[@]}"; do
  record="$(scontrol show job -o "${STALE_JOBS[$index]}")"
  [[ "${record}" == *"JobName=${STALE_NAMES[$index]} "* ]] || {
    echo "Unexpected stale job identity: ${record}" >&2
    exit 1
  }
  [[ "${record}" == *"JobState=PENDING "* ]] || {
    echo "Expected stale job ${STALE_JOBS[$index]} to remain pending." >&2
    exit 1
  }
  [[ "${record}" == *"Dependency=afterok:${STALE_PARENTS[$index]}("* ]] || {
    echo "Stale dependency changed for job ${STALE_JOBS[$index]}." >&2
    exit 1
  }
done

export ORIGINAL_MANIFEST PREVIOUS_REPAIR_MANIFEST CANARY_ROOT AUDIT_ROOT
export FAILED_LOG SCHEDULE_AUDIT_DIR OUTPUT_ROOT HANDOFF_ROOT
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

previous = json.loads(Path(os.environ["PREVIOUS_REPAIR_MANIFEST"]).read_text())
if previous["status"] != "submitted_and_rewired":
    raise RuntimeError("Qwen3-4B repair is not complete")
if previous["commit"] != "a0d49aa0407ed1e88a480494f9f416223571ae9d":
    raise RuntimeError("Qwen3-4B repair commit drift")
if previous["replacement_canary"] != 212498:
    raise RuntimeError("Qwen3-4B replacement job drift")
if previous["rewired_dependency"] != {
    "dependency": "afterok:212498",
    "job_id": 212336,
}:
    raise RuntimeError("Qwen3-4B repair dependency drift")

root = Path(os.environ["CANARY_ROOT"])
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
        raise RuntimeError(f"missing completed Qwen3-8B canary artifact: {path}")
for mode in ("custom_pretokenized", "stock_messages"):
    checkpoint = root / mode / "weights/iter_0000000"
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    shards = sorted(set(index["weight_map"].values()))
    if len(shards) != 5 or any(not (checkpoint / shard).is_file() for shard in shards):
        raise RuntimeError(f"incomplete five-shard Qwen3-8B checkpoint for {mode}")
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

schedule_dir = Path(os.environ["SCHEDULE_AUDIT_DIR"])
run_keys = (
    "qwen2_5_3b_8k",
    "qwen2_5_3b_16k",
    "qwen2_5_7b_8k",
    "qwen3_4b_8k",
    "qwen3_8b_8k",
)
for run_key in run_keys:
    schedule = schedule_dir / f"{run_key}.json"
    if not schedule.is_file():
        raise RuntimeError(f"missing memory_r2 schedule audit: {schedule}")

output_root = Path(os.environ["OUTPUT_ROOT"])
handoff_root = Path(os.environ["HANDOFF_ROOT"])
for run_key in run_keys:
    training = output_root / "training" / run_key
    handoff = handoff_root / "checkpoints" / f"{run_key}.json"
    if training.exists() or handoff.exists():
        raise RuntimeError(f"refusing to collide with full-training output for {run_key}")

print(
    "QWEN3_8B_AUDIT_INPUTS_VALID "
    + " ".join(
        f"{label}_sha256={sha256(path)}"
        for label, path in {
            "failed_log": Path(os.environ["FAILED_LOG"]),
            "custom_source": required[0],
            "stock_source": required[1],
            "custom_index": required[3],
            "stock_index": required[5],
        }.items()
    ),
    flush=True,
)
PY

resolve_run qwen3_8b_8k
[[ "${SFT_MEMORY_PROFILE}" == "memory_r2" ]] || {
  echo "Unexpected memory profile ${SFT_MEMORY_PROFILE}." >&2
  exit 1
}
[[ "${SFT_TENSOR_MODEL_PARALLEL_SIZE}:${SFT_MAX_TOKENS_PER_GPU}:${SFT_OPTIMIZER_CPU_OFFLOAD}" == "2:16384:1" ]] || {
  echo "Qwen3-8B runtime profile drift." >&2
  exit 1
}

sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" \
  --partition="${PARTITION}" --qos="${QOS}" \
  --job-name=qco-canary-8b8k-a1 \
  --export=ALL,RUN_KEY=qwen3_8b_8k,CANARY_AUDIT_ATTEMPT="${AUDIT_ATTEMPT}" \
  "${SCRIPT_DIR}/02b_resume_canary_audit.sbatch"
sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" \
  --partition="${PARTITION}" --qos="${QOS}" \
  --job-name=qco-train-repair --export=ALL,RUN_KEY=qwen2_5_3b_8k \
  "${SCRIPT_DIR}/03_train.sbatch"
sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" \
  --partition="${PARTITION}" --qos="${QOS}" \
  --job-name=qco-finalize-r1 "${SCRIPT_DIR}/04_finalize.sbatch"

if [[ "${dry_run}" == "1" ]]; then
  echo "QWEN3_8B_AUDIT_REPAIR_DRY_RUN failed=${FAILED_CANARY_JOB} replace=212337-212342 audit_attempt=${AUDIT_ATTEMPT}"
  exit 0
fi

export FAILED_CANARY_JOB COMPLETED_PREVIOUS_CANARY_JOB FAILED_LOG AUDIT_ROOT
export REPAIR_PENDING REPAIR_FAILED REPAIR_COMMIT ORIGINAL_COMMIT
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


root = Path(os.environ["CANARY_ROOT"])
reused = {
    "custom_source": root / "source/custom_pretokenized.jsonl",
    "stock_source": root / "source/stock_messages.jsonl",
    "custom_checkpoint_index": root / "custom_pretokenized/weights/iter_0000000/model.safetensors.index.json",
    "stock_checkpoint_index": root / "stock_messages/weights/iter_0000000/model.safetensors.index.json",
}
value = {
    "artifact_schema_version": 1,
    "status": "submitting_replacement_chain",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "commit": os.environ["REPAIR_COMMIT"],
    "original_submission_commit": os.environ["ORIGINAL_COMMIT"],
    "contract_hash": os.environ["CONTRACT_HASH"],
    "reason": "Qwen3-8B canary audit omitted the required tensor-parallel-size argument",
    "retry_scope": "reuse completed custom and stock checkpoints; audit and generate only; replace stale full-training script snapshots",
    "audit_attempt": "snapshot_audit_r1",
    "audit_root": os.environ["AUDIT_ROOT"],
    "jobs": {
        "completed_preceding_canary": int(os.environ["COMPLETED_PREVIOUS_CANARY_JOB"]),
        "failed_canary": int(os.environ["FAILED_CANARY_JOB"]),
        "stale_pending_tail": list(range(212337, 212343)),
    },
    "failed_log": {
        "path": os.environ["FAILED_LOG"],
        "sha256": sha256(Path(os.environ["FAILED_LOG"])),
    },
    "reused_artifacts": {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in reused.items()
    },
}
output = Path(os.environ["REPAIR_PENDING"])
output.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(output)
PY

replacement_jobs=()
replacement_names=()
tail_job=""
replacement_chain_verified=0
stale_retirement_started=0
stale_jobs_canceled=0
repair_complete=0

record_failed_repair() {
  local shell_status="$1"
  export REPAIR_SHELL_STATUS="${shell_status}" REPLACEMENT_JOBS="${replacement_jobs[*]:-}"
  export REPLACEMENT_NAMES="${replacement_names[*]:-}"
  export REPLACEMENT_CHAIN_VERIFIED="${replacement_chain_verified}"
  export STALE_RETIREMENT_STARTED="${stale_retirement_started}"
  export STALE_JOBS_CANCELED="${stale_jobs_canceled}"
  python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

pending = Path(os.environ["REPAIR_PENDING"])
failed = Path(os.environ["REPAIR_FAILED"])
value = json.loads(pending.read_text()) if pending.is_file() else {}
job_ids = [int(value) for value in os.environ.get("REPLACEMENT_JOBS", "").split()]
names = os.environ.get("REPLACEMENT_NAMES", "").split()
value.update(
    {
        "status": "repair_failed",
        "failed_at": datetime.now(timezone.utc).isoformat(),
        "shell_status": int(os.environ["REPAIR_SHELL_STATUS"]),
        "replacement_chain_verified": os.environ["REPLACEMENT_CHAIN_VERIFIED"] == "1",
        "stale_retirement_started": os.environ["STALE_RETIREMENT_STARTED"] == "1",
        "stale_jobs_canceled": os.environ["STALE_JOBS_CANCELED"] == "1",
        "replacement_jobs": [
            {"name": name, "job_id": job_id}
            for name, job_id in zip(names, job_ids)
        ],
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
    if [[ "${stale_retirement_started}" != "1" && "${#replacement_jobs[@]}" -gt 0 ]]; then
      scancel "${replacement_jobs[@]}" || true
    fi
    record_failed_repair "${status}" || true
  fi
  exit "${status}"
}
trap fail_closed EXIT

submit_after_tail() {
  local name="$1"
  shift
  local args=(
    --parsable --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}"
    --partition="${PARTITION}" --qos="${QOS}"
  )
  if [[ -n "${tail_job}" ]]; then
    args+=(--dependency="afterok:${tail_job}")
  fi
  local raw
  raw="$(sbatch "${args[@]}" "$@")"
  tail_job="${raw%%;*}"
  [[ "${tail_job}" =~ ^[0-9]+$ ]] || {
    echo "Unexpected sbatch response: ${raw}" >&2
    exit 1
  }
  replacement_jobs+=("${tail_job}")
  replacement_names+=("${name}")
  echo "SUBMITTED_REPLACEMENT name=${name} job=${tail_job}"
}

submit_after_tail audit_qwen3_8b_8k \
  --job-name=qco-canary-8b8k-a1 \
  --export=ALL,RUN_KEY=qwen3_8b_8k,CANARY_AUDIT_ATTEMPT="${AUDIT_ATTEMPT}" \
  "${SCRIPT_DIR}/02b_resume_canary_audit.sbatch"
for run_key in "${RUN_KEYS[@]}"; do
  submit_after_tail "train_${run_key}" \
    --job-name="qco-train-${run_key}-r1" --export=ALL,RUN_KEY="${run_key}" \
    "${SCRIPT_DIR}/03_train.sbatch"
done
submit_after_tail finalize_handoff \
  --job-name=qco-finalize-r1 "${SCRIPT_DIR}/04_finalize.sbatch"

for index in "${!replacement_jobs[@]}"; do
  record="$(scontrol show job -o "${replacement_jobs[$index]}")"
  case "${index}" in
    0) expected_job_name=qco-canary-8b8k-a1 ;;
    1) expected_job_name=qco-train-qwen2_5_3b_8k-r1 ;;
    2) expected_job_name=qco-train-qwen2_5_3b_16k-r1 ;;
    3) expected_job_name=qco-train-qwen2_5_7b_8k-r1 ;;
    4) expected_job_name=qco-train-qwen3_4b_8k-r1 ;;
    5) expected_job_name=qco-train-qwen3_8b_8k-r1 ;;
    6) expected_job_name=qco-finalize-r1 ;;
  esac
  [[ "${record}" == *"JobName=${expected_job_name} "* ]] || {
    echo "Unexpected replacement job identity: ${record}" >&2
    exit 1
  }
  if [[ "${index}" -gt 0 ]]; then
    parent="${replacement_jobs[$((index - 1))]}"
    [[ "${record}" == *"Dependency=afterok:${parent}("* ]] || {
      echo "Replacement dependency drift for job ${replacement_jobs[$index]}." >&2
      exit 1
    }
  fi
done
replacement_chain_verified=1

export REPLACEMENT_JOBS="${replacement_jobs[*]}" REPLACEMENT_NAMES="${replacement_names[*]}"
python3 - <<'PY'
import json
import os
import tempfile
from pathlib import Path

path = Path(os.environ["REPAIR_PENDING"])
value = json.loads(path.read_text())
job_ids = [int(value) for value in os.environ["REPLACEMENT_JOBS"].split()]
names = os.environ["REPLACEMENT_NAMES"].split()
value.update(
    {
        "status": "replacement_chain_verified",
        "replacement_jobs": [
            {"name": name, "job_id": job_id}
            for name, job_id in zip(names, job_ids)
        ],
        "replacement_dependency_policy": "strict serial afterok",
    }
)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temporary = Path(handle.name)
temporary.replace(path)
PY

stale_retirement_started=1
scancel "${STALE_JOBS[@]}"
stale_jobs_canceled=1
for stale_job in "${STALE_JOBS[@]}"; do
  for _ in 1 2 3 4 5; do
    state="$(job_state "${stale_job}")"
    [[ "${state}" == CANCELLED* ]] && break
    sleep 1
  done
  [[ "${state}" == CANCELLED* ]] || {
    echo "Stale job ${stale_job} did not reach CANCELLED; state=${state}" >&2
    exit 1
  }
done

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
        "status": "submitted_and_replaced_stale_tail",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "canceled_stale_jobs": list(range(212337, 212343)),
        "canceled_stale_job_states": ["CANCELLED"] * 6,
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
echo "QWEN3_8B_AUDIT_REPAIR_COMPLETE audit_job=${replacement_jobs[0]} first_train=${replacement_jobs[1]} finalize=${replacement_jobs[6]} canceled=212337-212342"
