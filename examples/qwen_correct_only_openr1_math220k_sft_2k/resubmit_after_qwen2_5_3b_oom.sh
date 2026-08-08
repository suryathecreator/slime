#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

readonly COMPLETED_AUDIT_JOB=212698
readonly FAILED_TRAINING_JOB=212699
readonly PRIOR_REPAIR_MANIFEST="${MANIFEST_ROOT}/qwen3_8b_audit_repair.json"
readonly FAILED_LOG="${SLURM_LOG_DIR}/qco-train-qwen2_5_3b_8k-r1_${FAILED_TRAINING_JOB}.log"
readonly FAILED_ROOT="${OUTPUT_ROOT}/training/qwen2_5_3b_8k/correct_only"
readonly RETRY_ROOT="${FAILED_ROOT}/attempts/oom_r1"
readonly PREFIX_ATTEMPT=optimizer_offload_r1
readonly PREFIX_8K_ROOT="${OUTPUT_ROOT}/training_prefix_canaries/qwen2_5_3b_8k/${PREFIX_ATTEMPT}"
readonly PREFIX_16K_ROOT="${OUTPUT_ROOT}/training_prefix_canaries/qwen2_5_3b_16k/${PREFIX_ATTEMPT}"
readonly REPAIR_MANIFEST="${MANIFEST_ROOT}/qwen2_5_3b_oom_repair.json"
readonly REPAIR_PENDING="${MANIFEST_ROOT}/qwen2_5_3b_oom_repair.pending.json"
readonly REPAIR_FAILED="${MANIFEST_ROOT}/qwen2_5_3b_oom_repair.failed.json"
readonly -a STALE_JOBS=(212700 212701 212702 212703 212704)
readonly -a STALE_NAMES=(
  qco-train-qwen2_5_3b_16k-r1
  qco-train-qwen2_5_7b_8k-r1
  qco-train-qwen3_4b_8k-r1
  qco-train-qwen3_8b_8k-r1
  qco-finalize-r1
)
readonly -a STALE_PARENTS=(212699 212700 212701 212702 212703)
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
    echo "Refusing to overwrite prior Qwen2.5-3B OOM repair state: ${path}" >&2
    exit 1
  }
done

job_state() {
  sacct -X -n -P --jobs="$1" --format=State | sed -n '1{s/|.*//;p}'
}

[[ "$(job_state "${COMPLETED_AUDIT_JOB}")" == COMPLETED* ]] || {
  echo "Expected completed Qwen3-8B audit ${COMPLETED_AUDIT_JOB}." >&2
  exit 1
}
[[ "$(job_state "${FAILED_TRAINING_JOB}")" == FAILED* ]] || {
  echo "Expected failed Qwen2.5-3B training job ${FAILED_TRAINING_JOB}." >&2
  exit 1
}
[[ -f "${FAILED_LOG}" ]] || { echo "Missing failed log ${FAILED_LOG}." >&2; exit 1; }
for signature in \
  "SFT max tokens per GPU: 16384" \
  "SFT optimizer CPU offload: 0" \
  "step 0:" "step 1:" "step 2:" \
  "torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 4.64 GiB" \
  "of which 4.36 GiB is free"; do
  grep -Fq "${signature}" "${FAILED_LOG}" || {
    echo "Failed training log is missing diagnosed signature: ${signature}" >&2
    exit 1
  }
done

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

export PRIOR_REPAIR_MANIFEST FAILED_ROOT RETRY_ROOT PREFIX_8K_ROOT PREFIX_16K_ROOT
export PREP_STATS_JSON SCHEDULE_AUDIT_DIR OUTPUT_ROOT HANDOFF_ROOT FAILED_LOG
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


prior = json.loads(Path(os.environ["PRIOR_REPAIR_MANIFEST"]).read_text())
if prior["status"] != "submitted_and_replaced_stale_tail":
    raise RuntimeError("Qwen3-8B audit repair is not complete")
if prior["commit"] != "bb1cb9b7e291e7eac23204502113ef38a098029c":
    raise RuntimeError("Qwen3-8B audit repair commit drift")
if prior["replacement_jobs"][0] != {
    "job_id": 212698,
    "name": "audit_qwen3_8b_8k",
}:
    raise RuntimeError("Qwen3-8B replacement audit job drift")

failed_root = Path(os.environ["FAILED_ROOT"])
for rollout_id in range(5):
    path = failed_root / "details/rollout_data" / f"{rollout_id}.pt"
    if not path.is_file():
        raise RuntimeError(f"missing failed-run rollout evidence: {path}")
for rollout_id in range(3):
    dumps = sorted((failed_root / "details/train_data").glob(f"{rollout_id}_*.pt"))
    if len(dumps) != 4:
        raise RuntimeError(f"expected four rank dumps for completed step {rollout_id}")
if list((failed_root / "full_state").glob("iter_*")):
    raise RuntimeError("failed run unexpectedly contains an optimizer checkpoint")
if list((failed_root / "weights").glob("iter_*")):
    raise RuntimeError("failed run unexpectedly contains an HF checkpoint")
if (Path(os.environ["HANDOFF_ROOT"]) / "checkpoints/qwen2_5_3b_8k.json").exists():
    raise RuntimeError("failed run unexpectedly contains a handoff manifest")
for path in (
    Path(os.environ["RETRY_ROOT"]),
    Path(os.environ["PREFIX_8K_ROOT"]),
    Path(os.environ["PREFIX_16K_ROOT"]),
    Path(os.environ["SCHEDULE_AUDIT_DIR"]),
):
    if path.exists():
        raise RuntimeError(f"refusing to overwrite repair output: {path}")

stats_path = Path(os.environ["PREP_STATS_JSON"])
stats = json.loads(stats_path.read_text())
if stats["contract"]["rows_per_selection"] != 2000:
    raise RuntimeError("prepared row-count contract drift")
if stats["invariants"]["same_ordered_2k_across_all_8k_models"] is not True:
    raise RuntimeError("shared 8K trace invariant drift")
for run_key, run in stats["runs"].items():
    for path_key, hash_key in (("dataset", "dataset_sha256"), ("messages", "messages_sha256")):
        path = Path(run[path_key])
        if sha256(path) != run[hash_key]:
            raise RuntimeError(f"prepared artifact hash drift: {path}")

output_root = Path(os.environ["OUTPUT_ROOT"])
handoff_root = Path(os.environ["HANDOFF_ROOT"])
for run_key in (
    "qwen2_5_3b_16k",
    "qwen2_5_7b_8k",
    "qwen3_4b_8k",
    "qwen3_8b_8k",
):
    if (output_root / "training" / run_key).exists():
        raise RuntimeError(f"unexpected full-training output for {run_key}")
    if (handoff_root / "checkpoints" / f"{run_key}.json").exists():
        raise RuntimeError(f"unexpected handoff manifest for {run_key}")

print(
    "QWEN25_3B_OOM_INPUTS_VALID "
    f"failed_log_sha256={sha256(Path(os.environ['FAILED_LOG']))} "
    f"prep_stats_sha256={sha256(stats_path)}",
    flush=True,
)
PY

[[ "${SFT_MEMORY_PROFILE}" == "memory_r3" ]] || {
  echo "Unexpected memory profile ${SFT_MEMORY_PROFILE}." >&2
  exit 1
}
declare -A expected_profiles=(
  [qwen2_5_3b_8k]=1:10240:1
  [qwen2_5_3b_16k]=1:16384:1
  [qwen2_5_7b_8k]=2:16384:1
  [qwen3_4b_8k]=1:9216:0
  [qwen3_8b_8k]=2:16384:1
)
for run_key in "${RUN_KEYS[@]}"; do
  resolve_run "${run_key}"
  actual="${SFT_TENSOR_MODEL_PARALLEL_SIZE}:${SFT_MAX_TOKENS_PER_GPU}:${SFT_OPTIMIZER_CPU_OFFLOAD}"
  [[ "${actual}" == "${expected_profiles[$run_key]}" ]] || {
    echo "Runtime profile drift for ${run_key}: ${actual}" >&2
    exit 1
  }
done

sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" \
  --partition="${PARTITION}" --qos="${QOS}" \
  --job-name=qco-schedule-r4 --export=ALL,REUSE_PREPARED_DATA=1 \
  "${SCRIPT_DIR}/01_prepare_data.sbatch"
sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" \
  --partition="${PARTITION}" --qos="${QOS}" \
  --job-name=qco-prefix-3b8k-r1 \
  --export=ALL,RUN_KEY=qwen2_5_3b_8k,PREFIX_CANARY_ATTEMPT="${PREFIX_ATTEMPT}" \
  "${SCRIPT_DIR}/02c_training_prefix_canary.sbatch"
sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" \
  --partition="${PARTITION}" --qos="${QOS}" \
  --job-name=qco-train-3b8k-r2 \
  --export=ALL,RUN_KEY=qwen2_5_3b_8k,TRAINING_ATTEMPT=oom_r1 \
  "${SCRIPT_DIR}/03_train.sbatch"
sbatch --test-only \
  --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" \
  --partition="${PARTITION}" --qos="${QOS}" \
  --job-name=qco-finalize-r2 "${SCRIPT_DIR}/04_finalize.sbatch"

if [[ "${dry_run}" == "1" ]]; then
  echo "QWEN25_3B_OOM_REPAIR_DRY_RUN failed=${FAILED_TRAINING_JOB} replace=212700-212704 profile=memory_r3"
  exit 0
fi

export COMPLETED_AUDIT_JOB FAILED_TRAINING_JOB FAILED_LOG RETRY_ROOT
export PREFIX_8K_ROOT PREFIX_16K_ROOT REPAIR_PENDING REPAIR_FAILED REPAIR_COMMIT
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


value = {
    "artifact_schema_version": 1,
    "status": "submitting_replacement_chain",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "commit": os.environ["REPAIR_COMMIT"],
    "contract_hash": os.environ["CONTRACT_HASH"],
    "reason": "Qwen2.5-3B TP1 fused cross-entropy OOM on a packed 16384-token microbatch",
    "memory_profile": {
        "name": "memory_r3",
        "qwen2_5_3b_8k": "tp1:max_tokens10240:optimizer_cpu_offload1",
        "qwen2_5_3b_16k": "tp1:max_tokens16384:optimizer_cpu_offload1",
    },
    "retry_scope": "reuse prepared data; regenerate schedules; replay five prefixes per 3B run; restart all full runs from base",
    "training_retry_root": os.environ["RETRY_ROOT"],
    "prefix_canary_roots": [os.environ["PREFIX_8K_ROOT"], os.environ["PREFIX_16K_ROOT"]],
    "jobs": {
        "completed_preceding_audit": int(os.environ["COMPLETED_AUDIT_JOB"]),
        "failed_training": int(os.environ["FAILED_TRAINING_JOB"]),
        "stale_pending_tail": list(range(212700, 212705)),
    },
    "failed_log": {
        "path": os.environ["FAILED_LOG"],
        "sha256": sha256(Path(os.environ["FAILED_LOG"])),
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
job_ids = [int(item) for item in os.environ.get("REPLACEMENT_JOBS", "").split()]
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

submit_after_tail regenerate_memory_r3_schedules \
  --job-name=qco-schedule-r4 --export=ALL,REUSE_PREPARED_DATA=1 \
  "${SCRIPT_DIR}/01_prepare_data.sbatch"
submit_after_tail prefix_qwen2_5_3b_8k \
  --job-name=qco-prefix-3b8k-r1 \
  --export=ALL,RUN_KEY=qwen2_5_3b_8k,PREFIX_CANARY_ATTEMPT="${PREFIX_ATTEMPT}" \
  "${SCRIPT_DIR}/02c_training_prefix_canary.sbatch"
submit_after_tail prefix_qwen2_5_3b_16k \
  --job-name=qco-prefix-3b16k-r1 \
  --export=ALL,RUN_KEY=qwen2_5_3b_16k,PREFIX_CANARY_ATTEMPT="${PREFIX_ATTEMPT}" \
  "${SCRIPT_DIR}/02c_training_prefix_canary.sbatch"
submit_after_tail train_qwen2_5_3b_8k \
  --job-name=qco-train-qwen2_5_3b_8k-r2 \
  --export=ALL,RUN_KEY=qwen2_5_3b_8k,TRAINING_ATTEMPT=oom_r1 \
  "${SCRIPT_DIR}/03_train.sbatch"
for run_key in qwen2_5_3b_16k qwen2_5_7b_8k qwen3_4b_8k qwen3_8b_8k; do
  submit_after_tail "train_${run_key}" \
    --job-name="qco-train-${run_key}-r2" --export=ALL,RUN_KEY="${run_key}" \
    "${SCRIPT_DIR}/03_train.sbatch"
done
submit_after_tail finalize_handoff \
  --job-name=qco-finalize-r2 "${SCRIPT_DIR}/04_finalize.sbatch"

expected_names=(
  qco-schedule-r4
  qco-prefix-3b8k-r1
  qco-prefix-3b16k-r1
  qco-train-qwen2_5_3b_8k-r2
  qco-train-qwen2_5_3b_16k-r2
  qco-train-qwen2_5_7b_8k-r2
  qco-train-qwen3_4b_8k-r2
  qco-train-qwen3_8b_8k-r2
  qco-finalize-r2
)
for index in "${!replacement_jobs[@]}"; do
  record="$(scontrol show job -o "${replacement_jobs[$index]}")"
  [[ "${record}" == *"JobName=${expected_names[$index]} "* ]] || {
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
job_ids = [int(item) for item in os.environ["REPLACEMENT_JOBS"].split()]
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
        "canceled_stale_jobs": list(range(212700, 212705)),
        "canceled_stale_job_states": ["CANCELLED"] * 5,
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
echo "QWEN25_3B_OOM_REPAIR_COMPLETE schedule=${replacement_jobs[0]} prefix8k=${replacement_jobs[1]} prefix16k=${replacement_jobs[2]} first_train=${replacement_jobs[3]} finalize=${replacement_jobs[8]} canceled=212700-212704"
