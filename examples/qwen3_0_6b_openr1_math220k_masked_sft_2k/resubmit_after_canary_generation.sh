#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

readonly COMPLETED_MODEL_JOB=207511
readonly COMPLETED_DATA_JOB=207512
readonly FAILED_CANARY_JOB=207513
readonly BLOCKED_SCORE_JOB=207514
readonly NEXT_TRAIN_JOB=207515
readonly TAIL_TRAIN_JOB=207525
readonly ORIGINAL_MANIFEST="${MANIFEST_ROOT}/training_chain.json"
readonly FAILED_LOG="${SLURM_LOG_DIR}/q3-06b-canary_${FAILED_CANARY_JOB}.log"
readonly REPAIR_STEM="canary_generation_repair"
readonly REPAIR_MANIFEST="${MANIFEST_ROOT}/${REPAIR_STEM}.json"
readonly REPAIR_PENDING="${MANIFEST_ROOT}/${REPAIR_STEM}.pending.json"
readonly REPAIR_FAILED="${MANIFEST_ROOT}/${REPAIR_STEM}.failed.json"
readonly CUSTOM_FINAL="${CANARY_ROOT}/custom_pretokenized/weights/iter_0000001"
readonly STOCK_FINAL="${CANARY_ROOT}/stock_messages/weights/iter_0000001"

python3 "${SCRIPT_DIR}/validate_config.py" \
  --example-dir "${SCRIPT_DIR}" --require-pushed-clean
for path in "${REPAIR_MANIFEST}" "${REPAIR_PENDING}" "${REPAIR_FAILED}"; do
  [[ ! -e "${path}" ]] || {
    echo "Refusing to overwrite prior repair state: ${path}" >&2
    exit 1
  }
done
[[ ! -e "${CANARY_GENERATION_JSON}" ]] || {
  echo "Generation diagnostic already exists: ${CANARY_GENERATION_JSON}" >&2
  exit 1
}

job_state() {
  sacct -X -n -P --jobs="$1" --format=State | sed -n '1{s/|.*//;p}'
}
[[ "$(job_state "${COMPLETED_MODEL_JOB}")" == COMPLETED ]] || {
  echo "Expected completed model-preparation job ${COMPLETED_MODEL_JOB}." >&2
  exit 1
}
[[ "$(job_state "${COMPLETED_DATA_JOB}")" == COMPLETED ]] || {
  echo "Expected completed data-preparation job ${COMPLETED_DATA_JOB}." >&2
  exit 1
}
[[ "$(job_state "${FAILED_CANARY_JOB}")" == FAILED* ]] || {
  echo "Expected failed canary job ${FAILED_CANARY_JOB}." >&2
  exit 1
}
[[ -f "${FAILED_LOG}" ]] || {
  echo "Missing failed canary log: ${FAILED_LOG}" >&2
  exit 1
}
for needle in \
  "RUNTIME_BATCH_AUDIT_VALID samples=64 decoded_token_rows=278798" \
  "KeyError: 'shape'" \
  "batch_size = inputs_tensor.shape[0]" \
  "AttributeError"; do
  grep -Fq "${needle}" "${FAILED_LOG}" || {
    echo "Failed canary log is missing the approved failure marker: ${needle}" >&2
    exit 1
  }
done

blocked_record="$(scontrol show job -o "${BLOCKED_SCORE_JOB}")"
[[ "${blocked_record}" == *"JobName=q3-06b-score "* ]] || {
  echo "Unexpected blocked score-job identity: ${blocked_record}" >&2
  exit 1
}
[[ "${blocked_record}" == *"JobState=PENDING "* ]] || {
  echo "Expected score job ${BLOCKED_SCORE_JOB} to remain pending." >&2
  exit 1
}
[[ "${blocked_record}" == *"Dependency=afterok:${FAILED_CANARY_JOB}(failed)"* ]] || {
  echo "Blocked score dependency changed; refusing repair." >&2
  exit 1
}
for ((job_id = NEXT_TRAIN_JOB; job_id <= TAIL_TRAIN_JOB; job_id++)); do
  expected_dependency=$((job_id - 1))
  downstream_record="$(scontrol show job -o "${job_id}")"
  [[ "${downstream_record}" == *"JobState=PENDING "* ]] || {
    echo "Expected downstream job ${job_id} to remain pending." >&2
    exit 1
  }
  [[ "${downstream_record}" == *"Dependency=afterok:${expected_dependency}(unfulfilled)"* ]] || {
    echo "Downstream dependency drift for job ${job_id}." >&2
    exit 1
  }
done

export ORIGINAL_MANIFEST
python3 - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path(os.environ["ORIGINAL_MANIFEST"]).read_text())
actual = [int(item["job_id"]) for item in manifest["jobs_in_dependency_order"]]
if actual != list(range(207511, 207526)):
    raise RuntimeError(f"original job inventory drift: {actual}")
if manifest["contract_hash"] != "9c736f35abc76095":
    raise RuntimeError("original contract hash drift")
if manifest["commit"] != "48daf1003c4fdc77ac2763d94cf835e316a97ca0":
    raise RuntimeError("original submission commit drift")
PY

check_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  read -r actual _ < <(sha256sum "${path}")
  [[ "${actual}" == "${expected}" ]] || {
    echo "SHA-256 mismatch for ${path}: ${actual}" >&2
    exit 1
  }
}
check_sha256 "${CUSTOM_FINAL}/model.safetensors" \
  114ef55784275913daca9412a05c3aec98f5d87301498a5b784c9ddcf0ebca90
check_sha256 "${STOCK_FINAL}/model.safetensors" \
  04922866b39dddd21311c4a7c5f6ed1ce7a9ef5147bfc5a22afe92ad5b3e7b83
check_sha256 "${CUSTOM_FINAL}/config.json" \
  2ef72b777467b5747e58414b7f0d7b62833a35f2915f357e2a3a7d1f5a645edc
check_sha256 "${STOCK_FINAL}/config.json" \
  2ef72b777467b5747e58414b7f0d7b62833a35f2915f357e2a3a7d1f5a645edc
check_sha256 "${CANARY_CUSTOM_JSONL}" \
  686ffb8100d2186692cee59174ff7e31e40194982b9a3e8966a69c058f21aff3
check_sha256 "${CANARY_MESSAGES_JSONL}" \
  2d268fefb5a3f1c79ce8c747305559c7bfab79ddff37fcd6f85aa3a1b4b0578f
check_sha256 "${CANARY_AUDIT_JSON}" \
  28105350fd5c29f29ecfced5d454a570f8cc9ca2fe1d1eba6760ae676181dbe3
check_sha256 "${CANARY_TOKEN_DUMP_JSONL}" \
  8df4bd72d116ba9c91e6653ff911f9aea23315e5cf2c1d40e61f4ab99456b925
check_sha256 "${FAILED_LOG}" \
  d1bc62c206caf1814e7be0927f5ae2504a98777f8288346e356a9548267cd75a
[[ "$(wc -l < "${CANARY_CUSTOM_JSONL}")" == 32 ]] || exit 1
[[ "$(wc -l < "${CANARY_MESSAGES_JSONL}")" == 32 ]] || exit 1
[[ "$(wc -l < "${CANARY_TOKEN_DUMP_JSONL}")" == 278798 ]] || exit 1
[[ "$(< "${CANARY_ROOT}/custom_pretokenized/full_state/latest_checkpointed_iteration.txt")" == 1 ]] || exit 1
[[ "$(< "${CANARY_ROOT}/stock_messages/full_state/latest_checkpointed_iteration.txt")" == 1 ]] || exit 1

export CANARY_AUDIT_JSON
python3 - <<'PY'
import json
import os
from pathlib import Path

audit = json.loads(Path(os.environ["CANARY_AUDIT_JSON"]).read_text())
expected = {
    "custom_pretokenized": {
        "runtime_samples": 32,
        "runtime_rollouts": {"0": 16, "1": 16},
        "microbatches_per_dp_rank_per_update": 1,
        "max_sequence_length": 7929,
        "max_response_length": 7829,
        "terminal_newline_weights": {"0.0": 32},
        "total_supervised_weight": 136018.0,
    },
    "stock_messages": {
        "runtime_samples": 32,
        "runtime_rollouts": {"0": 16, "1": 16},
        "microbatches_per_dp_rank_per_update": 1,
        "max_sequence_length": 7929,
        "max_response_length": 7829,
        "terminal_newline_weights": {"1.0": 32},
        "total_supervised_weight": 136050.0,
    },
}
for section, fields in expected.items():
    for key, value in fields.items():
        if audit[section][key] != value:
            raise RuntimeError(f"runtime audit drift at {section}.{key}")
PY

export CUSTOM_FINAL STOCK_FINAL REPAIR_COMMIT
REPAIR_COMMIT="$(git rev-parse HEAD)"
export REPAIR_PENDING FAILED_LOG
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["REPAIR_PENDING"])
value = {
    "artifact_schema_version": 1,
    "status": "submitting_generation_replacement",
    "started_at": datetime.now(timezone.utc).isoformat(),
    "commit": os.environ["REPAIR_COMMIT"],
    "contract_hash": os.environ["CONTRACT_HASH"],
    "reason": "BatchEncoding was passed positionally to model.generate after both canary trainings completed",
    "copy_regression_policy": "diagnostic_only_does_not_block_training",
    "failed_log": os.environ["FAILED_LOG"],
    "jobs": {
        "completed_model_preparation": 207511,
        "completed_data_preparation": 207512,
        "failed_canary": 207513,
        "blocked_score_to_rewire": 207514,
        "preserved_downstream_first": 207515,
        "preserved_downstream_last": 207525,
    },
    "reused_canary_artifacts": {
        "custom_hf_checkpoint": os.environ["CUSTOM_FINAL"],
        "stock_hf_checkpoint": os.environ["STOCK_FINAL"],
        "runtime_audit": os.environ["CANARY_AUDIT_JSON"],
        "runtime_token_dump": os.environ["CANARY_TOKEN_DUMP_JSONL"],
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
  if [[ "${repair_complete}" != 1 ]]; then
    if [[ -n "${replacement_job}" && "${dependency_rewired}" != 1 ]]; then
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
  --job-name=q3-06b-canary-gen-r1 \
  "${SCRIPT_DIR}/02_canary_generation_only.sbatch")"
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
next_record="$(scontrol show job -o "${NEXT_TRAIN_JOB}")"
[[ "${next_record}" == *"Dependency=afterok:${BLOCKED_SCORE_JOB}(unfulfilled)"* ]] || {
  echo "First training dependency changed during repair." >&2
  exit 1
}
tail_record="$(scontrol show job -o "${TAIL_TRAIN_JOB}")"
[[ "${tail_record}" == *"Dependency=afterok:207524(unfulfilled)"* ]] || {
  echo "Tail training dependency changed during repair." >&2
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
        "replacement_canary_generation": int(os.environ["REPLACEMENT_JOB"]),
        "rewired_dependency": {
            "job_id": 207514,
            "dependency": f"afterok:{os.environ['REPLACEMENT_JOB']}",
        },
        "reused_jobs": list(range(207514, 207526)),
        "generation_output": os.environ["CANARY_GENERATION_JSON"],
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
echo "CANARY_GENERATION_REPAIR_COMPLETE failed=${FAILED_CANARY_JOB} replacement=${replacement_job} rewired=${BLOCKED_SCORE_JOB} reused=207514-207525"
