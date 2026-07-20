#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

[[ "$(git branch --show-current)" == "qwen3-8b-32b-full-repro" ]] || { echo "Wrong branch." >&2; exit 1; }
[[ -z "$(git status --porcelain)" ]] || { echo "Refusing submission from a dirty worktree." >&2; exit 1; }
python3 "${SCRIPT_DIR}/validate_config.py" --require-pinned-commit

upstream="$(git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}' 2>/dev/null || true)"
[[ -n "${upstream}" ]] || { echo "Push this branch and set its upstream before submission." >&2; exit 1; }
[[ "$(git rev-parse HEAD)" == "$(git rev-parse "${upstream}")" ]] || { echo "Local HEAD is not the pushed upstream commit." >&2; exit 1; }

for script in 00_preflight.sbatch 01_cleanup.sbatch 02_eval_math500.sbatch 03_sft.sbatch 04_opd.sbatch; do
  grep -Eq '^#SBATCH --gres=gpu:h200:4$' "${SCRIPT_DIR}/${script}" || {
    echo "GPU contract violation in ${script}" >&2
    exit 1
  }
done

if [[ -e "${MANIFEST_ROOT}/submission.json" || -e "${MANIFEST_ROOT}/submission.pending.json" ]]; then
  echo "Refusing to overwrite a prior or partial submission under ${MANIFEST_ROOT}" >&2
  exit 1
fi
if find "${OUTPUT_ROOT}" -mindepth 1 -print -quit 2>/dev/null | grep -q .; then
  echo "Refusing nonempty output root for a new contract: ${OUTPUT_ROOT}" >&2
  exit 1
fi

mkdir -p "${SLURM_LOG_DIR}" "${MANIFEST_ROOT}" "${OUTPUT_ROOT}" "${TMPDIR}"
export SUBMISSION_PENDING_PATH="${MANIFEST_ROOT}/submission.pending.json"
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["SUBMISSION_PENDING_PATH"])
value = {"status": "submitting", "started_at": datetime.now(timezone.utc).isoformat(), "contract_hash": os.environ["CONTRACT_HASH"]}
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temp = Path(handle.name)
temp.replace(path)
PY

submit_job() {
  local dependency="$1"
  shift
  local -a args=(--parsable --chdir="${SLIME_REPO_ROOT}" --account="${ACCOUNT}" --partition="${PARTITION}" --qos="${QOS}")
  if [[ -n "${dependency}" ]]; then args+=(--dependency="afterok:${dependency}"); fi
  local raw
  raw="$(sbatch "${args[@]}" "$@")"
  printf '%s' "${raw%%;*}"
}

preflight_job="$(submit_job "" --job-name=q8b32b-preflight "${SCRIPT_DIR}/00_preflight.sbatch")"
cleanup_job="$(submit_job "${preflight_job}" --job-name=q8b32b-cleanup "${SCRIPT_DIR}/01_cleanup.sbatch")"
base_eval_job="$(submit_job "${cleanup_job}" --job-name=q8b32b-eval-base --export=ALL,EVAL_STAGE=qwen3_8b_base,EVAL_MODEL_DIR="${STUDENT_HF_DIR}" "${SCRIPT_DIR}/02_eval_math500.sbatch")"
teacher_eval_job="$(submit_job "${base_eval_job}" --job-name=q8b32b-eval-teacher --export=ALL,EVAL_STAGE=qwen3_32b_teacher,EVAL_MODEL_DIR="${TEACHER_HF_DIR}" "${SCRIPT_DIR}/02_eval_math500.sbatch")"
sft_job="$(submit_job "${teacher_eval_job}" --job-name=q8b32b-sft200k "${SCRIPT_DIR}/03_sft.sbatch")"
sft_eval_job="$(submit_job "${sft_job}" --job-name=q8b32b-eval-sft --export=ALL,EVAL_STAGE=sft_200000,EVAL_MODEL_DIR="${SFT_FINAL_HF_DIR}" "${SCRIPT_DIR}/02_eval_math500.sbatch")"
opd_job="$(submit_job "${sft_eval_job}" --job-name=q8b32b-opd1472 "${SCRIPT_DIR}/04_opd.sbatch")"
opd_eval_job="$(submit_job "${opd_job}" --job-name=q8b32b-eval-opd --export=ALL,EVAL_STAGE=opd_100pct,EVAL_MODEL_DIR="${OPD_FINAL_HF_DIR}",EVAL_COMPARE_TO="${EVAL_OUTPUT_ROOT}/sft_200000/summary.json",BUILD_FINAL_REPORT=1 "${SCRIPT_DIR}/02_eval_math500.sbatch")"

export preflight_job cleanup_job base_eval_job teacher_eval_job sft_job sft_eval_job opd_job opd_eval_job
python3 - <<'PY'
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(os.environ["MANIFEST_ROOT"]) / "submission.json"
value = {
    "submitted_at": datetime.now(timezone.utc).isoformat(),
    "contract_hash": os.environ["CONTRACT_HASH"],
    "commit": os.popen("git rev-parse HEAD").read().strip(),
    "gpu_contract": "every job requests exactly four H200 GPUs",
    "jobs": {
        "preflight": os.environ["preflight_job"],
        "cleanup": os.environ["cleanup_job"],
        "base_eval": os.environ["base_eval_job"],
        "teacher_eval": os.environ["teacher_eval_job"],
        "sft_200k": os.environ["sft_job"],
        "sft_eval": os.environ["sft_eval_job"],
        "opd_1472x4": os.environ["opd_job"],
        "opd_eval_100pct": os.environ["opd_eval_job"],
    },
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
    json.dump(value, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temp = Path(handle.name)
temp.replace(path)
Path(os.environ["SUBMISSION_PENDING_PATH"]).unlink()
print(json.dumps(value, indent=2, sort_keys=True))
PY

echo "Submitted serial afterok chain ending in job ${opd_eval_job}."
