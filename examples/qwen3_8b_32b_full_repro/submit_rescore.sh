#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
source "${SCRIPT_DIR}/env.sh"
cd "${SLIME_REPO_ROOT}"

mode="${1:-}"
[[ "${mode}" == "draft" || "${mode}" == "final" ]] || {
  echo "Usage: $0 {draft|final}" >&2
  exit 2
}
if [[ "${mode}" == "final" ]]; then
  [[ -f "${SCRIPT_DIR}/MATH500_SCORER_V3_EXPECTED.json" ]] &&
    [[ -f "${SCRIPT_DIR}/MATH500_SCORER_V3_REVIEW.json" ]] || {
    echo "Final submission requires V3 expected metrics and review manifest" >&2
    exit 1
  }
fi

expected_branch=qwen3-8b-32b-full-repro
branch="$(git branch --show-current)"
[[ "${branch}" == "${expected_branch}" ]] || {
  echo "Expected ${expected_branch}, found ${branch}" >&2
  exit 1
}
git diff --quiet && git diff --cached --quiet || {
  echo "Refusing to submit from a dirty checkout" >&2
  exit 1
}
head_sha="$(git rev-parse HEAD)"
remote_sha="$(git rev-parse "origin/${expected_branch}")"
[[ "${head_sha}" == "${remote_sha}" ]] || {
  echo "Launch commit is not pushed: local=${head_sha} remote=${remote_sha}" >&2
  exit 1
}

job_id=""
cancel_on_failure() {
  status=$?
  if [[ ${status} -ne 0 && -n "${job_id}" ]]; then
    scancel "${job_id}" || true
  fi
  exit "${status}"
}
trap cancel_on_failure EXIT

raw="$(
  sbatch --parsable \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --qos="${QOS}" \
    --export="ALL,RESCORE_MODE=${mode},EXPECTED_LAUNCH_COMMIT=${head_sha}" \
    "${SCRIPT_DIR}/06_rescore_math500.sbatch"
)"
job_id="${raw%%;*}"
[[ "${job_id}" =~ ^[0-9]+$ ]] || {
  echo "Unexpected sbatch job id: ${raw}" >&2
  exit 1
}

manifest="${MANIFEST_ROOT}/scorer_v3_${mode}_submission.json"
python3 - "${manifest}" "${job_id}" "${head_sha}" "${mode}" <<'PY'
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

path = Path(sys.argv[1])
mode = sys.argv[4]
scorer = "math500_strict_boxed_scorer_v3"
suffix = f"_draft_{sys.argv[3][:12]}" if mode == "draft" else ""
payload = {
    "schema_version": 1,
    "job_id": int(sys.argv[2]),
    "launch_commit": sys.argv[3],
    "branch": "qwen3-8b-32b-full-repro",
    "mode": mode,
    "scorer_version": scorer,
    "source_root": os.environ["EVAL_OUTPUT_ROOT"],
    "v2_root": os.path.join(
        os.environ["EVAL_OUTPUT_ROOT"], "rescored", "math500_event_scorer_v2"
    ),
    "output_root": os.path.join(
        os.environ["EVAL_OUTPUT_ROOT"], "rescored", scorer + suffix
    ),
    "resources": {
        "gpu": "h200",
        "gpus": 1,
        "cpus": 8,
        "memory": "64G",
        "walltime": "04:00:00",
    },
    "submitted_at": datetime.now(timezone.utc).isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
    json.dump(payload, handle, indent=2, sort_keys=True)
    handle.write("\n")
    temp = Path(handle.name)
temp.replace(path)
PY

trap - EXIT
echo "SUBMITTED_SCORER_V3_${mode^^} job=${job_id} manifest=${manifest}"
