#!/bin/bash
#SBATCH --job-name=q3-06b-canary-gen-r1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_0_6b_openr1_math220k_masked_sft_2k"
source "${SCRIPT_DIR}/env.sh"
: "${SLURM_JOB_ID:?Canary generation must run under Slurm}"
export TMPDIR="/tmp/${USER:-suryadv}/q3-06b-canary-gen-${SLURM_JOB_ID}"
export RAY_TMPDIR="${TMPDIR}/ray"
mkdir -p "${TMPDIR}" "${RAY_TMPDIR}" "${SLURM_LOG_DIR}"
exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1

CUSTOM_FINAL="${CANARY_ROOT}/custom_pretokenized/weights/iter_0000001"
STOCK_FINAL="${CANARY_ROOT}/stock_messages/weights/iter_0000001"
required_files=(
  "${STUDENT_HF_DIR}/config.json"
  "${STUDENT_HF_DIR}/model.safetensors"
  "${STUDENT_HF_DIR}/tokenizer.json"
  "${CUSTOM_FINAL}/config.json"
  "${CUSTOM_FINAL}/model.safetensors"
  "${STOCK_FINAL}/config.json"
  "${STOCK_FINAL}/model.safetensors"
  "${CANARY_AUDIT_JSON}"
  "${CANARY_TOKEN_DUMP_JSONL}"
  "${CANARY_MESSAGES_JSONL}"
)
for path in "${required_files[@]}"; do
  [[ -f "${path}" ]] || {
    echo "Missing required canary artifact: ${path}" >&2
    exit 1
  }
done
[[ "$(wc -l < "${CANARY_MESSAGES_JSONL}")" == 32 ]] || {
  echo "Canary message row count drifted." >&2
  exit 1
}
[[ "$(wc -l < "${CANARY_TOKEN_DUMP_JSONL}")" == 278798 ]] || {
  echo "Runtime token-dump row count drifted." >&2
  exit 1
}
[[ ! -e "${CANARY_GENERATION_JSON}" ]] || {
  echo "Refusing to overwrite ${CANARY_GENERATION_JSON}" >&2
  exit 1
}

"${SCRIPT_DIR}/container_exec.sh" python3 "${SCRIPT_DIR}/canary_generate.py" \
  --base "${STUDENT_HF_DIR}" --custom "${CUSTOM_FINAL}" --stock "${STOCK_FINAL}" \
  --prompts "${CANARY_MESSAGES_JSONL}" --output "${CANARY_GENERATION_JSON}" --count 8

python3 - "${CANARY_GENERATION_JSON}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
diagnostic = json.loads(path.read_text())
expected = {
    "base",
    "custom_pretokenized_after_two_updates",
    "stock_messages_after_two_updates",
}
if set(diagnostic.get("generation", {})) != expected:
    raise RuntimeError("generation diagnostic model inventory drift")
for label, result in diagnostic["generation"].items():
    rows = result.get("rows", [])
    if len(rows) != 8 or any(not isinstance(row.get("completion"), str) for row in rows):
        raise RuntimeError(f"invalid verbatim completions for {label}")
copy_regression = diagnostic.get("copy_regression", {})
if copy_regression.get("policy") != "diagnostic_only_does_not_block_training":
    raise RuntimeError("copy-regression policy drift")
if len(copy_regression.get("comparisons", [])) != 4:
    raise RuntimeError("copy-regression comparison inventory drift")
print(
    "CANARY_GENERATION_DIAGNOSTIC_COMPLETE "
    f"copy_regression_detected={int(bool(copy_regression.get('detected')))} "
    "copy_regression_policy=diagnostic_only",
    flush=True,
)
PY

echo "QWEN3_0P6B_CANARY_GENERATION_RECOVERY_COMPLETE at=$(date -Is)"
