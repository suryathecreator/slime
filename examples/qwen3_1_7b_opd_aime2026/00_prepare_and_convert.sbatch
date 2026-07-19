#!/bin/bash
#SBATCH --job-name=q3-aime26-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_1_7b_opd_aime2026/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_1_7b_opd_aime2026"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${SLURM_LOG_DIR}" "${DATA_ROOT}" "${MODEL_ROOT}" "${VLLM_CACHE_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

cd "${SLIME_REPO_ROOT}"
echo "Preparing Qwen3-1.7B/Qwen3-8B AIME 2026 experiment on $(hostname) at $(date)"
bash examples/qwen3_1_7b_opd_aime2026/write_source_manifest.sh "${SOURCE_MANIFEST}"
export SLIME_SGLANG_PATCH_SITE=0
export SLIME_VLLM_PATCH_SITE=0

"${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
mkdir -p "${AIME_SOURCE_DIR}" "${MODEL_ROOT}" "${DATA_ROOT}" "${VLLM_CACHE_ROOT}"
export PYTHONPATH="${VLLM_EVAL_SITE}:${AIME_EXAMPLE_DIR}:${SLIME_REPO_ROOT}:${PYTHONPATH:-}"

python3 - <<"PY"
import importlib.metadata as metadata
import pyarrow
expected = {"vllm": "0.24.0", "transformers": "5.13.0"}
for package, wanted in expected.items():
    found = metadata.version(package)
    print(package, found)
    if found != wanted:
        raise SystemExit(f"Dependency mismatch for {package}: expected {wanted}, found {found}")
print("pyarrow", pyarrow.__version__)
PY

hf download "${AIME_DATASET_REPO}" \
  --repo-type dataset \
  --revision "${AIME_DATASET_REVISION}" \
  --include README.md \
  --include data/train-00000-of-00001.parquet \
  --local-dir "${AIME_SOURCE_DIR}"
python3 examples/qwen3_1_7b_opd_aime2026/prepare_experiment.py \
  --aime-parquet "${AIME_PARQUET}" \
  --aime-output "${AIME_JSONL}" \
  --opd-source-1k "${SOURCE_OPD_1K_JSONL}" \
  --opd-source-4k "${SOURCE_OPD_4K_JSONL}" \
  --opd-output-1k "${CLEANED_OPD_1K_JSONL}" \
  --opd-output-4k "${CLEANED_OPD_4K_JSONL}" \
  --metadata "${AIME_PREP_METADATA}"

hf download "${STUDENT_HF_REPO}" --revision "${STUDENT_HF_REVISION}" --local-dir "${STUDENT_HF_DIR}"
hf download "${TEACHER_HF_REPO}" --revision "${TEACHER_HF_REVISION}" --local-dir "${TEACHER_HF_DIR}"
python3 - <<"PY"
import json, os
from pathlib import Path
payload = {
    "student": {"repo": os.environ["STUDENT_HF_REPO"], "revision": os.environ["STUDENT_HF_REVISION"]},
    "teacher": {"repo": os.environ["TEACHER_HF_REPO"], "revision": os.environ["TEACHER_HF_REVISION"]},
}
Path(os.environ["MODEL_ROOT"], "model_revisions.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

python3 examples/qwen3_8b_opd_tillicum/validate_qwen3_generation.py \
  --tokenizer "${STUDENT_HF_DIR}" \
  --im-end-token-id "${QWEN3_IM_END_TOKEN_ID}" \
  --endoftext-token-id "${QWEN3_ENDOFTEXT_TOKEN_ID}"
python3 examples/qwen3_8b_opd_tillicum/validate_qwen3_generation.py \
  --tokenizer "${TEACHER_HF_DIR}" \
  --im-end-token-id "${QWEN3_IM_END_TOKEN_ID}" \
  --endoftext-token-id "${QWEN3_ENDOFTEXT_TOKEN_ID}"

if [[ ! -f "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt" ]]; then
  source "${STUDENT_MODEL_ARGS_SCRIPT}"
  mkdir -p "${STUDENT_TORCH_DIST_DIR}"
  PYTHONPATH="/root/Megatron-LM:${SLIME_REPO_ROOT}" \
  torchrun --nproc_per_node=4 tools/convert_hf_to_torch_dist.py \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "${STUDENT_HF_DIR}" \
    --save "${STUDENT_TORCH_DIST_DIR}"
fi

test -s "${AIME_JSONL}"
test -s "${AIME_PREP_METADATA}"
test -f "${STUDENT_HF_DIR}/config.json"
test -f "${TEACHER_HF_DIR}/config.json"
test -f "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt"
python3 - <<"PY"
import json, os
from pathlib import Path
rows = [json.loads(line) for line in Path(os.environ["AIME_JSONL"]).read_text().splitlines() if line]
assert len(rows) == 30
assert [row["problem_idx"] for row in rows] == list(range(1, 31))
print("Prepared", len(rows), "AIME 2026 problems")
PY
'

echo "Finished AIME 2026 setup at $(date)"
