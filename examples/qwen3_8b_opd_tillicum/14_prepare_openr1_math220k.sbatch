#!/bin/bash
#SBATCH --job-name=slime-qwen3-openr1-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h200:1
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_8b_opd_tillicum"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
source "${SCRIPT_DIR}/openr1_math220k_env.sh"

mkdir -p "${SLURM_LOG_DIR}" "${OPENR1_DATA_DIR}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Preparing OpenR1 data on $(hostname) at $(date)"
echo "Dataset/config/split/revision: ${OPENR1_DATASET}/${OPENR1_CONFIG}/${OPENR1_SPLIT}/${OPENR1_REVISION}"
echo "Seed and SFT/OPD quotas: ${DATA_SEED} ${SFT_SIZE}/${CLEANED_OPD_FIRST_SIZE}/${CLEANED_OPD_NEXT_SIZE}"
echo "Output root: ${OPENR1_DATA_DIR}"

cd "${SLIME_REPO_ROOT}"
"${SCRIPT_DIR}/container_exec.sh" python3 examples/qwen3_8b_opd_tillicum/prepare_openr1_math220k.py

for spec in \
  "${SFT_PARQUET}:50000" \
  "${CLEANED_OPD_RESERVE_JSONL}:5120" \
  "${CLEANED_OPD_1K_JSONL}:1024" \
  "${CLEANED_OPD_4K_JSONL}:4096"; do
  path="${spec%:*}"
  expected="${spec##*:}"
  actual="$(wc -l <"${path}")"
  if [[ "${actual}" -ne "${expected}" ]]; then
    echo "Row-count mismatch for ${path}: expected ${expected}, found ${actual}" >&2
    exit 1
  fi
done

python3 - "${SPLIT_METADATA}" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
for field in ("source_id_intersection", "uuid_intersection", "prompt_hash_intersection"):
    if metadata["disjointness"][field] != 0:
        raise SystemExit(f"Nonzero OpenR1 split overlap: {field}={metadata['disjointness'][field]}")
print(json.dumps(metadata["selection"], indent=2))
print(json.dumps(metadata["disjointness"], indent=2))
PY

echo "Finished OpenR1 data preparation at $(date)"
