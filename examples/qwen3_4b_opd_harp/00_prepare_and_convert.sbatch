#!/bin/bash
#SBATCH --job-name=q3-4b-harp-setup
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --mail-user=suryadv@cs.washington.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp/env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/examples/qwen3_4b_opd_harp"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
fi
source "${SCRIPT_DIR}/env.sh"
mkdir -p "${SLURM_LOG_DIR}" "${DATA_ROOT}" "${MODEL_ROOT}" "${HARP_EVAL_SITE}" "${VLLM_CACHE_ROOT}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

cd "${SLIME_REPO_ROOT}"
echo "Preparing Qwen3-4B/HARP V2 experiment on $(hostname) at $(date)"
{
  find examples/qwen3_4b_opd_harp -type f ! -path '*/__pycache__/*' -print
  printf '%s\n' \
    examples/qwen3_8b_opd_tillicum/03_convert_models_if_needed.sbatch \
    examples/qwen3_8b_opd_tillicum/05_run_opd_50k_8xh200.sbatch \
    examples/qwen3_8b_opd_tillicum/container_exec.sh \
    examples/qwen3_8b_opd_tillicum/env.sh \
    examples/qwen3_8b_opd_tillicum/write_opd_trained_manifest.py \
    scripts/models/qwen3-4B.sh
} | sort -u | xargs sha256sum >"${SOURCE_MANIFEST}"
export SLIME_SGLANG_PATCH_SITE=0
export SLIME_VLLM_PATCH_SITE=0

"${SLIME_REPO_ROOT}/examples/qwen3_8b_opd_tillicum/container_exec.sh" bash -lc '
set -euo pipefail
cd "${SLIME_REPO_ROOT}"
mkdir -p "${HARP_EVAL_SITE}" "${MODEL_ROOT}" "${DATA_ROOT}" "${VLLM_CACHE_ROOT}"
export PYTHONPATH="${HARP_EVAL_SITE}:${VLLM_EVAL_SITE}:${SLIME_REPO_ROOT}:${PYTHONPATH:-}"

if ! python3 -c "import math_verify, antlr4, sympy, ziglang" >/dev/null 2>&1; then
  python3 -m pip install --upgrade --target "${HARP_EVAL_SITE}" ${HARP_EVAL_INSTALL_SPECS}
fi
python3 - <<"PY"
import importlib.metadata as metadata
expected = {
    "vllm": "0.24.0",
    "transformers": "5.13.0",
    "math-verify": "0.9.0",
    "latex2sympy2-extended": "1.11.0",
    "antlr4-python3-runtime": "4.13.2",
    "sympy": "1.14.0",
    "ziglang": "0.15.2",
}
for package, wanted in expected.items():
    found = metadata.version(package)
    print(package, found)
    if found != wanted:
        raise SystemExit(f"Dependency mismatch for {package}: expected {wanted}, found {found}")
PY

python3 - <<"PY"
import os
import shlex
import shutil
from pathlib import Path

site = Path(os.environ["HARP_EVAL_SITE"])
zig = site / "ziglang" / "zig"
if not zig.is_file():
    raise SystemExit(f"Missing Zig executable: {zig}")

cc = Path(os.environ["HARP_ZIG_CC"])
cc.parent.mkdir(parents=True, exist_ok=True)
cc.write_text(
    """#!/usr/bin/bash
set -euo pipefail
translated=()
for argument in "$@"; do
  case "${argument}" in
    -l:libcuda.so.1) translated+=("-lcuda") ;;
    *) translated+=("${argument}") ;;
  esac
done
exec %s cc -target x86_64-linux-gnu "${translated[@]}"
"""
    % shlex.quote(str(zig)),
    encoding="utf-8",
)
cc.chmod(0o755)

source = Path(os.environ["HARP_PYTHON_HEADER_SOURCE"])
destination = Path(os.environ["HARP_ZIG_INCLUDE_ROOT"])
shutil.copytree(source / "python3.12", destination / "python3.12", dirs_exist_ok=True)
multiarch = destination / "x86_64-linux-gnu" / "python3.12"
multiarch.mkdir(parents=True, exist_ok=True)
shutil.copy2(source / "x86_64-linux-gnu" / "python3.12" / "pyconfig.h", multiarch / "pyconfig.h")

if not os.access(cc, os.X_OK):
    raise SystemExit(f"Zig C compiler wrapper is not executable: {cc}")
if not (destination / "python3.12" / "Python.h").is_file() or not (multiarch / "pyconfig.h").is_file():
    raise SystemExit("Staged Python headers are incomplete")
PY

zig_probe_output="${TMPDIR}/harp_zig_libcuda_probe.so"
printf "%s\n" \
  "#include <stdlib.h>" \
  "extern int cuInit(unsigned int);" \
  "void *harp_zig_probe(void) {" \
  "  (void)cuInit(0);" \
  "  return malloc(1);" \
  "}" | "${HARP_ZIG_CC}" \
  -x c - \
  -O3 \
  -shared \
  -fPIC \
  -o "${zig_probe_output}" \
  -l:libcuda.so.1 \
  -L/.singularity.d/libs
test -s "${zig_probe_output}"
python3 - "${zig_probe_output}" <<"PY"
import ctypes
import sys

ctypes.CDLL(sys.argv[1])
print(f"Loaded Zig CUDA/libc probe: {sys.argv[1]}")
PY

python3 examples/qwen3_4b_opd_harp/prepare_experiment.py \
  --harp-source "${HARP_SOURCE_CACHE}" \
  --cached-harp-source "${HARP_EXISTING_SOURCE}" \
  --harp-output "${HARP_JSONL}" \
  --opd-source-1k "${SOURCE_OPD_1K_JSONL}" \
  --opd-source-4k "${SOURCE_OPD_4K_JSONL}" \
  --opd-output-1k "${CLEANED_OPD_1K_JSONL}" \
  --opd-output-4k "${CLEANED_OPD_4K_JSONL}" \
  --metadata "${HARP_PREP_METADATA}"

if [[ ! -f "${STUDENT_HF_DIR}/config.json" ]]; then
  hf download "${STUDENT_HF_REPO}" --local-dir "${STUDENT_HF_DIR}"
fi
if [[ ! -f "${TEACHER_HF_DIR}/config.json" ]]; then
  hf download "${TEACHER_HF_REPO}" --local-dir "${TEACHER_HF_DIR}"
fi

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

test -f "${STUDENT_TORCH_DIST_DIR}/latest_checkpointed_iteration.txt"
test -f "${HARP_PREP_METADATA}"
test -f "${HARP_JSONL}"
test -x "${HARP_ZIG_CC}"
test -f "${HARP_ZIG_INCLUDE_ROOT}/python3.12/Python.h"
test -f "${HARP_ZIG_INCLUDE_ROOT}/x86_64-linux-gnu/python3.12/pyconfig.h"
'

echo "Preparation complete at $(date)"
