#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

bash -n examples/qwen3_1_7b_opd_aime2026/*.sh examples/qwen3_1_7b_opd_aime2026/*.sbatch
python3 -m py_compile examples/qwen3_1_7b_opd_aime2026/*.py
rg -q 'SCORER_VERSION = "aime_answer_v1"' examples/qwen3_1_7b_opd_aime2026/aime_answer_v1.py
rg -q 'SCORER_VERSION = "aime_answer_v2"' examples/qwen3_1_7b_opd_aime2026/aime_answer_v2.py
rg -q '^from aime_answer_v2 import' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
rg -q '^from aime_answer_v2 import' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_once_vllm.py
rg -q '^from aime_answer_v2 import' examples/qwen3_1_7b_opd_aime2026/report_aime.py
if rg -n '^from aime_answer_v1 import' \
  examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py \
  examples/qwen3_1_7b_opd_aime2026/evaluate_aime_once_vllm.py \
  examples/qwen3_1_7b_opd_aime2026/report_aime.py; then
  echo "A production AIME entrypoint still imports scorer v1." >&2
  exit 1
fi
rg -q 'NUM_GENERATIONS = NUM_PROBLEMS \* SAMPLES_PER_PROBLEM' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
rg -q 'SAMPLES_PER_PROBLEM = 16' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
rg -q 'TEMPERATURE = 0.6' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
rg -q 'TOP_P = 0.95' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
rg -q 'TOP_K = 20' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
rg -q 'MIN_P = 0.0' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
rg -q 'BASE_SEED = 42' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
rg -Fq 'export LD_LIBRARY_PATH="$(find "${VLLM_EVAL_SITE}" -path "*/nvidia/*/lib"' examples/qwen3_1_7b_opd_aime2026/01_eval_aime_vllm.sbatch
rg -Fq 'export CC="${AIME_ZIG_CC}"' examples/qwen3_1_7b_opd_aime2026/01_eval_aime_vllm.sbatch
rg -Fq 'export CPATH="${AIME_ZIG_INCLUDE_ROOT}/python3.12:${AIME_ZIG_INCLUDE_ROOT}' examples/qwen3_1_7b_opd_aime2026/01_eval_aime_vllm.sbatch
rg -Fq 'export CC="${AIME_ZIG_CC}"' examples/qwen3_1_7b_opd_aime2026/05_run_opd_remaining_2k.sbatch
rg -Fq 'export CPATH="${AIME_ZIG_INCLUDE_ROOT}/python3.12:${AIME_ZIG_INCLUDE_ROOT}' examples/qwen3_1_7b_opd_aime2026/05_run_opd_remaining_2k.sbatch
rg -Fq 'test -x "${CC}"' examples/qwen3_1_7b_opd_aime2026/05_run_opd_remaining_2k.sbatch
rg -q 'OPD_ACTOR_GPUS=2' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'OPD_ROLLOUT_GPUS=3' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'OPD_GLOBAL_BATCH_SIZE=128' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'OPD_GLOBAL_BATCH_SIZE % data_parallel_size' examples/qwen3_1_7b_opd_aime2026/05_run_opd_remaining_2k.sbatch
rg -q 'samples_per_rollout % OPD_GLOBAL_BATCH_SIZE' examples/qwen3_1_7b_opd_aime2026/05_run_opd_remaining_2k.sbatch
rg -q 'VLLM_IMPORT_PREFLIGHT_OK' examples/qwen3_1_7b_opd_aime2026/01_eval_aime_vllm.sbatch
python3 - <<'PY'
from pathlib import Path

path = Path("examples/qwen3_1_7b_opd_aime2026/01_eval_aime_vllm.sbatch")
text = path.read_text(encoding="utf-8")
payload = text.split('container_exec.sh" bash -lc \'\n', 1)[1].split("\n'\n", 1)[0]
if "'" in payload:
    raise SystemExit("AIME eval container payload contains a nested single quote")
PY
test "$(rg -c 'engine\.generate\(' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py)" = "1"
rg -q 'engine\.generate\(\[item\["rendered_prompt"\] for item in prepared\]' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py
if rg -n 'AsyncLLM|asyncio' examples/qwen3_1_7b_opd_aime2026/evaluate_aime_vllm.py; then
  echo "The AIME evaluator contains a prohibited async generation path." >&2
  exit 1
fi
rg -q 'STUDENT_HF_REPO="Qwen/Qwen3-1.7B"' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'TEACHER_HF_REPO="Qwen/Qwen3-8B"' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'AIME_DATASET_REPO="MathArena/aime_2026"' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'AIME_DATASET_REVISION="d2de22f3c656b4f56cf8981212186377d1e23bc3"' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'AIME_PARQUET_SHA256 = "d91db799651b4cc1f0734f52792a695c9cc60dac342524b3d8e5b2ff31c3e957"' examples/qwen3_1_7b_opd_aime2026/prepare_experiment.py
rg -q 'AIME_NORMALIZED_SHA256 = "6e07e802a416526fe52559216e6e3f13f35db1c169443a39c7ae2790c8b5ca2b"' examples/qwen3_1_7b_opd_aime2026/prepare_experiment.py
rg -q 'VLLM_EVAL_MAX_MODEL_LEN=32768' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'VLLM_EVAL_MAX_RESPONSE_LEN=31744' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'VLLM_EVAL_STUDENT_MAX_NUM_SEQS=34' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'VLLM_EVAL_TEACHER_MAX_NUM_SEQS=21' examples/qwen3_1_7b_opd_aime2026/env.sh
rg -q 'OPD_STAGE=1k' examples/qwen3_1_7b_opd_aime2026/submit_chain.sh
rg -q 'OPD_STAGE=5k' examples/qwen3_1_7b_opd_aime2026/submit_chain.sh
rg -q 'REPORT_MODE=base' examples/qwen3_1_7b_opd_aime2026/submit_chain.sh
rg -q 'REPORT_MODE=opd1k' examples/qwen3_1_7b_opd_aime2026/submit_chain.sh
test "$(rg -c -- '--dependency="afterok:' examples/qwen3_1_7b_opd_aime2026/submit_chain.sh)" = "9"
for entrypoint in examples/qwen3_1_7b_opd_aime2026/*.sbatch; do
  rg -q 'SLURM_SUBMIT_DIR.*examples/qwen3_1_7b_opd_aime2026/env\.sh' "${entrypoint}"
done
rg -q '^  AIME_EXAMPLE_DIR$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q '^  STUDENT_HF_REVISION$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q '^  TEACHER_HF_REVISION$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q '^  CC$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q '^  CPATH$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q '^  SLURM_JOB_ID$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q '^  SLURM_JOB_NAME$' examples/qwen3_8b_opd_tillicum/container_exec.sh
for forwarded in \
  AIME_EXAMPLE_DIR \
  AIME_JSONL \
  AIME_ZIG_CC \
  AIME_ZIG_INCLUDE_ROOT \
  EVAL_MODEL_DIR \
  EVAL_TOKENIZER_DIR \
  EVAL_STAGE_DIR \
  EVAL_MAX_NUM_SEQS \
  VLLM_EVAL_SITE \
  VLLM_EVAL_MAX_MODEL_LEN \
  VLLM_EVAL_MAX_RESPONSE_LEN \
  VLLM_EVAL_MAX_NUM_BATCHED_TOKENS \
  VLLM_EVAL_GPU_MEMORY_UTILIZATION \
  VLLM_EVAL_ENGINE_READY_TIMEOUT_SECONDS \
  VLLM_EVAL_PREFER_CUDA_GRAPH; do
  rg -q "^  ${forwarded}$" examples/qwen3_8b_opd_tillicum/container_exec.sh
done
rg -q -- '--completed-teacher-job' examples/qwen3_1_7b_opd_aime2026/submit_remaining_2k_chain.sh
test "$(rg -c -- '--dependency="afterok:' examples/qwen3_1_7b_opd_aime2026/submit_remaining_2k_chain.sh)" = "4"
rg -q 'max_concurrent_gpu_count=4' examples/qwen3_1_7b_opd_aime2026/submit_remaining_2k_chain.sh
(
  source examples/qwen3_1_7b_opd_aime2026/env.sh
  model_parallel_size=$((OPD_TENSOR_MODEL_PARALLEL_SIZE * OPD_CONTEXT_PARALLEL_SIZE))
  test $((OPD_ACTOR_GPUS / model_parallel_size)) = 2
  test $((OPD_GLOBAL_BATCH_SIZE % (OPD_ACTOR_GPUS / model_parallel_size))) = 0
  test $(((OPD_ROLLOUT_BATCH_SIZE * OPD_N_SAMPLES_PER_PROMPT) % OPD_GLOBAL_BATCH_SIZE)) = 0
  test $((16 * OPD_ROLLOUT_BATCH_SIZE)) = 2048
  test $((40 * OPD_ROLLOUT_BATCH_SIZE)) = 5120
)
echo "Qwen3-1.7B/Qwen3-8B AIME 2026 dry checks passed."
