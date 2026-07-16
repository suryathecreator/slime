#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

bash -n examples/qwen3_4b_opd_harp/*.sh examples/qwen3_4b_opd_harp/*.sbatch
python3 -m py_compile examples/qwen3_4b_opd_harp/*.py
if rg -n 'AsyncLLM|asyncio' examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py; then
  echo "The HARP evaluator contains a prohibited async generation path." >&2
  exit 1
fi
test "$(rg -c 'engine\.generate\(' examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py)" = "1"
rg -q 'engine\.generate\(\[item\["rendered_prompt"\] for item in prepared\]' examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py
rg -q 'STUDENT_HF_REPO="Qwen/Qwen3-4B"' examples/qwen3_4b_opd_harp/env.sh
rg -q 'OPD_STAGE=1k' examples/qwen3_4b_opd_harp/submit_chain.sh
rg -q 'OPD_STAGE=5k' examples/qwen3_4b_opd_harp/submit_chain.sh
rg -q 'REPORT_MODE=base' examples/qwen3_4b_opd_harp/submit_chain.sh
rg -q 'qwen3_harp_boxed_v1' examples/qwen3_4b_opd_harp/prompt_contract.py
test "$(sha256sum examples/qwen3_4b_opd_harp/harp_official/latex_answer_check.py | awk '{print $1}')" = "54b191b9ec51be29538e881d2b0abd64cb176bdb822511a8d4a1694faedf1af0"
test "$(sha256sum examples/qwen3_4b_opd_harp/harp_official/parsing_lib.py | awk '{print $1}')" = "5ca6f58beddabb87dbdbfb824cd8a4fc5c275087261df50725d27290ce42a9d0"
echo "Qwen3-4B/HARP V2 dry checks passed."
