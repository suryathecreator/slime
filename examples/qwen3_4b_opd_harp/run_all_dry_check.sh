#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

bash -n examples/qwen3_4b_opd_harp/*.sh examples/qwen3_4b_opd_harp/*.sbatch
python3 -m py_compile examples/qwen3_4b_opd_harp/*.py
test "$(sha256sum examples/qwen3_4b_opd_harp/harp_answer_v2.py | awk '{print $1}')" = "76524e2e5a6659ccb354928ebc014b5c3fecf17bedff38c5e2610e0d659beae9"
rg -q 'SCORER_VERSION = "harp_answer_v3"' examples/qwen3_4b_opd_harp/harp_answer_v3.py
rg -q '"needs_review"' examples/qwen3_4b_opd_harp/harp_answer_v3.py
rg -q -- '--scorer-version' examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py
rg -q -- '--exclusion-manifest' examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py
rg -q '"exclusions": \[\]' examples/qwen3_4b_opd_harp/harp_exclusions_v3.json
for entrypoint in examples/qwen3_4b_opd_harp/*.sbatch; do
  rg -q 'SLURM_SUBMIT_DIR.*examples/qwen3_4b_opd_harp/env\.sh' "${entrypoint}"
  rg -q 'SCRIPT_DIR="\$\{SLURM_SUBMIT_DIR\}/examples/qwen3_4b_opd_harp"' "${entrypoint}"
done
if rg -n 'export LD_LIBRARY_PATH=.*VLLM_EVAL_SITE' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch; then
  echo "Model conversion setup globally overlays vLLM CUDA libraries." >&2
  exit 1
fi
rg -Fq 'PYTHONPATH="/root/Megatron-LM:${SLIME_REPO_ROOT}"' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
if rg -n 'AsyncLLM|asyncio' examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py; then
  echo "The HARP evaluator contains a prohibited async generation path." >&2
  exit 1
fi
test "$(rg -c 'engine\.generate\(' examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py)" = "1"
rg -q 'engine\.generate\(\[item\["rendered_prompt"\] for item in prepared\]' examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py
rg -q 'STUDENT_HF_REPO="Qwen/Qwen3-4B"' examples/qwen3_4b_opd_harp/env.sh
rg -Fq 'SLURM_MEM_SETUP="${SLURM_MEM_SETUP:-64G}"' examples/qwen3_4b_opd_harp/env.sh
rg -Fq 'SLURM_MEM_EVAL_4B="${SLURM_MEM_EVAL_4B:-96G}"' examples/qwen3_4b_opd_harp/env.sh
rg -Fq 'SLURM_MEM_EVAL_32B="${SLURM_MEM_EVAL_32B:-384G}"' examples/qwen3_4b_opd_harp/env.sh
rg -Fq 'SLURM_MEM_OPD="${SLURM_MEM_OPD:-512G}"' examples/qwen3_4b_opd_harp/env.sh
rg -Fq 'SLURM_MEM_REPORT="${SLURM_MEM_REPORT:-16G}"' examples/qwen3_4b_opd_harp/env.sh
! rg -n '^#SBATCH --mem=0$' examples/qwen3_4b_opd_harp/*.sbatch
rg -q '^#SBATCH --mem=64G$' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -q '^#SBATCH --mem=384G$' examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch
rg -q '^#SBATCH --mem=512G$' examples/qwen3_4b_opd_harp/02_tune_opd.sbatch examples/qwen3_4b_opd_harp/03_run_opd.sbatch
rg -q '^#SBATCH --mem=16G$' examples/qwen3_4b_opd_harp/04_report_harp.sbatch
rg -Fq -- '--mem="${SLURM_MEM_SETUP}"' examples/qwen3_4b_opd_harp/submit_chain.sh
test "$(rg -Fc -- '--mem="${SLURM_MEM_EVAL_4B}"' examples/qwen3_4b_opd_harp/submit_chain.sh)" = "3"
rg -Fq -- '--mem="${SLURM_MEM_EVAL_32B}"' examples/qwen3_4b_opd_harp/submit_chain.sh
test "$(rg -Fc -- '--mem="${SLURM_MEM_OPD}"' examples/qwen3_4b_opd_harp/submit_chain.sh)" = "2"
test "$(rg -Fc -- '--mem="${SLURM_MEM_REPORT}"' examples/qwen3_4b_opd_harp/submit_chain.sh)" = "3"
rg -q 'OPD_STAGE=1k' examples/qwen3_4b_opd_harp/submit_chain.sh
rg -q 'OPD_STAGE=5k' examples/qwen3_4b_opd_harp/submit_chain.sh
rg -q 'REPORT_MODE=base' examples/qwen3_4b_opd_harp/submit_chain.sh
rg -q 'REPORT_MODE=opd1k' examples/qwen3_4b_opd_harp/submit_chain.sh
rg -q 'VLLM_EVAL_4B_MAX_NUM_SEQS=20' examples/qwen3_4b_opd_harp/env.sh
rg -q 'VLLM_EVAL_32B_MAX_NUM_SEQS=6' examples/qwen3_4b_opd_harp/env.sh
rg -q 'ziglang==0.15.2' examples/qwen3_4b_opd_harp/env.sh
rg -q 'HARP_ZIG_CC=' examples/qwen3_4b_opd_harp/env.sh
rg -q 'HARP_ZIG_INCLUDE_ROOT=' examples/qwen3_4b_opd_harp/env.sh
rg -q 'ziglang": "0.15.2"' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -Fq 'case "${argument}" in' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -Fq -- '-l:libcuda.so.1) translated+=("-lcuda") ;;' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -Fq 'exec %s cc -target x86_64-linux-gnu "${translated[@]}"' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -Fq 'zig_probe_output="${TMPDIR}/harp_zig_libcuda_probe.so"' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -Fq 'extern int cuInit(unsigned int);' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -Fq -- '-L/.singularity.d/libs' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -Fq 'test -s "${zig_probe_output}"' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -Fq 'ctypes.CDLL(sys.argv[1])' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -Fq 'test -x "${HARP_ZIG_CC}"' examples/qwen3_4b_opd_harp/00_prepare_and_convert.sbatch
rg -q '^  CC$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q '^  CPATH$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q 'OPD_CONTEXT_PARALLEL_SIZE=1' examples/qwen3_4b_opd_harp/env.sh
rg -q 'OPD_MAX_TOKENS_PER_GPU=8192' examples/qwen3_4b_opd_harp/env.sh
rg -q 'OPD_LOG_PROBS_CHUNK_SIZE=2048' examples/qwen3_4b_opd_harp/env.sh
rg -q 'run_eval_attempt cuda_graph 0' examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch
rg -q 'run_eval_attempt eager 1' examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch
rg -Fq 'epoch=${EPOCHSECONDS}' examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch
! rg -Fq 'rm -f "${ready}"' examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch
rg -Fq -- '--output-dir "${EVAL_STAGE_DIR%/*}"' examples/qwen3_4b_opd_harp/01_eval_harp_vllm.sbatch
rg -q 'cudagraph_capture_sizes' examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py
rg -q 'CUDA graphs were requested, but vLLM disabled them' examples/qwen3_4b_opd_harp/evaluate_harp_vllm.py
rg -q '^  EVAL_MAX_NUM_SEQS$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q '^  EVAL_PREFER_CUDA_GRAPH$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q '^  OPD1K_REPORT_DIR$' examples/qwen3_8b_opd_tillicum/container_exec.sh
rg -q '^  VLLM_EVAL_ENGINE_READY_TIMEOUT_SECONDS$' examples/qwen3_8b_opd_tillicum/container_exec.sh
if rg -n -- '--job-name=[^ ]*tune' examples/qwen3_4b_opd_harp/submit_chain.sh; then
  echo "The production chain still submits a tuning job." >&2
  exit 1
fi
rg -q 'qwen3_harp_boxed_v1' examples/qwen3_4b_opd_harp/prompt_contract.py
test "$(sha256sum examples/qwen3_4b_opd_harp/harp_official/latex_answer_check.py | awk '{print $1}')" = "54b191b9ec51be29538e881d2b0abd64cb176bdb822511a8d4a1694faedf1af0"
test "$(sha256sum examples/qwen3_4b_opd_harp/harp_official/parsing_lib.py | awk '{print $1}')" = "5ca6f58beddabb87dbdbfb824cd8a4fc5c275087261df50725d27290ce42a9d0"
echo "Qwen3-4B/HARP V2 dry checks passed."
