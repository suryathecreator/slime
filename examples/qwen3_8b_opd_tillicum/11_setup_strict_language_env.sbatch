#!/bin/bash
#SBATCH --job-name=slime-qwen3-strict-lang
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

mkdir -p "${SLURM_LOG_DIR}" "${STRICT_LANGUAGE_SITE}"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  exec >"${SLURM_LOG_DIR}/${SLURM_JOB_NAME}_${SLURM_JOB_ID}.log" 2>&1
fi

echo "Installing ${STRICT_LANGUAGE_INSTALL_SPEC} into ${STRICT_LANGUAGE_SITE}"
"${SCRIPT_DIR}/container_exec.sh" bash -lc '
set -euo pipefail
export PYTHONPATH="${STRICT_LANGUAGE_SITE}:${PYTHONPATH:-}"
if ! python3 -c "import lingua; import importlib.metadata as m; assert m.version(\"lingua-language-detector\") == \"2.2.0\""; then
  python3 -m pip install --target "${STRICT_LANGUAGE_SITE}" "${STRICT_LANGUAGE_INSTALL_SPEC}"
fi
python3 -c "from lingua import LanguageDetectorBuilder; import importlib.metadata as m; print(m.version(\"lingua-language-detector\")); LanguageDetectorBuilder.from_all_languages().with_low_accuracy_mode().build()"
'
echo "Strict language environment ready at $(date)"
