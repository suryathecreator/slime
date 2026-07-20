from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_job_respects_four_gpu_cpu_limit_and_short_tmpdir() -> None:
    for path in sorted(ROOT.glob("*.sbatch")):
        text = path.read_text()
        assert "#SBATCH --gres=gpu:h200:4" in text
        cpus = re.search(r"^#SBATCH --cpus-per-task=(\d+)$", text, re.M)
        assert cpus is not None
        assert int(cpus.group(1)) <= 32
        wall = re.search(r"^#SBATCH --time=(\d+):(\d+):(\d+)$", text, re.M)
        assert wall is not None
        hours, minutes, seconds = map(int, wall.groups())
        assert hours * 3600 + minutes * 60 + seconds <= 86_400
        assert 'export TMPDIR="/tmp/${USER:-suryadv}/q8b32b-${SLURM_JOB_ID}"' in text
        assert "unset LD_LIBRARY_PATH" in text


def test_long_training_stages_self_requeue_before_qos_limit() -> None:
    for name in ("03_sft.sbatch", "04_opd.sbatch"):
        text = (ROOT / name).read_text()
        assert "#SBATCH --signal=B:USR1@600" in text
        assert "scontrol requeue" in text


def test_submission_is_fail_closed_and_cancels_partial_chain() -> None:
    text = (ROOT / "submit_chain.sh").read_text()
    assert "if ! raw=" in text
    assert "trap cancel_partial_submission EXIT" in text
    assert 'scancel "${submitted_jobs[@]}"' in text
    assert "submission.failed.json" in text
    assert 'raw="$(sbatch' not in text


def test_base_eval_recovery_is_fail_closed_and_records_superseded_jobs() -> None:
    text = (ROOT / "resubmit_from_base_eval.sh").read_text()
    assert 'audit.get("validation_gate", {}).get("passed")' in text
    assert 'scancel "${superseded_pending_jobs[@]}"' in text
    assert "trap cancel_partial_resubmission EXIT" in text
    assert 'scancel "${submitted_jobs[@]}"' in text
    assert 'RECOVERY_MANIFEST_STEM:-resubmission_after_base_eval' in text
    assert "RECOVERY_SOURCE_MANIFEST" in text
    assert "RECOVERY_MANIFEST_STEM" in text
    assert text.count("submit_job ") == 6


def test_sft_recovery_is_fail_closed_and_reuses_completed_evals() -> None:
    text = (ROOT / "resubmit_from_sft.sh").read_text()
    assert 'RECOVERY_MANIFEST_STEM:-resubmission_after_sft_oom' in text
    assert 'resubmission_after_cuda_visibility.json' in text
    assert 'scancel "${superseded_pending_jobs[@]}"' in text
    assert "trap cancel_partial_resubmission EXIT" in text
    assert 'scancel "${submitted_jobs[@]}"' in text
    assert 'failed_rollout_archive' in text
    assert text.count("submit_job ") == 4
    assert "q8b32b-eval-base" not in text
    assert "q8b32b-eval-teacher" not in text


def test_sft_uses_16k_dynamic_packing_with_32k_native_context() -> None:
    env = (ROOT / "env.sh").read_text()
    assert "export SFT_MAX_TOKENS_PER_GPU=16384" in env
    assert "export SFT_SEQ_LENGTH=32768" in env

    config = json.loads((ROOT / "config/sft_config.json").read_text())
    assert config["dynamic_packing_max_tokens_per_gpu"] == 16384
    assert config["native_context_length"] == 32768
    assert "never truncate" in config["overlength_policy"]

    validator = (ROOT / "validate_config.py").read_text()
    assert 'os.environ.get("SFT_MAX_TOKENS_PER_GPU", "0")' in validator
    assert 'sft["dynamic_packing_max_tokens_per_gpu"]' in validator

    wrapper = (ROOT.parent / "qwen3_8b_opd_tillicum" / "04_run_sft_100k_8xh200.sbatch").read_text()
    assert '--max-tokens-per-gpu "${SFT_MAX_TOKENS_PER_GPU}"' in wrapper


def test_sft_propagates_pinned_compiler_to_ray_workers() -> None:
    env = (ROOT / "env.sh").read_text()
    assert 'SFT_TORCH_COMPILE_CC="${SFT_TORCH_COMPILE_CC:-${EVAL_ZIG_CC}}"' in env
    assert "SFT_TORCH_COMPILE_CPATH=" in env

    sft_job = (ROOT / "03_sft.sbatch").read_text()
    assert '"${SFT_TORCH_COMPILE_CC}"' in sft_job
    assert '"${EVAL_ZIG_INCLUDE_ROOT}/python3.12/Python.h"' in sft_job

    wrapper = (ROOT.parent / "qwen3_8b_opd_tillicum" / "04_run_sft_100k_8xh200.sbatch").read_text()
    assert 'export CC="${SFT_TORCH_COMPILE_CC}"' in wrapper
    assert 'export CPATH="${SFT_TORCH_COMPILE_CPATH}"' in wrapper
    assert '\\"CC\\": \\"${CC}\\"' in wrapper
    assert '\\"CPATH\\": \\"${CPATH}\\"' in wrapper

    container = (ROOT.parent / "qwen3_8b_opd_tillicum" / "container_exec.sh").read_text()
    assert "  SFT_TORCH_COMPILE_CC\n" in container
    assert "  SFT_TORCH_COMPILE_CPATH\n" in container


def test_eval_disables_training_site_patch_and_uses_explicit_libraries() -> None:
    text = (ROOT / "02_eval_math500.sbatch").read_text()
    assert "export SLIME_VLLM_PATCH_SITE=0" in text
    assert "export SLIME_SGLANG_PATCH_SITE=0" in text
    assert 'export LD_LIBRARY_PATH="${VLLM_TARGET_LIBS}"' in text
    env = (ROOT / "env.sh").read_text()
    assert 'SLIME_SGLANG_PATCH_SITE="${SLIME_SGLANG_PATCH_SITE:-1}"' in env


def test_eval_configures_and_preflights_pinned_zig_compiler() -> None:
    env = (ROOT / "env.sh").read_text()
    assert 'EVAL_ZIG_CC="${EVAL_ZIG_CC:-${MATH_VERIFY_SITE}/bin/zig-cc}"' in env
    assert 'EVAL_ZIG_INCLUDE_ROOT="${EVAL_ZIG_INCLUDE_ROOT:-${MATH_VERIFY_SITE}/python_include}"' in env

    eval_job = (ROOT / "02_eval_math500.sbatch").read_text()
    assert 'export CC="${EVAL_ZIG_CC}"' in eval_job
    assert 'export CPATH="${EVAL_ZIG_INCLUDE_ROOT}/python3.12:${EVAL_ZIG_INCLUDE_ROOT}' in eval_job
    assert 'test -x "${CC}"' in eval_job
    assert 'EVAL_COMPILER_READY cc=${CC}' in eval_job
    assert 'EVAL_SHARD_GPU_READY shard=${expected_visible_gpu}' in eval_job

    preflight = (ROOT / "00_preflight.sbatch").read_text()
    assert '"${EVAL_ZIG_CC}"' in preflight
    assert '"${EVAL_ZIG_INCLUDE_ROOT}/python3.12/Python.h"' in preflight

    wrapper = (ROOT.parent / "qwen3_8b_opd_tillicum" / "container_exec.sh").read_text()
    assert "  EVAL_ZIG_CC\n" in wrapper
    assert "  EVAL_ZIG_INCLUDE_ROOT\n" in wrapper
    assert "  CUDA_VISIBLE_DEVICES\n" in wrapper


def test_environment_capture_does_not_initialize_unused_cudnn() -> None:
    text = (ROOT / "capture_environment.py").read_text()
    assert "torch.backends.cudnn.version()" not in text
    assert 'optional_package("nvidia-cudnn-cu12")' in text
