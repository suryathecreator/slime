from __future__ import annotations

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
    assert "resubmission_after_base_eval.json" in text
    assert text.count("submit_job ") == 6


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

    preflight = (ROOT / "00_preflight.sbatch").read_text()
    assert '"${EVAL_ZIG_CC}"' in preflight
    assert '"${EVAL_ZIG_INCLUDE_ROOT}/python3.12/Python.h"' in preflight

    wrapper = (ROOT.parent / "qwen3_8b_opd_tillicum" / "container_exec.sh").read_text()
    assert "  EVAL_ZIG_CC\n" in wrapper
    assert "  EVAL_ZIG_INCLUDE_ROOT\n" in wrapper


def test_environment_capture_does_not_initialize_unused_cudnn() -> None:
    text = (ROOT / "capture_environment.py").read_text()
    assert "torch.backends.cudnn.version()" not in text
    assert 'optional_package("nvidia-cudnn-cu12")' in text
