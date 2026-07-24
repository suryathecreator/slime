from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_every_job_respects_gpu_cpu_limit_and_short_tmpdir() -> None:
    for path in sorted(ROOT.glob("*.sbatch")):
        text = path.read_text()
        expected_gpus = 1 if path.name == "06_rescore_math500.sbatch" else 4
        assert f"#SBATCH --gres=gpu:h200:{expected_gpus}" in text
        cpus = re.search(r"^#SBATCH --cpus-per-task=(\d+)$", text, re.M)
        assert cpus is not None
        assert int(cpus.group(1)) <= 8 * expected_gpus
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


def test_scorer_v3_resubmit_is_one_gpu_fail_closed_and_remote_matched() -> None:
    submitter = (ROOT / "submit_rescore.sh").read_text()
    assert "git diff --quiet && git diff --cached --quiet" in submitter
    assert 'git rev-parse "origin/${expected_branch}"' in submitter
    assert '[[ "${head_sha}" == "${remote_sha}" ]]' in submitter
    assert submitter.count("sbatch --parsable") == 1
    assert "scorer_v3_${mode}_submission.json" in submitter
    assert 'scancel "${job_id}"' in submitter
    assert "RESCORE_MODE=${mode}" in submitter
    assert "EXPECTED_LAUNCH_COMMIT=${head_sha}" in submitter
    assert "MATH500_SCORER_V3_REVIEW.json" in submitter

    job = (ROOT / "06_rescore_math500.sbatch").read_text()
    assert "#SBATCH --gres=gpu:h200:1" in job
    assert "rescore_math500.py" in job
    assert "--v1-eval-root" in job
    assert "--v2-eval-root" in job
    assert "--scorer-audit" in job
    assert "--mode \"${RESCORE_MODE}\"" in job
    assert "--review-manifest" in job


def test_rescore_has_exact_audit_gate_and_preserves_v1_v2() -> None:
    scorer_v1 = (ROOT / "math500_scorer_v1.py").read_text()
    scorer_v2 = (ROOT / "math500_scorer_v2.py").read_text()
    scorer_v3 = (ROOT / "math500_scorer.py").read_text()
    driver = (ROOT / "rescore_math500.py").read_text()
    assert 'SCORER_VERSION = "math500_event_scorer_v1"' in scorer_v1
    assert 'SCORER_VERSION = "math500_event_scorer_v2"' in scorer_v2
    assert 'SCORER_VERSION = "math500_strict_boxed_scorer_v3"' in scorer_v3
    assert "EXPECTED_TAG_COUNTS" in driver
    assert "suggested_expected_metrics.json" in driver
    assert "Final aggregate/decision gate mismatch" in driver
    assert "Final review-manifest gate failed" in driver
    assert '"review_count": int(review["unresolved_review_count"]) if review else None' in driver


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
    assert 'SFT_TORCH_COMPILE_CPATH="${SFT_TORCH_COMPILE_CPATH:-${EVAL_ZIG_INCLUDE_ROOT}/python3.12:${EVAL_ZIG_INCLUDE_ROOT}}"' in env

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


def test_opd_propagates_and_preflights_pinned_compiler() -> None:
    env = (ROOT / "env.sh").read_text()
    assert 'OPD_TORCH_COMPILE_CC="${OPD_TORCH_COMPILE_CC:-${EVAL_ZIG_CC}}"' in env
    assert 'OPD_TORCH_COMPILE_CPATH="${OPD_TORCH_COMPILE_CPATH:-${EVAL_ZIG_INCLUDE_ROOT}/python3.12:${EVAL_ZIG_INCLUDE_ROOT}}"' in env

    opd_job = (ROOT / "04_opd.sbatch").read_text()
    assert 'export CC="${OPD_TORCH_COMPILE_CC}"' in opd_job
    assert 'export CPATH="${OPD_TORCH_COMPILE_CPATH}' in opd_job
    assert "from triton.backends.nvidia.driver import CudaUtils" in opd_job
    assert "OPD_TRITON_COMPILER_READY" in opd_job

    wrapper = (ROOT.parent / "qwen3_8b_opd_tillicum" / "05_run_opd_50k_8xh200.sbatch").read_text()
    assert 'OPD_TORCH_COMPILER_READY cc=${CC} cpath=${CPATH}' in wrapper
    assert '\\"CC\\": \\"${CC}\\"' in wrapper
    assert '\\"CPATH\\": \\"${CPATH}\\"' in wrapper


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


def test_cap_hit_proxy_is_exact_fail_closed_and_serial() -> None:
    config = json.loads((ROOT / "config/eval_config.json").read_text())
    proxy = config["cap_hit_context_proxy"]
    assert proxy["target_total_context_tokens"] == 32768
    assert proxy["status"] == "abandoned_incomplete_no_score"
    assert "exact prefix" in proxy["publication_gate"]
    assert proxy["cancelled_jobs"] == [183040, 183041, 183042]

    job = (ROOT / "05_eval_cap_hits_32k.sbatch").read_text()
    assert "cap-rerun-shard" in job
    assert "aggregate-cap-proxy" in job
    assert '--target-context "${EVAL_CAP_PROXY_TARGET_CONTEXT}"' in job
    assert 'CAP_PROXY_SHARD_GPU_READY shard=${expected_visible_gpu}' in job

    recovery = (ROOT / "resubmit_after_opd_compiler.sh").read_text()
    assert 'scancel' not in recovery or '"${SCANCEL_BIN}" 181914' in recovery
    assert "trap cancel_partial_resubmission EXIT" in recovery
    assert '"${SCANCEL_BIN}" "${submitted_jobs[@]}"' in recovery
    assert recovery.count("submit_job ") == 6
    ordered_names = [
        "submit_job opd_job",
        "submit_job opd_eval_job",
        "submit_job opd_proxy_job",
        "submit_job sft_proxy_job",
        "submit_job teacher_proxy_job",
        "submit_job base_proxy_job",
    ]
    offsets = [recovery.index(name) for name in ordered_names]
    assert offsets == sorted(offsets)


def test_environment_capture_does_not_initialize_unused_cudnn() -> None:
    text = (ROOT / "capture_environment.py").read_text()
    assert "torch.backends.cudnn.version()" not in text
    assert 'optional_package("nvidia-cudnn-cu12")' in text
