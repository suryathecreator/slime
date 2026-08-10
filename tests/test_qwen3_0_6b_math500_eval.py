from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
HANDOFF = REPO / "examples/qwen3_0_6b_openr1_math220k_masked_sft_eval_handoff"
MANIFEST = json.loads((HANDOFF / "available_eval_manifest.json").read_text())
POLICY = json.loads((HANDOFF / "eval_policy.json").read_text())


def test_inventory_contains_base_and_all_eleven_final_checkpoints() -> None:
    expected = {
        "base_0p6b",
        "correct_only",
        "unmasked",
        "random_mask_25",
        "random_mask_50",
        "random_mask_70",
        "random_mask_80",
        "random_mask_90",
        "inverse_tau_0p20",
        "inverse_tau_0p05",
        "margin_mask",
        "prob_ratio_mask",
    }
    checkpoints = MANIFEST["checkpoints"]
    assert len(checkpoints) == 12
    assert {item["variant"] for item in checkpoints} == expected
    for item in checkpoints:
        assert (REPO / item["model"] / "config.json").is_file()
        assert (REPO / item["manifest"]).is_file()


def test_dataset_runtime_and_sharding_are_frozen() -> None:
    assert MANIFEST["dataset"]["revision"] == (
        "6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be"
    )
    assert MANIFEST["runtime"] == {
        "math_verify": "0.9.0",
        "torch": "2.8.0",
        "transformers": "4.57.6",
        "vllm": "0.10.2",
    }
    assert MANIFEST["sharding"] == {
        "chunk_size": 32,
        "count": 4,
        "layout": "contiguous",
        "problems_per_shard": 125,
    }


def test_policy_is_last_box_only_and_gold_independent() -> None:
    scoring = POLICY["scoring"]
    assert scoring["scorer"] == "math500_last_boxed_symmetric_scorer_v5"
    assert scoring["gold_conditioned_routing"] is False
    assert scoring["non_boxed_fallback"] is None
    assert scoring["cap_hits"] == (
        "score the saved response identically to natural stops"
    )
    assert scoring["accuracy_denominator"] == 500
    assert set(scoring["parser_lanes"]) == {"latex", "expr"}


def test_generation_policy_matches_standard_eval() -> None:
    decoding = POLICY["decoding"]
    assert decoding["temperature"] == 0.0
    assert decoding["n"] == 1
    assert decoding["max_tokens"] == 32768
    assert decoding["max_model_len"] == 32768
    assert decoding["tensor_parallel_size"] == 1
    assert decoding["stop_token_ids"] == [151643, 151645]
    assert POLICY["prompt"]["instruction"] == (
        "Please reason step by step, and put your final answer within \\boxed{}."
    )


def test_h200_jobs_use_minimal_buffered_request_and_requeue() -> None:
    for name in ("canary_h200.sbatch", "run_h200.sbatch"):
        text = (HANDOFF / name).read_text()
        assert "#SBATCH --gpus=h200:1" in text
        assert "#SBATCH --time=02:00:00" in text
        assert "#SBATCH --cpus-per-task=8" in text
        assert "#SBATCH --mem=96G" in text
        assert "#SBATCH --requeue" in text
        assert "#SBATCH --signal=B:USR1@600" in text


def test_submission_shape_is_48_unthrottled_h200_tasks() -> None:
    text = (HANDOFF / "submit.sh").read_text()
    assert "--array=0-3" in text
    assert "MATH500_V5_SLURM_SHAPES_VALID checkpoints=12 shards=48" in text
    assert "--array=0-3%" not in text
    assert "--dependency=\"afterok:$canary_job\"" in text


def test_h200_workers_use_exclusive_persistent_compile_caches() -> None:
    run = (HANDOFF / "run_h200.sbatch").read_text()
    canary = (HANDOFF / "canary_h200.sbatch").read_text()
    assert "tasks/array_${SLURM_ARRAY_JOB_ID:?}/$EVAL_VARIANT/shard_" in run
    assert "tasks/job_${SLURM_JOB_ID:?}/canary" in canary
    for text in (run, canary):
        assert 'export VLLM_CACHE_ROOT="$TASK_CACHE_ROOT/vllm"' in text
        assert 'export TORCHINDUCTOR_CACHE_DIR="$TASK_CACHE_ROOT/torchinductor"' in text
        assert 'export TRITON_CACHE_DIR="$TASK_CACHE_ROOT/triton"' in text
        assert 'export XDG_CACHE_HOME="$TASK_CACHE_ROOT/xdg"' in text
        assert '$TASK_CACHE_ROOT/tmp' not in text
        assert "MATH500_V5_CACHE_NAMESPACE" in text
    assert 'mktemp -d "/tmp/q06m5-${SLURM_JOB_ID:?}.XXXXXX"' in canary
    assert (
        'mktemp -d "/tmp/q06m5-${SLURM_ARRAY_JOB_ID:?}-${SHARD_INDEX}.XXXXXX"'
        in run
    )


def test_vllm_ipc_paths_fit_the_unix_socket_limit() -> None:
    uuid = "x" * 36
    longest_slurm_id = "9" * 20
    canary = f"/tmp/q06m5-{longest_slurm_id}.XXXXXX/{uuid}"
    shard = f"/tmp/q06m5-{longest_slurm_id}-3.XXXXXX/{uuid}"
    assert len(canary) < 107
    assert len(shard) < 107


def test_resubmission_is_guarded_and_archives_prior_attempt() -> None:
    text = (HANDOFF / "submit.sh").read_text()
    assert '"--resubmit"' in text
    assert "torch._dynamo.exc.BackendCompilerFailed" in text
    assert "/q06-math500-v5/vllm/torch_compile_cache/" in text
    assert "zmq.error.ZMQError" in text
    assert "is longer than 107 characters" in text
    assert "zmq_ipc_path_too_long" in text
    assert 'scancel "${ACTIVE_PRIOR_JOBS[@]}"' in text
    assert "rollback_unrecorded_submission" in text
    assert 'scancel "${SUBMITTED_JOBS[@]}"' in text
    assert "submission_attempts/$attempt_stamp" in text
    assert 'mv "$CONTROL_ROOT/canary" "$ATTEMPT_ARCHIVE/canary"' in text
