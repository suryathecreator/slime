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
