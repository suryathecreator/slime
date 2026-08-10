from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_eval_handoff import (
    generate_math500 as generation,
)
from examples.qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_eval_handoff import (
    math500_eval as control,
)


NUM_GPUS = 0
ROOT = Path(__file__).resolve().parents[1]
HANDOFF = (
    ROOT
    / "examples/qwen2_5_7b_openr1_math220k_correct_recipe_masked_sft_eval_handoff"
)


def test_inventory_has_ten_generated_and_two_pinned_reference_targets() -> None:
    inventory = control.load_inventory(ROOT)
    assert tuple(item["id"] for item in inventory["checkpoints"]) == control.TARGETS
    assert len(control.TARGETS) == 10
    assert control.REFERENCE_TARGETS == ("base_qwen2_5_7b", "qwen2_5_7b_8k")
    assert len(control.ALL_TARGETS) == 12
    assert inventory["sharding"]["array_tasks_per_checkpoint"] == 12
    assert inventory["contract_hash"] == "07291674d43cf3da"


def test_every_transferred_checkpoint_and_manifest_is_present() -> None:
    inventory = control.load_inventory(ROOT)
    for item in inventory["checkpoints"]:
        model = control.absolute(ROOT, item, "model")
        manifest = control.absolute(ROOT, item, "manifest")
        assert model.is_dir()
        assert manifest.is_file()
        assert len(list(model.glob("model-*-of-00004.safetensors"))) == 4
        assert control.load_json(manifest)["checkpoint_manifest_sha256"]


def test_reference_results_are_hash_pinned_and_exact_policy_compatible() -> None:
    inventory = control.load_inventory(ROOT)
    references = control.validate_reference_results(ROOT, inventory)
    assert tuple(references) == control.REFERENCE_TARGETS
    assert [
        references[target]["sampled"]["aggregate"]["accuracy"]["values"]
        for target in control.REFERENCE_TARGETS
    ] == [[0.556, 0.556, 0.558], [0.704, 0.726, 0.73]]


def test_array_layout_is_exactly_three_sampled_passes() -> None:
    assert [control.pass_spec(index) for index in range(12)] == [
        *(("sampled", repeat, shard) for repeat in range(3) for shard in range(4))
    ]
    with pytest.raises(ValueError):
        control.pass_spec(12)


def test_sampled_decoding_seed_and_dynamic_context_contract() -> None:
    assert generation.decoding() == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": -1,
    }
    assert generation.sample_seed(0, 0) == 1234
    assert generation.sample_seed(1, 0) == 1734
    assert generation.sample_seed(2, 499) == 2733
    with pytest.raises(ValueError):
        generation.sample_seed(3, 0)
    assert generation.effective_response_budget(1) == 32767
    assert generation.effective_response_budget(512) == 32256
    with pytest.raises(RuntimeError):
        generation.effective_response_budget(32768)


def test_prompt_stops_engine_and_scorer_contract() -> None:
    policy = json.loads((HANDOFF / "eval_policy.json").read_text())
    assert generation.rendered_messages("What is 1+1?") == [
        {
            "role": "user",
            "content": (
                "Please reason step by step, and put your final answer within "
                "\\boxed{}.\n\nProblem:\nWhat is 1+1?"
            ),
        }
    ]
    assert generation.STOP_TOKENS == {
        "<|endoftext|>": 151643,
        "<|im_end|>": 151645,
    }
    assert policy["stopping"]["accepted"] == [
        {"id": 151643, "token": "<|endoftext|>"},
        {"id": 151645, "token": "<|im_end|>"},
    ]
    assert policy["response_budget"]["formula"] == "32768 - rendered_prompt_tokens"
    assert policy["engine"]["gpu_memory_utilization"] == 0.9
    assert policy["scoring"] == {
        "cap_hits": "score saved response identically to natural stops",
        "candidate_selection": "last complete nonempty balanced boxed group only",
        "non_boxed_fallback": None,
        "scorer": "math500_last_boxed_symmetric_scorer_v6",
    }


def test_submission_shape_push_gate_and_job_local_caches_are_frozen() -> None:
    submit = (HANDOFF / "submit.sh").read_text()
    worker = (HANDOFF / "run_h200.sbatch").read_text()
    canary = (HANDOFF / "canary_h200.sbatch").read_text()
    control_source = (HANDOFF / "math500_eval.py").read_text()
    assert "--array=0-11" in submit
    assert '"eval_task_count": 120' in control_source
    assert "Evaluator commit is not pushed" in submit
    assert "origin/$BRANCH" in submit
    assert "qwen2_5:unmasked:sampled:0" in canary
    assert "TRITON_CACHE_DIR" in worker
    assert "VLLM_CACHE_ROOT" in worker
    assert "--time=02:00:00" in submit
    assert "--exclude=g3130" in submit


def test_sampled_statistic_is_sample_standard_deviation() -> None:
    result = control.mean_sd([0.50, 0.52, 0.54])
    assert result["mean"] == pytest.approx(0.52)
    assert result["sample_std"] == pytest.approx(0.02)
    with pytest.raises(ValueError):
        control.mean_sd([0.50, 0.52])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
