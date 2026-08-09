from __future__ import annotations

import json
from pathlib import Path

import pytest

from examples.qwen_correct_only_openr1_math220k_sft_eval_handoff import (
    generate_math500 as generation,
)
from examples.qwen_correct_only_openr1_math220k_sft_eval_handoff import (
    math500_eval as control,
)


ROOT = Path(__file__).resolve().parents[3]
HANDOFF = Path(__file__).resolve().parents[1]


def test_inventory_covers_nine_unique_targets_and_five_pairs() -> None:
    inventory = control.load_inventory(ROOT)
    assert tuple(item["id"] for item in inventory["checkpoints"]) == control.TARGETS
    assert len(control.TARGETS) == 9
    assert len(control.PAIRS) == 5
    assert [base for _, base, _ in control.PAIRS].count("base_qwen2_5_3b") == 2


def test_array_layout_is_one_greedy_and_three_sampled_passes() -> None:
    assert [control.pass_spec(index) for index in range(16)] == [
        *(('greedy', 0, shard) for shard in range(4)),
        *(('sampled', 0, shard) for shard in range(4)),
        *(('sampled', 1, shard) for shard in range(4)),
        *(('sampled', 2, shard) for shard in range(4)),
    ]
    with pytest.raises(ValueError):
        control.pass_spec(16)


def test_total_context_budget_is_prompt_dependent_without_margin() -> None:
    assert generation.effective_response_budget(1) == 32767
    assert generation.effective_response_budget(512) == 32256
    with pytest.raises(RuntimeError):
        generation.effective_response_budget(32768)


def test_decoding_and_seed_contract() -> None:
    assert generation.decoding("greedy") == {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
    }
    assert generation.decoding("sampled") == {
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": -1,
    }
    assert generation.sample_seed("greedy", 0, 499) == 0
    assert generation.sample_seed("sampled", 0, 0) == 1234
    assert generation.sample_seed("sampled", 1, 0) == 1734
    assert generation.sample_seed("sampled", 2, 499) == 2733
    with pytest.raises(ValueError):
        generation.sample_seed("sampled", 3, 0)


def test_prompt_and_exact_stop_tokens() -> None:
    messages = generation.rendered_messages("What is 1+1?")
    assert messages == [
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
    policy = json.loads((HANDOFF / "eval_policy.json").read_text())
    assert policy["stopping"]["accepted"] == [
        {"id": 151643, "token": "<|endoftext|>"},
        {"id": 151645, "token": "<|im_end|>"},
    ]
    assert policy["stopping"]["think_close_stopping"] is False
    assert policy["stopping"]["answer_based_stopping"] is False


def test_sampled_statistic_is_sample_standard_deviation() -> None:
    result = control.mean_sd([0.50, 0.52, 0.54])
    assert result["mean"] == pytest.approx(0.52)
    assert result["sample_std"] == pytest.approx(0.02)
    assert result["values"] == [0.50, 0.52, 0.54]
    with pytest.raises(ValueError):
        control.mean_sd([0.50, 0.52])


def test_submission_scripts_freeze_shape_and_stop_contract() -> None:
    submit = (HANDOFF / "submit.sh").read_text()
    worker = (HANDOFF / "run_h200.sbatch").read_text()
    generator = (HANDOFF / "generate_math500.py").read_text()
    runtime = (HANDOFF / "runtime_python.sh").read_text()
    assert "--array=0-15" in submit
    assert "eval_task_count\": 144" in (HANDOFF / "math500_eval.py").read_text()
    assert "--time=02:00:00" in submit
    assert "--mode \"$mode\" --repeat \"$repeat\"" in worker
    assert 'STOP_TOKENS = {"<|endoftext|>": 151643, "<|im_end|>": 151645}' in generator
    assert "ignore_eos=False" in generator
    assert "python3.11" in runtime
    assert "qwen-math500-v1/venv/bin/python" in runtime
