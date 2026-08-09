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
    assert inventory["runtime"] == {
        "cffi": "2.0.0",
        "math_verify": "0.9.0",
        "mistral_common": "1.11.7",
        "numpy": "2.2.6",
        "pycountry": "26.2.16",
        "pycparser": "3.0",
        "pydantic_extra_types": "2.11.1",
        "soundfile": "0.14.0",
        "soxr": "1.1.0",
        "tokenizers": "0.22.2",
        "torch": "2.8.0",
        "transformers": "4.57.6",
        "triton": "3.4.0",
        "vllm": "0.10.2",
    }


def test_runtime_contract_checks_versions_health_and_both_architectures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = control.load_inventory(ROOT)
    distribution_versions = {
        distribution: inventory["runtime"][name]
        for name, distribution in control.RUNTIME_DISTRIBUTIONS.items()
    }
    calls: list[list[str]] = []
    monkeypatch.setattr(
        control.importlib.metadata,
        "version",
        lambda distribution: distribution_versions[distribution],
    )
    monkeypatch.setattr(
        control.subprocess,
        "run",
        lambda command, check: calls.append(command),
    )

    assert control.validate_runtime_contract(inventory) == inventory["runtime"]
    assert calls[0][1:] == ["-m", "pip", "check"]
    assert "Qwen2ForCausalLM" in calls[1][2]
    assert "Qwen3ForCausalLM" in calls[1][2]


def test_canary_covers_both_model_families_and_decoding_modes() -> None:
    assert control.CANARY_CASES == (
        ("qwen2_5", "base_qwen2_5_3b", "greedy", 0),
        ("qwen3", "base_qwen3_4b", "sampled", 0),
    )
    assert control.canary_output(ROOT, "qwen3", "sampled", 0).parts[-4:] == (
        "canary",
        "qwen3",
        "sampled",
        "repeat_00",
    )


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
    canary = (HANDOFF / "canary_h200.sbatch").read_text()
    generator = (HANDOFF / "generate_math500.py").read_text()
    runtime = (HANDOFF / "runtime_python.sh").read_text()
    assert "--array=0-15" in submit
    assert "eval_task_count\": 144" in (HANDOFF / "math500_eval.py").read_text()
    assert "--time=02:00:00" in submit
    assert "--time=00:30:00" in submit
    assert "--exclude=g3130" in submit
    assert "validate-runtime" in submit
    assert "--mode \"$mode\" --repeat \"$repeat\"" in worker
    assert "#SBATCH --exclude=g3130" in worker
    assert "qwen2_5:base_qwen2_5_3b:greedy:0" in canary
    assert "qwen3:base_qwen3_4b:sampled:0" in canary
    assert "#SBATCH --time=00:30:00" in canary
    assert "#SBATCH --exclude=g3130" in canary
    assert 'STOP_TOKENS = {"<|endoftext|>": 151643, "<|im_end|>": 151645}' in generator
    assert "ignore_eos=False" in generator
    assert "python3.11" in runtime
    assert "qwen-math500-v1/venv/bin/python" in runtime
