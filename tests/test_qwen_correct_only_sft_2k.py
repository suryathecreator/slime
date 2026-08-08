from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.qwen_correct_only_openr1_math220k_sft_2k.correct_only_sft_rollout import (
    generate_rollout,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.data_utils import (
    build_training_record,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.experiment import (
    RUN_ORDER,
    load_contract,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.prepare_data import (
    deterministic_order,
    eligible_ref,
)
from examples.qwen_correct_only_openr1_math220k_sft_2k.validate_schedule import (
    main as validate_schedule,
)


NUM_GPUS = 0
EXPERIMENT_DIR = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "qwen_correct_only_openr1_math220k_sft_2k"
)


class FakeTokenizer:
    def __init__(self, reject_long: bool = False):
        self.reject_long = reject_long

    def apply_chat_template(self, messages, **kwargs):
        del messages, kwargs
        return [151644, 77091, 198]

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        if text == "<|im_end|>\n":
            return [151645, 198]
        if self.reject_long and "LONG" in text:
            return [17] * 8193
        return [17] * 12

    def decode(self, tokens, skip_special_tokens=False):
        del skip_special_tokens
        if tokens == [151644, 77091, 198]:
            return "<|im_start|>assistant\n"
        return "x"


def selected() -> dict:
    return {
        "assistant_trace": "<think>work</think>\\boxed{1}",
        "generation_index": 0,
        "problem": "1+0?",
        "problem_id": "p1",
        "source_row_index": 4,
        "trace_id": "p1__row_0000004__gen_000",
    }


def test_contract_has_only_base_and_correct_only_with_shared_8k_selection():
    contract = load_contract()
    assert contract["variants"] == ["base", "correct_only"]
    assert contract["training"]["global_batch_size"] == 64
    assert contract["training"]["epochs"] == 4
    assert contract["training"]["optimizer_updates"] == 125
    assert contract["training"]["learning_rate"] == 5e-6
    assert contract["training"]["min_learning_rate"] == 1e-6
    eight_k = [key for key in RUN_ORDER if key.endswith("_8k")]
    assert len(eight_k) == 4
    assert {contract["runs"][key]["selection"] for key in eight_k} == {"8k_shared"}
    assert "same ordered 2000 trace IDs" in contract["selection"]["shared_8k_policy"]
    assert contract["runs"]["qwen2_5_3b_16k"]["max_sequence_length"] == 18432


def test_exact_token_record_fully_supervises_assistant_and_im_end():
    record = build_training_record(
        FakeTokenizer(), selected(), {}, 10240, 8192, "qwen2_5_3b"
    )
    metadata = record["metadata"]
    assert record["input_ids"][:3] == [151644, 77091, 198]
    assert record["input_ids"][-2:] == [151645, 198]
    assert metadata["assistant_token_count"] == 12
    assert metadata["response_length"] == 14
    assert metadata["loss_weights"] == [1.0] * 13 + [0.0]
    assert 151643 not in record["input_ids"]


def test_shared_eligibility_rejects_a_trace_that_fails_any_model_tokenizer():
    contract = load_contract()
    row = {
        "uuid": "problem-1",
        "problem": "problem",
        "answer": "1",
        "generations": [
            "<think>LONG work</think>\\boxed{1}",
            "<think>short work</think>\\boxed{1}",
        ],
        "correctness_math_verify": [True, True],
        "is_reasoning_complete": [True, True],
    }
    tokenizers = {
        "qwen2_5_3b": FakeTokenizer(reject_long=True),
        "qwen2_5_7b": FakeTokenizer(),
        "qwen3_4b": FakeTokenizer(),
        "qwen3_8b": FakeTokenizer(),
    }
    from collections import Counter

    ref = eligible_ref(
        row,
        0,
        "8k_shared",
        tuple(tokenizers),
        tokenizers,
        contract,
        8192,
        10240,
        42,
        Counter(),
    )
    assert ref is not None
    assert ref["generation_index"] == 1
    assert set(ref["per_model_lengths"]) == set(tokenizers)


def test_seeded_generation_order_is_reproducible():
    assert deterministic_order([0, 1, 2, 3], 42, 17, "8k_shared") == deterministic_order(
        [0, 1, 2, 3], 42, 17, "8k_shared"
    )
    assert deterministic_order([0, 1, 2, 3], 42, 17, "8k_shared") != deterministic_order(
        [0, 1, 2, 3], 43, 17, "8k_shared"
    )


def test_rollout_adapter_preserves_full_correct_only_weights():
    record = build_training_record(
        FakeTokenizer(), selected(), {}, 10240, 8192, "qwen2_5_3b"
    )
    sample = SimpleNamespace(prompt=record["input_ids"], metadata=record["metadata"])

    class Buffer:
        def get_samples(self, size):
            assert size == 1
            return [[sample]]

    args = SimpleNamespace(rollout_global_dataset=True, rollout_batch_size=1)
    result = generate_rollout(args, 0, Buffer())
    assert result == [[sample]]
    assert sample.tokens == record["input_ids"]
    assert sample.loss_mask == record["metadata"]["loss_weights"]
    assert sample.reward == 0


def test_four_epoch_schedule_resolves_to_125_global_updates(tmp_path, monkeypatch):
    data = tmp_path / "data.jsonl"
    data.write_text(
        "".join(json.dumps({"input_ids": list(range(32))}) + "\n" for _ in range(2000)),
        encoding="utf-8",
    )
    output = tmp_path / "schedule.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "validate_schedule.py",
            "--data",
            str(data),
            "--output",
            str(output),
            "--dp-size",
            "4",
            "--max-tokens-per-gpu",
            "32768",
        ],
    )
    validate_schedule()
    value = json.loads(output.read_text(encoding="utf-8"))
    assert value["global_examples"] == 8000
    assert value["optimizer_updates"] == 125
    assert value["examples_per_global_update"] == 64


def test_gpu_jobs_respect_eight_cpus_per_gpu_cluster_policy():
    for sbatch_path in EXPERIMENT_DIR.glob("*.sbatch"):
        text = sbatch_path.read_text(encoding="utf-8")
        gpu_match = re.search(r"^#SBATCH --gres=gpu:h200:(\d+)$", text, re.MULTILINE)
        if gpu_match is None:
            continue
        cpu_match = re.search(r"^#SBATCH --cpus-per-task=(\d+)$", text, re.MULTILINE)
        assert cpu_match is not None
        assert int(cpu_match.group(1)) <= 8 * int(gpu_match.group(1)), sbatch_path


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
