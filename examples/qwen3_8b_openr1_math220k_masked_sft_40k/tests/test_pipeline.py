"""Cheap deterministic tests for masked-SFT token and weighting semantics."""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from examples.qwen3_8b_openr1_math220k_masked_sft_40k.build_weighted import (
    inverse_margin_weight,
    weighted_rows,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_40k.data_utils import (
    build_training_record,
    correctness_at,
    token_sha256,
    trace_structure_ok,
    user_content,
)
from examples.qwen3_8b_openr1_math220k_masked_sft_40k.weighted_sft_rollout import generate_rollout


class FakeTokenizer:
    """Minimal exact-boundary tokenizer used by the helper tests."""

    trace_ids = [31, 32, 33, 34]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        self.assert_true(enable_thinking)
        del tokenize
        if add_generation_prompt:
            return [10, 11]
        if messages[-1] != {"role": "assistant", "content": ""}:
            raise AssertionError(messages)
        return [10, 11, 151645, 12]

    @staticmethod
    def assert_true(value):
        if value is not True:
            raise AssertionError(value)

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise AssertionError("special tokens must be disabled")
        if text == "<|im_end|>\n":
            return [151645, 12]
        if "Known correct answer:" in text:
            return [20, 21, 22]
        return list(self.trace_ids)

    @staticmethod
    def convert_tokens_to_ids(token):
        return {"<|im_end|>": 151645, "<|endoftext|>": 151643}[token]


class FakeBuffer:
    """One-batch data buffer for rollout transport."""

    def __init__(self, sample):
        self.sample = sample

    def get_samples(self, count):
        if count != 1:
            raise AssertionError(count)
        return [[self.sample]]


class FakeSample:
    """Mutable Sample-compatible fixture."""

    def __init__(self):
        self.prompt = [10, 11, 31, 32, 151645, 12]
        self.metadata = {
            "loss_weights": [0.2, 1.0, 1.0, 0.0],
            "response_length": 4,
            "trace_id": "fixture",
        }
        self.tokens = []
        self.response_length = 0
        self.loss_mask = None
        self.reward = None


class PipelineTests(unittest.TestCase):
    """Assert the scientific decisions encoded in the runnable pipeline."""

    def test_inverse_margin_weight(self):
        self.assertEqual(inverse_margin_weight(0.0, 0.2), 1.0)
        self.assertEqual(inverse_margin_weight(0.5, 0.2), 1.0)
        self.assertAlmostEqual(inverse_margin_weight(-0.2, 0.2), 0.5)
        self.assertAlmostEqual(inverse_margin_weight(-0.8, 0.2), 0.2)

    def test_structure_and_correctness_priority(self):
        self.assertEqual(trace_structure_ok("<think>reason</think>answer"), (True, "ok"))
        self.assertEqual(trace_structure_ok("<think></think>answer")[1], "empty_think")
        self.assertEqual(trace_structure_ok("<think>reason</think>")[1], "empty_post_think")
        row = {"correctness_math_verify": [False], "correctness_llama": [True]}
        self.assertEqual(correctness_at(row, 0), (False, "correctness_math_verify"))

    def test_exact_training_terminal_and_weights(self):
        tokenizer = FakeTokenizer()
        selected = {
            "assistant_token_sha256": token_sha256(tokenizer.trace_ids),
            "assistant_trace": "<think>reason</think>answer",
            "correctness_source": "correctness_math_verify",
            "generation_index": 0,
            "is_correct": False,
            "problem": "1+1?",
            "problem_id": "p",
            "source_row_index": 0,
            "trace_id": "t",
            "wrong_index": 0,
        }
        record = build_training_record(tokenizer, selected, [0.25, 0.5, 0.75, 1.0], 32)
        self.assertEqual(record["input_ids"], [10, 11, 31, 32, 33, 34, 151645, 12])
        self.assertNotIn(151643, record["input_ids"])
        self.assertEqual(record["metadata"]["loss_weights"], [0.25, 0.5, 0.75, 1.0, 1.0, 0.0])
        self.assertEqual(record["metadata"]["response_length"], 6)

    def test_rollout_preserves_fractional_weights(self):
        sample = FakeSample()
        args = type("Args", (), {"rollout_global_dataset": True, "rollout_batch_size": 1})()
        result = generate_rollout(args, 0, FakeBuffer(sample), evaluation=False)
        self.assertIs(result[0][0], sample)
        self.assertEqual(sample.tokens, sample.prompt)
        self.assertEqual(sample.loss_mask, [0.2, 1.0, 1.0, 0.0])

    def test_weighted_builder_changes_weights_not_tokens(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            margin_dir = root / "margins"
            (margin_dir / "records").mkdir(parents=True)
            correct = {
                "input_ids": [1, 2, 151645, 3],
                "metadata": {
                    "assistant_token_count": 2,
                    "assistant_token_sha256": "correct-hash",
                    "is_correct": True,
                    "loss_weights": [1.0, 1.0, 1.0, 0.0],
                    "response_length": 4,
                    "trace_id": "correct",
                },
            }
            wrong = {
                "input_ids": [4, 5, 151645, 3],
                "metadata": {
                    "assistant_token_count": 2,
                    "assistant_token_sha256": "wrong-hash",
                    "is_correct": False,
                    "loss_weights": [1.0, 1.0, 1.0, 0.0],
                    "response_length": 4,
                    "trace_id": "wrong",
                    "wrong_index": 0,
                },
            }
            unmasked = root / "unmasked.jsonl"
            unmasked.write_text(json.dumps(correct) + "\n" + json.dumps(wrong) + "\n", encoding="utf-8")
            score = {
                "assistant_token_sha256": "wrong-hash",
                "delta_logprobs": [-0.2, 0.1],
                "trace_id": "wrong",
                "wrong_index": 0,
            }
            (margin_dir / "records/wrong_00000.json").write_text(json.dumps(score), encoding="utf-8")
            args = argparse.Namespace(
                expected_correct=1,
                expected_rows=2,
                expected_wrong=1,
                margin_dir=str(margin_dir),
                tau=0.2,
                unmasked_data=str(unmasked),
            )
            aggregate = {
                "wrong_fraction_below_0p1_count": 0,
                "wrong_fraction_below_0p5_count": 0,
                "wrong_fraction_equal_1_count": 0,
                "wrong_token_count": 0,
                "wrong_weight_max": float("-inf"),
                "wrong_weight_min": float("inf"),
                "wrong_weight_sum": 0.0,
            }
            rows = list(weighted_rows(args, aggregate))
            self.assertEqual(rows[0]["input_ids"], correct["input_ids"])
            self.assertEqual(rows[1]["input_ids"], wrong["input_ids"])
            self.assertEqual(rows[0]["metadata"]["loss_weights"], [1.0, 1.0, 1.0, 0.0])
            self.assertEqual(rows[1]["metadata"]["loss_weights"], [0.5, 1.0, 1.0, 0.0])
            self.assertEqual(aggregate["wrong_token_count"], 2)

    def test_prompt_contract(self):
        self.assertEqual(
            user_content("x"),
            "Please reason step by step, and put your final answer within \\boxed{}.\n\nProblem:\nx",
        )
        self.assertTrue(user_content("x", "2").endswith("Known correct answer: 2"))


if __name__ == "__main__":
    unittest.main()
